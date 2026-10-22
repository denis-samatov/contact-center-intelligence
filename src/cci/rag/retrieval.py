"""Knowledge-base retrieval: section chunks, lemmatized BM25, dense embeddings, RRF fusion.

* Chunks are ``##`` sections of the Markdown KB, id ``<file stem>#<section number>``.
* BM25 runs over lemmas (pymorphy3) so that "заливе" matches "залив".
* Dense retrieval uses an OpenAI-compatible ``/embeddings`` endpoint (Qwen3-Embedding-0.6B
  served by llama.cpp by default); queries get the model's instruction prefix.
* Hybrid ranking = reciprocal rank fusion (k = 60) of the two lists.

The in-memory index is used for evaluation; the API persists the same vectors in
PostgreSQL/pgvector (``cci.app.db``).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

QUERY_INSTRUCTION = "Instruct: Given a customer question, retrieve the insurance rule that answers it\nQuery: "
_WORD = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
STOP = set(
    [
        "и",
        "в",
        "во",
        "не",
        "что",
        "он",
        "на",
        "я",
        "с",
        "со",
        "как",
        "а",
        "то",
        "все",
        "она",
        "так",
        "его",
        "но",
        "да",
        "ты",
        "к",
        "у",
        "же",
        "вы",
        "за",
        "бы",
        "по",
        "только",
        "ее",
        "мне",
        "было",
        "вот",
        "от",
        "меня",
        "еще",
        "нет",
        "о",
        "из",
        "ему",
        "теперь",
        "когда",
        "даже",
        "ну",
        "вдруг",
        "ли",
        "если",
        "уже",
        "или",
        "ни",
        "быть",
        "был",
        "него",
        "до",
        "вас",
        "нибудь",
        "опять",
        "уж",
        "вам",
        "ведь",
        "там",
        "потом",
        "себя",
        "ничего",
        "ей",
        "может",
        "они",
        "тут",
        "где",
        "есть",
        "надо",
        "ней",
        "для",
        "мы",
        "тебя",
        "их",
        "чем",
        "была",
        "сам",
        "чтоб",
        "без",
        "будто",
        "чего",
        "раз",
        "тоже",
        "себе",
        "под",
        "будет",
        "ж",
        "тогда",
        "кто",
        "этот",
        "того",
        "потому",
        "этого",
        "какой",
        "совсем",
        "ним",
        "здесь",
        "этом",
        "один",
        "почти",
        "мой",
        "тем",
        "чтобы",
        "нее",
        "сейчас",
        "были",
        "куда",
        "зачем",
        "всех",
        "никогда",
        "можно",
        "при",
        "наконец",
        "два",
        "об",
        "другой",
        "хоть",
        "после",
        "над",
        "больше",
        "тот",
        "через",
        "эти",
        "нас",
        "про",
        "всего",
        "них",
        "какая",
        "много",
        "разве",
        "три",
        "эту",
        "моя",
        "впрочем",
        "хорошо",
        "свою",
        "этой",
        "перед",
        "иногда",
        "лучше",
        "чуть",
        "том",
        "нельзя",
        "такой",
        "им",
        "более",
        "всегда",
        "конечно",
        "всю",
        "между",
    ]
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    title: str
    text: str


def load_kb(kb_dir: Path) -> list[Chunk]:
    chunks = []
    for path in sorted(kb_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        doc_title, current, buf = "", None, []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                doc_title = line[2:].strip()
            elif line.startswith("## "):
                if current:
                    chunks.append(Chunk(current[0], f"{doc_title}: {current[1]}", " ".join(buf).strip()))
                num, _, title = line[3:].partition(". ")
                current, buf = (f"{path.stem}#{num.strip()}", title.strip()), []
            elif line.strip():
                buf.append(line.strip())
        if current:
            chunks.append(Chunk(current[0], f"{doc_title}: {current[1]}", " ".join(buf).strip()))
    return chunks


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=50000)
def lemma(word: str) -> str:
    return _morph().parse(word)[0].normal_form


def lemmas(text: str) -> list[str]:
    return [lemma(w) for w in (m.group().lower().replace("ё", "е") for m in _WORD.finditer(text)) if w not in STOP]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(map(len, docs)) / len(docs)
        df = Counter(t for d in docs for t in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs, strict=True)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avgdl)
            out[i] = sum(self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in query if t in tf)
        return out


class HybridIndex:
    def __init__(self, chunks: list[Chunk], embed=None):
        """``embed``: callable(list[str]) -> list[list[float]], or None for BM25 only."""
        self.chunks = chunks
        self.bm25 = BM25([lemmas(c.title + " " + c.text) for c in chunks])
        self.embed = embed
        self.vectors = None
        if embed is not None:
            v = np.asarray(embed([c.title + "\n" + c.text for c in chunks]), dtype=np.float32)
            self.vectors = v / np.linalg.norm(v, axis=1, keepdims=True)

    def rank_bm25(self, query: str) -> list[int]:
        s = self.bm25.scores(lemmas(query))
        return [int(i) for i in np.argsort(-s, kind="stable") if s[i] > 0]

    def rank_dense(self, query: str) -> list[int]:
        if self.vectors is None:
            return []
        q = np.asarray(self.embed([QUERY_INSTRUCTION + query])[0], dtype=np.float32)
        q /= np.linalg.norm(q)
        return [int(i) for i in np.argsort(-(self.vectors @ q), kind="stable")]

    def search(self, query: str, k: int = 3, mode: str = "hybrid") -> list[Chunk]:
        if mode == "bm25":
            order = self.rank_bm25(query)
        elif mode == "dense":
            order = self.rank_dense(query)
        else:
            rrf: Counter = Counter()
            for ranking in (self.rank_bm25(query), self.rank_dense(query)):
                for r, i in enumerate(ranking):
                    rrf[i] += 1.0 / (60 + r + 1)
            order = [i for i, _ in rrf.most_common()]
        return [self.chunks[i] for i in order[:k]]
