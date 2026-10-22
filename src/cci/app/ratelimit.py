"""Per-client token-bucket rate limiting for the HTTP API.

Clients are identified by a hash of their API key (never the key itself); requests without a
key are bucketed by client address, which also slows down key guessing. Health and metrics
endpoints are exempt so that probes keep working under load. Exceeding the limit returns
``429`` with ``Retry-After``.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Callable

EXEMPT_PREFIXES = ("/health", "/metrics")


class TokenBucketLimiter:
    def __init__(self, per_minute: int, burst: int | None = None, clock: Callable[[], float] = time.monotonic):
        if per_minute < 1:
            raise ValueError("per_minute must be >= 1")
        self.rate = per_minute / 60.0  # tokens per second
        self.capacity = float(burst or per_minute)
        self.clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # client -> (tokens, last refill)
        self._lock = threading.Lock()

    def take(self, client: str) -> float:
        """Consume one token. Returns 0 if allowed, else the seconds until a token is available."""
        now = self.clock()
        with self._lock:
            tokens, last = self._buckets.get(client, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[client] = (tokens - 1.0, now)
                return 0.0
            self._buckets[client] = (tokens, now)
            return (1.0 - tokens) / self.rate


def client_id(api_key: str | None, address: str | None) -> str:
    if api_key:
        return "key:" + hashlib.sha256(api_key.encode()).hexdigest()[:16]
    return "addr:" + (address or "unknown")


def retry_after(seconds: float) -> str:
    return str(max(1, math.ceil(seconds)))
