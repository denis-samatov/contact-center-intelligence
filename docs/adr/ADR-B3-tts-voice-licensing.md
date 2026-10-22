# ADR-B3: Use only CC0-trained TTS voices for published audio

Status: accepted (2026-10-22)

## Observation
The MODEL_CARD files of the Piper ru_RU voices state the training-dataset licenses:
`denis` and `dmitri` - CC0 (NabuCasa voice-datasets); `irina` - "Unknown" (RHVoice);
`ruslan` - CC BY-NC-SA 4.0 (non-commercial).

## Decision
Render the corpus with `dmitri` (operator) and `denis` (client) only. Do not publish audio
from `irina` or `ruslan`. Generated audio is not committed to Git; it is reproducible with
`cci render`.

## Consequence
Both speakers are male, so diarization cannot rely on a gender difference (closer to the
hard case in real contact centers).
