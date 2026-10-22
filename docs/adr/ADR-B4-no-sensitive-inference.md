# ADR-B4: Routing and recommendations use only stated intent and explicit preferences

Status: accepted (2026-10-22)

## Decision
* Queue routing is a deterministic, reviewable rule on the intent and on facts the customer
  stated (e.g. reported injuries -> priority claims queue).
* "Recommendations" are next actions derived from the stated intent and from preferences the
  customer explicitly said (channel, callback window). Nothing is inferred about the person
  (no emotion/psychological profiling, no demographic or health inference from voice or
  text, no automated insurance decisions such as claim approval or pricing).
* The extraction schema has no field for sensitive attributes; the API rejects unknown
  fields on correction (tested).

## Rationale
Insurance and banking decisions about people are high-risk; inferred attributes are hard to
justify, audit or consent to. Explicit statements are auditable in the transcript.
