# Jarvis brain — Service Contract

Contract version **1.0.0**. Schemas live in `jarvis_contracts.events.brain`.

See `packages/jarvis_contracts/src/jarvis_contracts/events/brain.py` in
`jarvis-core` for the authoritative payload definitions; this file must not
restate them, only the guarantees around them.

## Identity guarantees

Every envelope carries `trace_id` and `correlation_id`. Voice-owned
identifiers (`session_id`, `conversation_epoch`, `turn_id`,
`generation_id`) are propagated verbatim when present. This service **never
mints them**.

## Independence

This service declares **no required dependencies**. Its unavailability is a
service-level fact and must never make Core offline, nor stop another service
(spec §9, §30.13).

## Versioning

If this service declares a contract version Core cannot speak, Core reports
`incompatible` — never `offline`. See `CONTRACT_VERSIONING.md` in
`jarvis-core`.
