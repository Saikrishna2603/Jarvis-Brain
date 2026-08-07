# Jarvis Brain

Reasoning and model routing.

## What this service owns

LLM providers (Ollama, OpenAI-compatible) · OmniRoute integration, circuit
breaking, discovery and route health · intelligence routing and model selection
· prompt construction · Brain request execution · planning and intent
resolution · the agent lifecycle and swarm runtime · world intelligence ·
briefing composition.

## What this service does NOT own

Microphone, STT, VAD, wake, TTS, playback, turn ownership, `generation_id`
(Voice) · memory persistence and retrieval (Memory) · vision (Perception) ·
actions (Skills) · global Jarvis status, registry, frontend (Core).

**Brain health is not Jarvis health.** A degraded route makes Brain `degraded`
and nothing else. It must never reach Voice, and it must never make Core
offline.

## How it starts

```bash
python -m jarvis_brain.app
```

Not yet implemented — Brain extraction is Phase D. Today the monolith at
`Jarvis-Brain` *is* this service.

## How it registers

Core reads `service.yaml` and binds the Brain port. Brain reports its own
readiness; a provider that cannot be reached yields `degraded`, not `offline` —
Brain is running, it just cannot answer well right now.

## Contracts published

`brain.started` · `brain.completed` · `brain.failed` · `brain.cancelled`

`BrainCancelled.generation_id` is **optional and never minted here**. Voice owns
generation identity; Brain only echoes what it was given.

## Contracts consumed

`brain.requested` — carrying `prompt`, optional `speech_style` (supplied as
*data* by Voice; Brain must never import Voice identity code) and context.
Optionally `memory.context.completed`.

## Readiness

`ready` when at least one provider answers. `degraded` when providers are
configured but unreachable. `offline` when the service is not running.

## Tests

```bash
pytest -q
```

Currently the Brain tests live in the monolith. See `MIGRATION.md`.

## Current P0 / P1

- **P1** — nothing is extracted yet. This repository is a skeleton with a
  contract, a manifest and a roadmap; the implementation is still the monolith.
