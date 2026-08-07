# Jarvis Brain — Roadmap

Populated from repository evidence. Extraction is complete; nothing is
claimed as cut over.

## COMPLETED

Extracted into this repository and passing here:

- Ollama provider and direct-Ollama local path
- OpenAI-compatible provider
- OmniRoute integration: config, discovery, registry, policy, circuit breaker,
  classifier, runtime status projection
- Route health convergence (`tests/test_omniroute_route_health_convergence.py`)
- Intelligence router and classifiers; routing observatory
- Model registry and model router
- Safe LLM service
- Prompt construction and plan/intent validators
- Brain orchestrator: execution graph, agent team builder, context builder,
  proposal parser, verification
- Agent lifecycle, message bus, stream service, swarm coordinator and runtime
- World intelligence engines
- Daily briefing composition

## ONGOING

- Nothing. Extraction is complete; the service awaits its HTTP application
  and real-traffic acceptance.

## BLOCKED

- **Conflict C4** — turn ownership. `app/conversation_runtime/` implements turn
  execution with its own `CancellationToken`. Voice owns the turn and mints
  `generation_id`; Brain owns request execution and its cancellation must be
  driven by Voice's `generation_id`, never minted independently.
- **Conflict C5** — `app/agents/` holds lifecycle and swarm (Brain) alongside
  `finance_agent`, `productivity_agent` and `smart_home_agent`, which drive
  Skills drivers. The three domain agents belong to Skills.

## NEXT

1. Stand up `jarvis_brain.app` with independent liveness/readiness. 23 test
   modules cannot run standalone until it exists.
2. Have Core bind the 19 currently-unbound ports (`jarvis_brain.ports`).
   Brain reports them honestly rather than pretending they work.
3. Resolve C4 (turn ownership), then C5 (agent split).
4. Prove route degradation makes Brain `degraded` and changes nothing else.

## FUTURE

**Adaptive Compute Manager** — hardware inventory; CPU/GPU/RAM/VRAM awareness;
backend discovery; workload reservation; model placement; migration; rollback;
future GPU adaptation.

**Local Model Foundry** — LoRA; QLoRA; datasets; evaluation; adapter registry;
model registry; human promotion; rollback.

## DEFERRED

Explicitly not part of this migration (spec §39):

- Adaptive Compute Manager and Local Model Foundry implementation
- Changing the current Ollama model
- Changing provider routing to make tests pass
- Adding a message broker
