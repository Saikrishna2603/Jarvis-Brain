# Jarvis brain — Status

**As of 2026-08-07.**

| | |
|---|---|
| Service version | 0.1.0 |
| Contract version | 1.0.0 |
| Migration status | `extracted` — not `cutover`, not `legacy_removed` |
| Runtime mode | `in_process` (during migration), `local_process` planned |
| Production authority | **Still the monolith.** `Jarvis-Brain` remains the running implementation and the rollback path. |

## Test evidence

74 test files extracted. Result: **371 passed, 1 failed, 3 skipped, 23 collection errors**.

Collection errors are test modules that import `app.main` to exercise the
monolith's FastAPI app. They cannot run standalone until `jarvis_brain.app`
exists. None is a behavioural regression.

## Boundaries proven

- Zero imports of any sibling service's implementation.
- Zero imports of `jarvis_core`.
- Zero residual `app.*` imports from the monolith.
- Depends only on the Core-published `jarvis_contracts` and `jarvis_platform`.

Enforced by `jarvis-core/tests/architecture/test_workspace_integration.py`.

## What is not proven

- The service has never run as its own process; `jarvis_brain.app` does not exist.
- No real traffic. Core drives it through stub ports in tests only.
- Cross-service wiring is declared but unexercised end to end.
