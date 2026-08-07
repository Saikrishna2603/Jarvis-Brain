# Jarvis Brain — Migration Record

**Status: `extracted`.** Source: `Jarvis-Brain` at `b8a8a86`.

## Behaviour preservation

No logic was rewritten. The only edits were:

1. Package rename (`app.<domain>` → `jarvis_brain`).
2. Shared infrastructure imports redirected to the Core-published
   `jarvis_platform` SDK.
3. Cross-service imports replaced by protocols in `src/jarvis_brain/ports`.

Class bodies, thresholds and constants are byte-identical to the source.

## Ports

Where the monolith injected a concrete class from another domain, the type
annotation became a protocol and the call site was left alone. Where it
*constructed* a default, construction stayed put and only the source became
bindable — Core binds the real implementation at registration.

Unbound defaults were chosen to match what the monolith already did when the
dependency was missing, so an unwired service degrades the way it always did.

## Not deleted

`Jarvis-Brain` is untouched. It remains the production authority and the
rollback path (spec §28). This service is `extracted`, never
`legacy_removed`.

## Remaining work

1. Build `jarvis_brain.app` — the routes still live in the monolith's
   `app/api/`, which is why some test modules cannot run standalone.
2. Have Core bind this service's ports to the real sibling implementations.
3. Real-traffic acceptance before any cutover.

See `jarvis-core/docs/migration/MIGRATION_LEDGER.md` for the full mapping.
