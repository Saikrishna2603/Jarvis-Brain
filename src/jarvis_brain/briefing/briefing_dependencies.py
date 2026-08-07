"""The briefing's dependency root.

In the monolith this module built its singletons at import time, reaching into
`app.api.brain_routes` and `app.api.integration_routes` for objects the HTTP
layer happened to own. That works in one process and nowhere else: importing
this module started a Brain engine as a side effect, and the route module and
this module each thought they owned the graph.

Extraction keeps the *intent* — one shared graph, because there is one Jarvis
and the briefing must not read a private copy of the world — and drops the
import-time construction. Core builds the graph once and binds it here.

`daily_briefing_service` is still a module attribute so existing readers are
unchanged; it is simply `None` until wired, which is the same honest outcome
the Voice briefing provider already handles by speaking alone.
"""

from __future__ import annotations

from typing import Any

from jarvis_brain.briefing.briefing_factory import create_daily_briefing_service
from jarvis_brain.briefing.briefing_store import BriefingStore

#: Bound by :func:`build_briefing_dependencies`. ``None`` until then.
daily_briefing_service: Any | None = None

plugin_registry: Any | None = None
skill_catalog: Any | None = None
skill_registry: Any | None = None
source_registry: Any | None = None

#: Brain-owned and dependency-free, so it is built eagerly and stays a real
#: object. Only the collaborators that belong to *other* services wait for
#: binding — making this one lazy too would have broken every test that clears
#: briefing history between cases, for no isolation benefit.
briefing_store: BriefingStore = BriefingStore()


def build_briefing_dependencies(
    *,
    brain_engine: Any,
    agent_lifecycle_manager: Any,
    integration_registry: Any,
    plugin_registry_factory: Any,
    skill_catalog_factory: Any,
    skill_registry_factory: Any,
    source_registry_factory: Any,
) -> Any:
    """Construct the shared briefing graph and publish it on this module.

    Every collaborator is passed in rather than imported. Skills owns the skill
    registry and catalog, Memory owns the source registry, and Core owns the
    plugin and integration registries — Brain assembles the briefing from them
    but owns none of them.
    """

    global daily_briefing_service, plugin_registry, skill_catalog
    global skill_registry, source_registry

    plugin_registry = plugin_registry_factory()
    skill_catalog = skill_catalog_factory()
    skill_registry = skill_registry_factory(
        plugin_registry=plugin_registry,
        catalog=skill_catalog,
        approval_manager=brain_engine.approval_manager,
    )
    source_registry = source_registry_factory()

    daily_briefing_service = create_daily_briefing_service(
        brain_engine=brain_engine,
        agent_lifecycle_manager=agent_lifecycle_manager,
        skill_registry=skill_registry,
        integration_registry=integration_registry,
        source_registry=source_registry,
        briefing_store=briefing_store,
    )
    return daily_briefing_service


def reset_briefing_dependencies() -> None:
    """Drop the graph. For tests and shutdown."""

    global daily_briefing_service, plugin_registry, skill_catalog
    global skill_registry, source_registry

    daily_briefing_service = None
    plugin_registry = None
    skill_catalog = None
    skill_registry = None
    source_registry = None
    briefing_store.clear()
