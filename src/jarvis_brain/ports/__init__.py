"""Ports — what Brain needs but does not own (spec §5, §23).

Brain reasons. It does not remember (Memory), do things (Skills), see
(Perception), speak (Voice), or know the shape of the system (Core). Everything
it needs from those services arrives through this module and nowhere else.

Each name here is a structural stand-in for the concrete class the monolith
injected. They are typing aliases rather than reimplementations: Brain's call
sites are unchanged, and Core binds the real object during registration. That
is what lets Phase D extract Brain without rewriting it.

The one place this module does more than alias is `speech_style`. See below.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

class PortNotBound(RuntimeError):
    """A port was used before Core bound a real implementation to it."""


class _Unbound:
    """Stands in for a collaborator Core has not bound yet.

    It **constructs** without complaint and **fails on first use**. That split
    is deliberate. `BrainEngine.__init__` eagerly builds a dozen collaborators,
    so raising at construction would make an unwired Brain impossible to even
    instantiate — including in tests that never touch these paths. Raising on
    first *use* keeps the object graph buildable while guaranteeing nothing
    silently pretends to be a memory store or a skill registry.
    """

    __slots__ = ("_port_name",)

    def __init__(self, port_name: str) -> None:
        object.__setattr__(self, "_port_name", port_name)

    def _fail(self) -> None:
        raise PortNotBound(
            f"{self._port_name} is not bound. Jarvis Core binds this during "
            f"Brain registration; see jarvis_brain.ports.bind()."
        )

    def __getattr__(self, item: str) -> Any:
        self._fail()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._fail()

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<unbound port {self._port_name}>"


class _Port:
    """A port name that is both constructible and usable as an annotation.

    Call it to build the collaborator (via the bound factory, or an `_Unbound`
    placeholder). `X | None` works because `__or__` collapses to `Any`, which
    is what the surrounding annotations mean anyway.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        factory = _factories.get(self._name)
        if factory is None:
            return _Unbound(self._name)
        return factory(*args, **kwargs)

    # Annotation support: `Port | None`, `None | Port`.
    def __or__(self, other: Any) -> Any:
        return Any

    def __ror__(self, other: Any) -> Any:
        return Any

    def __repr__(self) -> str:
        return f"<port {self._name}>"


# --- Memory ---------------------------------------------------------------

SourceRegistry = _Port("SourceRegistry")
TaskMemoryManager = _Port("TaskMemoryManager")
AuditManager = _Port("AuditManager")
ContextManager = _Port("ContextManager")
PlanMemoryManager = _Port("PlanMemoryManager")
SemanticMemoryManager = _Port("SemanticMemoryManager")
WorldEventMemoryManager = _Port("WorldEventMemoryManager")
WorldEventRepository = _Port("WorldEventRepository")
RetrievalRegistry = _Port("RetrievalRegistry")
GuidanceEngine = _Port("GuidanceEngine")
KnowledgeGapDetector = _Port("KnowledgeGapDetector")
RetrievalPlanner = _Port("RetrievalPlanner")
LLMAssistedGuidanceEngine = _Port("LLMAssistedGuidanceEngine")

# --- Skills ---------------------------------------------------------------

SkillRegistry = _Port("SkillRegistry")
SkillCatalog = _Port("SkillCatalog")
WorldDataDriver = _Port("WorldDataDriver")

#: Skills' approval gate shapes. Brain reads the verdict; Skills decides it.
GateResult = Any
SkillApprovalKind = Any
MockTaskSystem = _Port("MockTaskSystem")

# --- Core -----------------------------------------------------------------

PluginRegistry = _Port("PluginRegistry")
SystemStatusHandler = _Port("SystemStatusHandler")
IntegrationRegistry = _Port("IntegrationRegistry")


def unbound_ports() -> tuple[str, ...]:
    """Every port with no implementation bound.

    Brain's readiness reports `degraded` while this is non-empty, so an
    unwired collaborator is visible in the status projection rather than
    surfacing as a mysterious failure mid-request.
    """

    names = [
        value._name
        for value in globals().values()
        if isinstance(value, _Port)
    ]
    return tuple(sorted(name for name in names if name not in _factories))


@runtime_checkable
class MemoryPort(Protocol):
    """Coarse Memory contract, for code that only needs read/write."""

    def context(self, *args: Any, **kwargs: Any) -> Any: ...
    def write(self, *args: Any, **kwargs: Any) -> Any: ...


# --- Bindable factories ---------------------------------------------------
#
# Where the monolith constructed a default (`X()` or `create_default_X()`),
# construction stays where it was and only the source becomes bindable.

_factories: dict[str, Callable[..., Any]] = {}


def bind(name: str, factory: Callable[..., Any]) -> None:
    """Core binds a concrete factory during Brain registration."""

    _factories[name] = factory


def reset_wiring() -> None:
    _factories.clear()


def _make(name: str, *args: Any, **kwargs: Any) -> Any:
    """Build via the bound factory, or hand back an unbound placeholder.

    Same construct-now / fail-on-use split as `_Port`, and for the same reason:
    these factories are called at *import* time by the route modules, which
    build a `BrainEngine` at module scope. Raising here would make the routes
    unimportable with no siblings wired, and Brain would be unable to serve
    even `/health/*`.
    """

    factory = _factories.get(name)
    if factory is None:
        return _Unbound(name)
    return factory(*args, **kwargs)


def create_default_retrieval_registry(*args: Any, **kwargs: Any) -> Any:
    return _make("retrieval_registry", *args, **kwargs)


def create_default_tool_registry(*args: Any, **kwargs: Any) -> Any:
    return _make("tool_registry", *args, **kwargs)


def create_llm_assisted_guidance_engine(*args: Any, **kwargs: Any) -> Any:
    """Memory's LLM-assisted guidance engine.

    Memory owns knowledge acquisition and guidance; Brain supplies the
    inference it uses. Brain consumes the result and constructs none of it.
    """

    return _make("llm_assisted_guidance_engine", *args, **kwargs)


# --- Voice: the C3 seam ---------------------------------------------------
#
# `brain_engine.py` and `conversation/manager.py` imported
# `app.voice.identity.policy` and `app.voice.identity.profile` directly to
# decide how Jarvis should sound. That is backwards: Voice owns Voice identity,
# and Brain importing it made the two services mutually dependent.
#
# Voice now supplies speech style as *data* on `brain.requested.speech_style`
# (see `jarvis_contracts.events.brain.BrainRequested`). These accessors exist
# so the existing call sites keep working while that data path is wired.

#: Voice's speech-style policy class. Bound by the composition root to Voice's
#: real implementation; unbound it yields a placeholder that fails on use
#: rather than silently giving Jarvis a different voice.
JarvisSpeechStylePolicy = _Port("JarvisSpeechStylePolicy")

_speech_style: Any | None = None


def bind_speech_style(style: Any) -> None:
    """Record the voice identity Voice supplied.

    Voice owns how Jarvis sounds. Brain receives it as data and never reads
    Voice's files — that reverse import was conflict C3.
    """

    global _speech_style
    _speech_style = style


def load_jarvis_voice_identity(*args: Any, **kwargs: Any) -> Any:
    """Return the identity Voice supplied, or None.

    None is a supported answer: the style policy falls back to its neutral
    default, which is exactly what the monolith did when the profile could not
    be read.
    """

    return _speech_style


def create_tts_provider(*args: Any, **kwargs: Any) -> Any:
    """Voice owns TTS. Brain has no business constructing one."""

    raise RuntimeError(
        "Brain must not create a TTS provider. Voice owns synthesis; send the "
        "text back through Core and let Voice speak it."
    )


LocalMacTTSProvider = Any


__all__ = [name for name in dir() if not name.startswith("_")]
