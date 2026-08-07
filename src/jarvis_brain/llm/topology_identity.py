from urllib.parse import quote


DIRECT_ROUTE_SEGMENT = "direct"


def provider_topology_id(*, gateway: str, provider_id: str) -> str:
    """Return a stable presentation identity without changing provider authority."""
    return _join(gateway, provider_id)


def model_topology_id(*, gateway: str, provider_id: str, model_id: str) -> str:
    """Keep identical model labels distinct across routing transports."""
    return _join(gateway, provider_id, model_id)


def route_topology_id(
    *,
    gateway: str,
    provider_id: str,
    model_id: str,
    route_id: str | None,
) -> str:
    """Return a collision-safe topology edge identity for one configured route."""
    return _join(
        gateway,
        provider_id,
        model_id,
        route_id or DIRECT_ROUTE_SEGMENT,
    )


def _join(*parts: str) -> str:
    return ":".join(quote(part.strip(), safe="-._~") for part in parts)
