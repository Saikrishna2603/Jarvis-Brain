import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jarvis_brain.llm.omniroute.config import OmniRouteSettings
from jarvis_brain.llm.omniroute.schemas import OmniRouteRoute, RouteLocality


class _RouteDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: int = Field(ge=1)
    source_commit: str
    routes: list[OmniRouteRoute] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_routes(self) -> "_RouteDocument":
        route_ids = [route.route_id for route in self.routes]
        if len(route_ids) != len(set(route_ids)):
            raise ValueError("OmniRoute route IDs must be unique.")
        for route in self.routes:
            unknown_fallbacks = set(route.fallback_route_ids) - set(route_ids)
            if unknown_fallbacks:
                raise ValueError(
                    f"Route {route.route_id} references unknown fallback routes."
                )
            for fallback_id in route.fallback_route_ids:
                fallback = next(item for item in self.routes if item.route_id == fallback_id)
                if (
                    fallback.privacy_class != route.privacy_class
                    or fallback.locality != route.locality
                ):
                    raise ValueError(
                        "Fallback routes cannot cross privacy or locality classes."
                    )
        return self


class OmniRouteRouteRegistry:
    """Validated, server-controlled allowlist of exact gateway routes."""

    def __init__(
        self,
        settings: OmniRouteSettings | None = None,
        *,
        routes: list[OmniRouteRoute] | None = None,
        source_commit: str = "operator-supplied",
    ) -> None:
        self.settings = settings or OmniRouteSettings()
        if routes is None:
            document = self._load(self.settings.route_registry_path)
            self.source_commit = document.source_commit
            routes = document.routes
        else:
            document = _RouteDocument(
                version=1, source_commit=source_commit, routes=routes
            )
            self.source_commit = document.source_commit
        self._routes = {route.route_id: route for route in routes}

    @staticmethod
    def _load(path: Path) -> _RouteDocument:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return _RouteDocument(version=1, source_commit="missing", routes=[])
        except json.JSONDecodeError as exc:
            raise ValueError("OmniRoute registry must be valid JSON-compatible YAML.") from exc
        return _RouteDocument.model_validate(raw)

    def list_routes(self) -> list[OmniRouteRoute]:
        return [self._routes[key] for key in sorted(self._routes)]

    def get(self, route_id: str) -> OmniRouteRoute | None:
        return self._routes.get(route_id)

    def find_model(self, model_id: str) -> OmniRouteRoute | None:
        matches = [route for route in self._routes.values() if route.model_id == model_id]
        return sorted(matches, key=lambda route: route.route_id)[0] if matches else None

    def approved_routes(self) -> list[OmniRouteRoute]:
        return [route for route in self.list_routes() if self.is_operator_approved(route)]

    def is_operator_approved(self, route: OmniRouteRoute) -> bool:
        if not route.enabled or route.locality == RouteLocality.UNKNOWN:
            return False
        if route.terms_status != "approved":
            return False
        if self.settings.allowed_routes and route.route_id not in self.settings.allowed_routes:
            return False
        if self.settings.allowed_providers and route.provider_id not in self.settings.allowed_providers:
            return False
        if self.settings.allowed_models and route.model_id not in self.settings.allowed_models:
            return False
        return True

    def safe_summary(self) -> dict[str, object]:
        routes = self.list_routes()
        return {
            "registry_source_commit": self.source_commit,
            "configured_route_count": len(routes),
            "approved_route_count": len(self.approved_routes()),
            "enabled_route_ids": [route.route_id for route in self.approved_routes()],
        }
