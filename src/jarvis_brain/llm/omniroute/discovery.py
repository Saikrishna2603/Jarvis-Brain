import json
import urllib.error
import urllib.request

from jarvis_brain.llm.omniroute.config import OmniRouteSettings
from jarvis_brain.llm.omniroute.provider import _RejectRedirects
from jarvis_brain.llm.omniroute.registry import OmniRouteRouteRegistry
from jarvis_brain.llm.omniroute.schemas import CatalogSnapshot, DiscoveredModel, RouteHealthState


class OmniRouteDiscoveryClient:
    """Read-only authenticated catalog discovery; discovery never grants approval."""

    def __init__(
        self,
        settings: OmniRouteSettings | None = None,
        registry: OmniRouteRouteRegistry | None = None,
    ) -> None:
        self.settings = settings or OmniRouteSettings()
        self.registry = registry or OmniRouteRouteRegistry(self.settings)
        self._opener = urllib.request.build_opener(_RejectRedirects())

    def fetch(self) -> CatalogSnapshot:
        if not self.settings.enabled:
            return CatalogSnapshot(state=RouteHealthState.DISABLED)
        if not self.settings.api_key:
            return CatalogSnapshot(
                state=RouteHealthState.MISCONFIGURED,
                safe_error="Gateway authentication is not configured.",
            )
        request = urllib.request.Request(
            f"{self.settings.base_url}/models",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with self._opener.open(
                request, timeout=self.settings.connect_timeout_seconds
            ) as response:
                try:
                    raw = response.read(self.settings.max_response_bytes + 1)
                except TypeError:
                    raw = response.read()
            if len(raw) > self.settings.max_response_bytes:
                raise ValueError("Catalog response exceeded the configured limit.")
            payload = json.loads(raw.decode("utf-8"))
            return self._normalize(payload)
        except urllib.error.HTTPError as exc:
            state = RouteHealthState.UNAUTHORIZED if exc.code in {401, 403} else RouteHealthState.CATALOG_UNAVAILABLE
            return CatalogSnapshot(state=state, safe_error=f"Catalog returned HTTP {exc.code}.")
        except (urllib.error.URLError, TimeoutError, OSError):
            return CatalogSnapshot(
                state=RouteHealthState.UNREACHABLE,
                safe_error="The loopback gateway is unreachable.",
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError):
            return CatalogSnapshot(
                state=RouteHealthState.CATALOG_UNAVAILABLE,
                safe_error="The model catalog was invalid.",
            )

    def _normalize(self, payload: object) -> CatalogSnapshot:
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("Unexpected catalog shape.")
        seen: set[str] = set()
        models: list[DiscoveredModel] = []
        duplicates = 0
        for item in payload["data"]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                continue
            model_id = item["id"].strip()
            if not model_id:
                continue
            if model_id in seen:
                duplicates += 1
                continue
            seen.add(model_id)
            provider_id = model_id.split("/", 1)[0] if "/" in model_id else None
            models.append(
                DiscoveredModel(
                    model_id=model_id,
                    provider_id=provider_id,
                    owned_by=item.get("owned_by") if isinstance(item.get("owned_by"), str) else None,
                )
            )
        approved = {route.model_id for route in self.registry.approved_routes()}
        return CatalogSnapshot(
            state=RouteHealthState.READY,
            models=sorted(models, key=lambda item: item.model_id),
            duplicate_count=duplicates,
            approved_present=sorted(approved & seen),
            approved_missing=sorted(approved - seen),
            discovered_unapproved=sorted(seen - approved),
        )
