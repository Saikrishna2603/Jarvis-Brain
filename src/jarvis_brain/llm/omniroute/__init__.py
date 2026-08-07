"""Optional OmniRoute gateway integration owned by Jarvis routing policy."""

from jarvis_brain.llm.omniroute.config import OmniRouteSettings
from jarvis_brain.llm.omniroute.discovery import OmniRouteDiscoveryClient
from jarvis_brain.llm.omniroute.policy import OmniRouteSelectionPolicy
from jarvis_brain.llm.omniroute.provider import OmniRouteGatewayProvider
from jarvis_brain.llm.omniroute.registry import OmniRouteRouteRegistry

__all__ = [
    "OmniRouteDiscoveryClient",
    "OmniRouteGatewayProvider",
    "OmniRouteRouteRegistry",
    "OmniRouteSelectionPolicy",
    "OmniRouteSettings",
]
