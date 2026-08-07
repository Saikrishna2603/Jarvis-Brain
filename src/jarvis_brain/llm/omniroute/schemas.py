from datetime import datetime
from enum import Enum
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from jarvis_platform.schemas.common import utc_now


class RouteLocality(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class RoutePrivacy(str, Enum):
    LOCAL_ONLY = "local_only"
    CLOUD_ALLOWED = "cloud_allowed"
    PUBLIC_ONLY = "public_only"


class RouteHealthState(str, Enum):
    DISABLED = "disabled"
    STARTING = "starting"
    UNREACHABLE = "unreachable"
    UNAUTHORIZED = "unauthorized"
    MISCONFIGURED = "misconfigured"
    CATALOG_UNAVAILABLE = "catalog_unavailable"
    ROUTE_MISSING = "route_missing"
    ROUTE_UNCLASSIFIED = "route_unclassified"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    POLICY_BLOCKED = "policy_blocked"
    READY = "ready"
    DEGRADED = "degraded"


class TaskCategory(str, Enum):
    SIMPLE_CONVERSATION = "simple_conversation"
    VOICE_CONVERSATION = "voice_conversation"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    CODING = "coding"
    CODE_REVIEW = "code_review"
    DEBUGGING = "debugging"
    SOFTWARE_ARCHITECTURE = "software_architecture"
    COMPLEX_REASONING = "complex_reasoning"
    LONG_CONTEXT = "long_context"
    STRUCTURED_EXTRACTION = "structured_extraction"
    RESEARCH_SYNTHESIS = "research_synthesis"
    CREATIVE_WRITING = "creative_writing"
    MULTILINGUAL = "multilingual"
    VISION = "vision"
    TOOL_PLANNING = "tool_planning"
    SYSTEM_COMMAND = "system_command"
    SAFETY_SENSITIVE = "safety_sensitive"
    PRIVATE_DOCUMENT = "private_document"
    UNKNOWN = "unknown"


class ModelCapability(str, Enum):
    GENERAL_CHAT = "general_chat"
    LOW_LATENCY = "low_latency"
    REASONING = "reasoning"
    CODING = "coding"
    LONG_CONTEXT = "long_context"
    STRUCTURED_OUTPUT = "structured_output"
    MULTILINGUAL = "multilingual"
    VISION = "vision"
    TOOL_CALLING = "tool_calling"
    HIGH_RELIABILITY = "high_reliability"
    LOCAL_EXECUTION = "local_execution"


class OmniRouteRoute(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route_id: str
    gateway: str = "omniroute"
    provider_id: str
    model_id: str
    display_name: str
    locality: RouteLocality
    privacy_class: RoutePrivacy
    capabilities: list[ModelCapability] = Field(default_factory=list)
    task_categories: list[TaskCategory] = Field(default_factory=list)
    context_window: int = Field(ge=1)
    supports_streaming: bool = True
    supports_tools: bool = False
    supports_structured_output: bool = False
    expected_latency_class: str = "unknown"
    expected_quality_class: str = "unknown"
    cost_class: str = "unknown"
    quota_class: str = "unknown"
    terms_status: str = "unreviewed"
    development_only: bool = False
    production_approved: bool = False
    enabled: bool = False
    operator_priority: int = Field(default=50, ge=0, le=100)
    fallback_route_ids: list[str] = Field(default_factory=list)
    last_verified_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("route_id", "provider_id")
    @classmethod
    def validate_path_segment(cls, value: str) -> str:
        cleaned = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]+", cleaned):
            raise ValueError("Route and provider IDs must be safe path segments.")
        return cleaned

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or any(char in cleaned for char in "?#\\\n\r") or ".." in cleaned:
            raise ValueError("Model IDs must be non-empty and URL-safe.")
        return cleaned

    @field_validator("gateway")
    @classmethod
    def validate_gateway(cls, value: str) -> str:
        if value != "omniroute":
            raise ValueError("Only the OmniRoute gateway is valid in this registry.")
        return value


class RouteClassification(BaseModel):
    task_category: TaskCategory
    complexity: str
    privacy: str
    required_capabilities: list[ModelCapability]
    latency_priority: str
    quality_priority: str
    estimated_context_tokens: int = 0
    reason_codes: list[str] = Field(default_factory=list)


class RouteSelection(BaseModel):
    selected: bool
    route: OmniRouteRoute | None = None
    score: float = 0.0
    reason_codes: list[str] = Field(default_factory=list)
    rejected_routes: dict[str, list[str]] = Field(default_factory=dict)


class DiscoveredModel(BaseModel):
    model_id: str
    provider_id: str | None = None
    owned_by: str | None = None


class CatalogSnapshot(BaseModel):
    state: RouteHealthState
    models: list[DiscoveredModel] = Field(default_factory=list)
    refreshed_at: datetime = Field(default_factory=utc_now)
    duplicate_count: int = 0
    approved_present: list[str] = Field(default_factory=list)
    approved_missing: list[str] = Field(default_factory=list)
    discovered_unapproved: list[str] = Field(default_factory=list)
    safe_error: str | None = None
