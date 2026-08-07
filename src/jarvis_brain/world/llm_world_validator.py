import json
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from jarvis_platform.schemas.llm_world_intelligence import (
    LLMWorldBriefingCandidate,
    LLMWorldDecision,
    LLMWorldRiskFlag,
    LLMWorldValidationResult,
)
from jarvis_platform.security.secret_policy import SecretPolicyEngine


class LLMWorldValidator:
    """Validate untrusted LLM world briefing refinements."""

    UNSAFE_PHRASES = {
        "bypass security",
        "disable safety",
        "reveal secrets",
        "reveal secret",
        "steal",
        "exfiltrate",
        "execute malware",
        "install malware",
        "send credentials",
    }
    EXECUTION_PHRASES = {
        "send email",
        "delete file",
        "make payment",
        "unlock door",
        "run command",
        "execute command",
        "buy ",
        "sell ",
    }
    LIVE_CLAIMS = {
        "live data",
        "live feeds",
        "real-time",
        "real time",
        "just retrieved",
        "breaking news",
        "confirmed today",
    }

    def __init__(self, secret_policy: SecretPolicyEngine | None = None) -> None:
        """Create a validator with SecretGuard output protection."""
        self.secret_policy = secret_policy or SecretPolicyEngine()

    def parse_candidate(
        self,
        briefing_type: str,
        llm_text: str,
    ) -> LLMWorldBriefingCandidate | None:
        """Parse the first JSON object without executing its contents."""
        decoder = json.JSONDecoder()
        for index, character in enumerate(llm_text):
            if character != "{":
                continue
            try:
                data, _ = decoder.raw_decode(llm_text[index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            try:
                return LLMWorldBriefingCandidate(
                    candidate_id=str(uuid4()),
                    briefing_type=briefing_type,
                    **data,
                )
            except (TypeError, ValidationError):
                return None
        return None

    def validate(
        self,
        candidate: LLMWorldBriefingCandidate,
        events: list | None = None,
        suggestions: list | None = None,
        alerts: list | None = None,
    ) -> LLMWorldValidationResult:
        """Accept only source-bounded, non-executing briefing refinements."""
        if candidate.confidence < 0.55:
            return self._rejected(
                candidate,
                "World briefing confidence is below 0.55.",
                [],
            )

        flags = self.detect_risk_flags(candidate, events, suggestions, alerts)
        if flags:
            return self._rejected(
                candidate,
                "World briefing failed one or more safety checks.",
                flags,
            )

        policy = self.secret_policy.enforce_output_policy(
            candidate.summary,
            context="llm_world_intelligence",
        )
        if policy["blocked"]:
            return self._rejected(
                candidate,
                "World briefing contained protected sensitive data.",
                [LLMWorldRiskFlag.SECRET_EXPOSURE],
            )

        metadata = {"low_confidence": True} if candidate.confidence < 0.7 else {}
        return LLMWorldValidationResult(
            decision=LLMWorldDecision.ACCEPTED,
            candidate=candidate,
            reason="World briefing passed source and safety checks.",
            sanitized_summary=policy["redacted_text"],
            metadata=metadata,
        )

    def detect_risk_flags(
        self,
        candidate: LLMWorldBriefingCandidate,
        events: list | None,
        suggestions: list | None,
        alerts: list | None,
    ) -> list[LLMWorldRiskFlag]:
        """Detect invented, unsafe, secret, or source-trust issues."""
        flags = {
            flag for flag in candidate.risk_flags if flag != LLMWorldRiskFlag.NONE
        }
        event_maps = [self._as_dict(event) for event in events or []]
        suggestion_maps = [self._as_dict(item) for item in suggestions or []]
        alert_maps = [self._as_dict(item) for item in alerts or []]
        valid_event_ids = {
            str(event.get("event_id"))
            for event in event_maps
            if event.get("event_id") is not None
        }
        evidence_ids = {str(event_id) for event_id in candidate.evidence_event_ids}
        if not evidence_ids <= valid_event_ids:
            flags.add(LLMWorldRiskFlag.INVENTED_FACT)

        text = self._candidate_text(candidate).lower()
        if self.secret_policy.inspect_text(text, context="llm_world").has_secrets:
            flags.add(LLMWorldRiskFlag.SECRET_EXPOSURE)
        if any(phrase in text for phrase in self.UNSAFE_PHRASES):
            flags.add(LLMWorldRiskFlag.UNSAFE_RECOMMENDATION)
        if any(phrase in text for phrase in self.EXECUTION_PHRASES):
            if "approval" not in text:
                flags.add(LLMWorldRiskFlag.UNSAFE_RECOMMENDATION)

        if self._has_mock_data(event_maps) and any(
            phrase in text for phrase in self.LIVE_CLAIMS
        ):
            flags.add(LLMWorldRiskFlag.SOURCE_TRUST_ISSUE)
            flags.add(LLMWorldRiskFlag.EXCESSIVE_CONFIDENCE)

        support_text = self._support_text(event_maps, suggestion_maps, alert_maps)
        for item in candidate.priority_items:
            if not self._is_supported(item, support_text):
                flags.add(LLMWorldRiskFlag.UNSUPPORTED_CLAIM)
        for item in candidate.alerts:
            if alert_maps and not self._is_supported(item, support_text):
                flags.add(LLMWorldRiskFlag.UNSUPPORTED_CLAIM)

        return sorted(flags, key=lambda flag: flag.value)

    def _candidate_text(self, candidate: LLMWorldBriefingCandidate) -> str:
        return " ".join(
            [
                candidate.summary,
                *candidate.priority_items,
                *candidate.alerts,
                *candidate.project_relevance,
                *candidate.suggested_next_steps,
            ]
        )

    def _as_dict(self, item: Any) -> dict[str, Any]:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if isinstance(item, dict):
            return item
        return {"value": str(item)}

    def _has_mock_data(self, events: list[dict[str, Any]]) -> bool:
        return any(
            str(event.get("source_name", "")).lower().startswith("mock")
            or bool(event.get("metadata", {}).get("mock"))
            for event in events
        )

    def _support_text(
        self,
        events: list[dict[str, Any]],
        suggestions: list[dict[str, Any]],
        alerts: list[dict[str, Any]],
    ) -> str:
        pieces: list[str] = []
        for item in [*events, *suggestions, *alerts]:
            pieces.extend(str(value) for value in item.values() if value is not None)
        return " ".join(pieces).lower()

    def _is_supported(self, text: str, support_text: str) -> bool:
        words = {
            word.strip(".,:;!?()[]{}").lower()
            for word in text.split()
            if len(word.strip(".,:;!?()[]{}")) >= 5
        }
        if not words:
            return True
        return bool(words & set(support_text.split()))

    def _rejected(
        self,
        candidate: LLMWorldBriefingCandidate,
        reason: str,
        flags: list[LLMWorldRiskFlag],
    ) -> LLMWorldValidationResult:
        return LLMWorldValidationResult(
            decision=LLMWorldDecision.REJECTED,
            candidate=candidate,
            reason=reason,
            risk_flags=flags,
        )
