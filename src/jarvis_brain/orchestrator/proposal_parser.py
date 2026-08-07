import json
from typing import Any

from jarvis_platform.schemas.brain_orchestration import BrainOrchestratorProposal


class BrainProposalParser:
    """Parse structured model output into a typed proposal."""

    def parse(self, text: str) -> BrainOrchestratorProposal | None:
        data = self._parse_json_object(text)
        if data is None:
            return None
        try:
            return BrainOrchestratorProposal.model_validate(data)
        except Exception:
            return None

    def _parse_json_object(self, text: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : index + 1])
                        return parsed if isinstance(parsed, dict) else None
                    except json.JSONDecodeError:
                        return None
        return None

