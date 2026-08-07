from jarvis_platform.schemas.brain_orchestration import BrainCapability, BrainIntentType


class BrainCapabilityRegistry:
    """Translate brain needs into abstract capabilities."""

    def capabilities_for_intents(self, intent_types: list[BrainIntentType]) -> list[BrainCapability]:
        capabilities: set[BrainCapability] = {BrainCapability.TEXT_GENERATION, BrainCapability.JSON}
        if BrainIntentType.CODING in intent_types:
            capabilities.add(BrainCapability.CODE_GENERATION)
            capabilities.add(BrainCapability.REASONING)
        if BrainIntentType.PLANNING in intent_types:
            capabilities.add(BrainCapability.REASONING)
        if BrainIntentType.VISION in intent_types:
            capabilities.add(BrainCapability.VISION)
            capabilities.add(BrainCapability.IMAGE_UNDERSTANDING)
        if BrainIntentType.VOICE in intent_types:
            capabilities.add(BrainCapability.SPEECH)
        if BrainIntentType.RESEARCH in intent_types or BrainIntentType.WORLD in intent_types:
            capabilities.add(BrainCapability.LONG_CONTEXT)
        if BrainIntentType.EXECUTION in intent_types or BrainIntentType.AUTOMATION in intent_types:
            capabilities.add(BrainCapability.TOOL_CALLING)
        return sorted(capabilities, key=lambda item: item.value)

