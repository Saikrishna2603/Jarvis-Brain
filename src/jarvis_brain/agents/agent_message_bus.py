from dataclasses import dataclass, field
from uuid import uuid4

from jarvis_platform.schemas.common import utc_now


@dataclass
class AgentMessage:
    """Message exchanged between swarm agents."""

    sender: str
    recipient: str
    content: str
    message_type: str = "proposal"
    metadata: dict = field(default_factory=dict)
    message_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: object = field(default_factory=utc_now)


class AgentMessageBus:
    """In-memory message bus for one swarm run."""

    def __init__(self) -> None:
        """Create an empty message bus."""
        self._messages: list[AgentMessage] = []

    def publish(self, message: AgentMessage) -> AgentMessage:
        """Store and return a message."""
        self._messages.append(message)
        return message

    def all_messages(self) -> list[AgentMessage]:
        """Return all messages in insertion order."""
        return list(self._messages)

    def messages_for(self, recipient: str) -> list[AgentMessage]:
        """Return messages for one recipient."""
        return [
            message
            for message in self._messages
            if message.recipient == recipient
        ]

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

