"""In-process agent interface used by the orchestration layer."""
from typing import Protocol, runtime_checkable

from agent.contracts import AgentName, AgentRequest, AgentResult


@runtime_checkable
class DomainAgent(Protocol):
    """A domain capability callable by the orchestrator.

    Implementations may wrap existing synchronous business logic. Transport
    concerns (A2A, HTTP, queues) do not belong in this interface.
    """

    name: AgentName

    def run(self, request: AgentRequest) -> AgentResult:
        """Handle one validated request and return a normalized result."""
        ...
