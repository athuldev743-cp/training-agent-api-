"""Adapter for the existing fitness dispatcher.

The adapter keeps legacy workout logic intact. The application supplies a handler
closure with the authenticated user's stores and exercise catalogue; no state or
database handles are passed through the LLM or serialized between agents.
"""
from __future__ import annotations

from typing import Callable

from agent.contracts import AgentName, AgentRequest, AgentResult, AgentStatus
from agent.interfaces import DomainAgent


class FitnessAgentAdapter:
    name = AgentName.FITNESS

    def __init__(self, handler: Callable[[AgentRequest], dict]):
        self.handler = handler

    def run(self, request: AgentRequest) -> AgentResult:
        try:
            output = self.handler(request)
            if not isinstance(output, dict):
                raise TypeError("fitness handler must return a dictionary")
            status_map = {
                "ok": AgentStatus.OK,
                "needs_info": AgentStatus.NEEDS_CLARIFICATION,
                "confirm": AgentStatus.NEEDS_CLARIFICATION,
                "safety_caution": AgentStatus.REFUSED,
                "safety_urgent": AgentStatus.REFUSED,
                "out_of_scope": AgentStatus.REFUSED,
                "unavailable": AgentStatus.UNAVAILABLE,
                "error": AgentStatus.ERROR,
            }
            status = status_map.get(str(output.get("status")), AgentStatus.ERROR)
            message = output.get("reply")
            if not isinstance(message, str) or not message.strip():
                message = "The fitness service could not produce a valid response."
                status = AgentStatus.ERROR
            return AgentResult(
                request_id=request.request_id,
                agent=self.name,
                status=status,
                message=message[:4000],
            )
        except Exception:
            return AgentResult(
                request_id=request.request_id,
                agent=self.name,
                status=AgentStatus.UNAVAILABLE,
                message="The fitness service is temporarily unavailable.",
            )
