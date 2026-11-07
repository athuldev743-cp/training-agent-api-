"""Validated contracts shared by the fitness and nutrition agents.

These are in-process application contracts. They are deliberately independent
of LangGraph, MCP and any transport protocol.
"""
from datetime import date
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from state.profile import UserProfile


class AgentName(str, Enum):
    FITNESS = "fitness"
    NUTRITION = "nutrition"


class AgentStatus(str, Enum):
    OK = "ok"
    NEEDS_CLARIFICATION = "needs_clarification"
    REFUSED = "refused"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class EvidenceReference(BaseModel):
    """A source reference returned by a knowledge-backed agent."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str = Field(min_length=1, max_length=300)
    page: int | None = Field(default=None, ge=1)
    chunk_number: int | None = Field(default=None, ge=0)
    quote: str | None = Field(default=None, max_length=1200)


class AgentRequest(BaseModel):
    """Input passed from the orchestrator to one domain agent."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    user_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=1000)
    profile: UserProfile | None = None
    today: date | None = None


class AgentResult(BaseModel):
    """Normalized output returned by a domain agent to the orchestrator."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    agent: AgentName
    status: AgentStatus
    message: str = Field(min_length=1, max_length=4000)
    evidence: list[EvidenceReference] = Field(default_factory=list, max_length=20)
