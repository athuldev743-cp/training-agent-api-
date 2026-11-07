"""LangGraph orchestration for the in-process fitness and nutrition agents.

The graph owns routing and result aggregation. Domain agents remain independently
testable and communicate through the validated AgentRequest/AgentResult contracts.
"""
from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent.contracts import AgentName, AgentRequest, AgentResult, AgentStatus
from agent.interfaces import DomainAgent

_NUTRITION = re.compile(
    r"\b(nutrition|diet|dietary|food|foods|meal|meals|calorie|calories|"
    r"protein|carbohydrate|carbs|fat intake|macros|hydration|water intake)\b",
    re.IGNORECASE,
)
_FITNESS = re.compile(
    r"\b(workout|workouts|exercise|exercises|training|train|gym|muscle|"
    r"sets|reps|routine|programme|program|progression|strength|cardio)\b|"
    r"\bprogressive\s+overload\b|\bweightlifting\b|\bresistance\s+training\b",
    re.IGNORECASE,
)


class OrchestrationState(TypedDict, total=False):
    request: AgentRequest
    selected: list[str]
    fitness_result: AgentResult | None
    nutrition_result: AgentResult | None
    results: list[AgentResult]
    final: AgentResult


def classify_request(message: str) -> list[AgentName]:
    """Deterministic, auditable routing; mixed requests invoke both agents."""
    fitness = bool(_FITNESS.search(message))
    nutrition = bool(_NUTRITION.search(message))
    if fitness and nutrition:
        return [AgentName.FITNESS, AgentName.NUTRITION]
    if nutrition:
        return [AgentName.NUTRITION]
    if fitness:
        return [AgentName.FITNESS]
    return []


class MultiAgentOrchestrator:
    def __init__(self, fitness_agent: DomainAgent, nutrition_agent: DomainAgent):
        if fitness_agent.name is not AgentName.FITNESS:
            raise ValueError("fitness_agent must identify itself as fitness")
        if nutrition_agent.name is not AgentName.NUTRITION:
            raise ValueError("nutrition_agent must identify itself as nutrition")
        self.agents = {
            AgentName.FITNESS.value: fitness_agent,
            AgentName.NUTRITION.value: nutrition_agent,
        }
        graph = StateGraph(OrchestrationState)
        graph.add_node("classify", self._classify)
        graph.add_node("fitness", self._fitness)
        graph.add_node("nutrition", self._nutrition)
        graph.add_node("aggregate", self._aggregate)
        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify", self._next, {
                "fitness": "fitness",
                "nutrition": "nutrition",
                "both": "fitness",
                "clarify": "aggregate",
            }
        )
        graph.add_conditional_edges("fitness", self._after_fitness, {
            "nutrition": "nutrition",
            "aggregate": "aggregate",
        })
        graph.add_edge("nutrition", "aggregate")
        graph.add_edge("aggregate", END)
        self.graph = graph.compile()

    def invoke(self, request: AgentRequest) -> AgentResult:
        return self.graph.invoke({"request": request})["final"]

    def _classify(self, state: OrchestrationState) -> dict:
        selected = [item.value for item in classify_request(state["request"].message)]
        return {"selected": selected}

    @staticmethod
    def _next(state: OrchestrationState) -> str:
        selected = state.get("selected", [])
        if selected == ["fitness"]:
            return "fitness"
        if selected == ["nutrition"]:
            return "nutrition"
        if selected == ["fitness", "nutrition"]:
            return "both"
        return "clarify"

    def _fitness(self, state: OrchestrationState) -> dict:
        result = self.agents["fitness"].run(state["request"])
        return {"fitness_result": result}

    def _nutrition(self, state: OrchestrationState) -> dict:
        result = self.agents["nutrition"].run(state["request"])
        return {"nutrition_result": result}

    @staticmethod
    def _after_fitness(state: OrchestrationState) -> str:
        return "nutrition" if "nutrition" in state.get("selected", []) else "aggregate"

    @staticmethod
    def _aggregate(state: OrchestrationState) -> dict:
        request = state["request"]
        selected = state.get("selected", [])
        results = []
        if state.get("fitness_result") is not None:
            results.append(state["fitness_result"])
        if state.get("nutrition_result") is not None:
            results.append(state["nutrition_result"])

        if not selected:
            final = AgentResult(
                request_id=request.request_id,
                agent=AgentName.FITNESS,
                status=AgentStatus.NEEDS_CLARIFICATION,
                message=("I can help with workouts, exercise and training, or with nutrition, "
                         "food and meal questions. Which area do you mean?"),
            )
        elif len(results) == 1:
            final = results[0]
        else:
            statuses = {item.status for item in results}
            status = AgentStatus.OK if statuses == {AgentStatus.OK} else (
                AgentStatus.UNAVAILABLE if AgentStatus.UNAVAILABLE in statuses
                else AgentStatus.NEEDS_CLARIFICATION
            )
            message = "\n\n".join(
                f"{item.agent.value.title()}: {item.message}" for item in results
            )
            evidence = [ref for item in results for ref in item.evidence][:20]
            final = AgentResult(
                request_id=request.request_id,
                agent=AgentName.FITNESS,
                status=status,
                message=message,
                evidence=evidence,
                metadata={
                    "fitness": state["fitness_result"].metadata if state.get("fitness_result") else {},
                    "nutrition": state["nutrition_result"].metadata if state.get("nutrition_result") else {},
                },
            )
        return {"results": results, "final": final}
