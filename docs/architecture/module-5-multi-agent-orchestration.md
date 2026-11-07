# Module 5 — LangGraph Multi-Agent Orchestration

## Architecture

`MultiAgentOrchestrator` is the workflow owner. It routes a validated `AgentRequest` to:
- `FitnessAgentAdapter`: wraps the existing deterministic fitness dispatcher through an injected handler.
- `NutritionAgent`: invokes the Module 3 nutrition retriever and returns evidence-grounded results.

Agents remain in the same Python process and exchange Pydantic contracts directly. There are no agent HTTP calls, queues, or separate deployments. MCP remains the tool interface; it is not used as the agent-to-agent transport. A2A is deferred until independent service deployment is justified.

## Routing behavior

The initial router is deterministic and auditable:
- Fitness terms route to the fitness adapter.
- Nutrition terms route to the nutrition agent.
- Requests containing both domains invoke both agents sequentially and aggregate their results.
- Unclear requests return a clarification response without calling either agent.

This intentionally avoids an LLM deciding whether to execute a domain capability. The keyword rules are a baseline and will be evaluated against a labeled routing dataset before being expanded.

## Safety and failure handling

- Each agent receives the same validated request ID, user ID, message and profile.
- Fitness legacy output is normalized to `AgentResult`; raw exceptions are not exposed.
- Mixed-request aggregation preserves an unavailable status rather than silently presenting a partial response as complete.
- Existing fitness write confirmations and deterministic execution remain in the legacy handler; the orchestrator itself has no database write capability.
- The fitness handler is injected by the authenticated application layer so user-scoped stores are not shared across requests.

## Integration boundary

The module provides the graph and adapter, but does not yet replace the existing `/v1/chat` endpoint. API-level dependency wiring, authentication-context propagation, and endpoint regression tests are the next integration step. No production route is switched in this module.

## Tests

```powershell
python -m pytest tests/test_multi_agent_orchestrator.py -q
```

Tests cover fitness-only, nutrition-only, mixed and ambiguous routing; result aggregation; failure propagation; adapter normalization; and agent identity validation. They use stub agents and do not claim production API integration.
