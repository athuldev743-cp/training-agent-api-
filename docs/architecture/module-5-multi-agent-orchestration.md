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

## FastAPI integration

The authenticated `/v1/chat` handler now constructs the graph per request:
- The verified token subject is the only source of `AgentRequest.user_id`.
- The user's profile, history and program stores are request-scoped and closed in a `finally` block.
- The existing fitness dispatcher is injected through `FitnessAgentAdapter`; its existing confirmation flow is preserved by carrying its pending action metadata through the in-process result.
- Nutrition requests use `NutritionAgent` with the configured model and Module 3 retrieval pipeline.
- Mixed requests invoke both agents and aggregate their responses. If the fitness dispatcher requests confirmation, the API still issues the existing user-bound confirmation token; no write is executed by orchestration.
- `/v1/chat/stream` reuses the same `run_chat` path, so streaming and non-streaming requests share routing and authorization behavior.

No separate agent server or A2A transport is introduced. The graph and both agents run in the FastAPI process.

## Integration tests

`tests/test_api_multi_agent_integration.py` covers authenticated user propagation, fitness dispatcher delegation, nutrition-only routing, mixed routing, and preservation of the fitness confirmation-token flow. These tests use isolated fake stores and do not write to the production database.

## Tests

```powershell
python -m pytest tests/test_multi_agent_orchestrator.py -q
```

Tests cover fitness-only, nutrition-only, mixed and ambiguous routing; result aggregation; failure propagation; adapter normalization; and agent identity validation. They use stub agents and do not claim production API integration.
