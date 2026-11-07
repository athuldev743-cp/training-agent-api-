# Fitness and Nutrition Multi-Agent Architecture

## Current baseline

The application is a single FastAPI service. The existing fitness workflow is
implemented in Python modules:

- `agent/dispatcher.py` routes a chat request and invokes existing tools.
- `agent/coach.py` and `agent/coach_tools.py` implement workout and programme
  operations.
- `engine/` contains exercise prescription, progression, safety and programme
  rules.
- `state/` stores the user profile, workout history and programmes.
- `rag/hybrid.py` retrieves from the existing `fitness_knowledge` PostgreSQL
  corpus using vector search, PostgreSQL full-text search, RRF and reranking.

The nutrition workflow is not implemented yet. Nutrition must be added without
replacing the fitness logic or mixing nutrition evidence into the exercise
knowledge corpus.

## Target architecture

```text
Android / API client
        |
        v
     FastAPI
  auth + validation
        |
        v
  Application boundary
        |
        v
 LangGraph orchestrator
        |
        +------ Fitness agent adapter
        |          |
        |          +-- existing dispatcher / coach tools
        |          +-- exercise rules and fitness RAG
        |
        +------ Nutrition agent
                   |
                   +-- nutrition retrieval pipeline
                   +-- nutrition response and safety rules

Agents access approved capabilities through application services.
MCP may expose selected tools and knowledge resources to compatible clients.
```

LangGraph will own multi-agent routing, state transitions and workflow
composition when orchestration is introduced. Existing fitness functions remain
the source of fitness business logic and are wrapped by an adapter rather than
rewritten as a second coach.

## Communication and protocol choices

### Within this application

Use direct in-process Python calls with Pydantic request/result models and a
small `DomainAgent` protocol. This is the correct first boundary while both
agents run in the same Python process:

- no network hop or second server is required;
- request and response shapes are validated;
- unit tests can replace agents with small fakes;
- the orchestration graph remains independent of domain implementation details.

The initial contracts live in `agent/contracts.py`; the protocol lives in
`agent/interfaces.py`.

### LangGraph

LangGraph is the orchestration layer, not the agent communication transport.
It will be introduced when the fitness and nutrition nodes are integrated. It
will manage routing and workflow state; it will not replace domain rules,
database repositories or safety validation.

### MCP

MCP is a tool/resource interface. It can expose approved operations such as
profile lookup and domain-specific knowledge retrieval. It is not the
fitness-to-nutrition agent protocol and does not replace LangGraph.

### A2A

Do not introduce A2A or deploy separate agent servers at this stage. Revisit
A2A only if fitness and nutrition become independently deployed services that
must discover and invoke one another across a service boundary. At that point,
the existing Pydantic contracts can inform a transport adapter, but should not
be treated as an A2A wire specification.

## Data and evidence boundaries

- User state and retrieved knowledge are different inputs.
- The existing validated `UserProfile` is passed as user context; missing
  profile fields remain unknown and must not be guessed.
- Nutrition-specific preferences, allergies and other constraints must be
  modelled and persisted deliberately in a later module. Do not silently add
  them to the current profile or infer them.
- Keep nutrition documents and retrieval isolated from `fitness_knowledge`.
  Reuse PostgreSQL/pgvector infrastructure where suitable, but use a separate
  nutrition corpus/table or an explicitly enforced domain partition.
- A nutrition answer should be grounded in nutrition evidence when the task
  requires evidence. Evidence references are represented by
  `EvidenceReference`.

## Safety and execution boundaries

- The LLM may propose a route; application code validates the route.
- Domain agents return typed results. They do not receive unrestricted database
  connections or execute arbitrary tools supplied by model output.
- State-changing operations remain behind existing application-layer
  authorization, validation and explicit confirmation.
- Never expose raw exception messages, credentials, internal database details or
  stack traces in agent results.
- The orchestrator must preserve the request identity across delegated calls.

## Module 1 scope

This module establishes the architecture decision record, typed contracts,
agent protocol and CI test baseline. It does not yet add LangGraph runtime,
MCP server tools, nutrition ingestion, nutrition retrieval or production
routing. Those are implemented and evaluated in later modules.
