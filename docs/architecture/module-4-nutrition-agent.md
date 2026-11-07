# Module 4 — Evidence-grounded Nutrition Agent

## Responsibility

The Nutrition Agent implements the shared `DomainAgent` interface and accepts an `AgentRequest`, returning a validated `AgentResult`. It is an in-process capability, not a separate HTTP service. Module 5 will connect it to the orchestrator.

## Request flow

1. Receive the validated user message and request ID.
2. Retrieve up to five candidates from `search_nutrition` (hybrid pgvector + PostgreSQL full-text search + RRF + reranking).
3. Build a bounded evidence context. Retrieved documents are treated as untrusted data, never as instructions.
4. Ask the configured LLM to explain only the supplied evidence and cite it as `[S1]`, `[S2]`, etc.
5. Validate that at least one citation exists and that every citation maps to a source supplied in this request.
6. Return an `AgentResult` with source, page, chunk number and excerpt in its evidence list.

## Failure behavior

- Retrieval exception or no matching evidence: `unavailable`; the LLM is not called when retrieval returns no results.
- Empty or unusable chunks: `unavailable`.
- LLM/provider exception: `unavailable`, with no raw exception details exposed.
- Missing or invalid citations: fail closed with `unavailable`, retaining the retrieved evidence for inspection.
- Successful answer: `ok` with verified source references.

## Safety boundaries

The agent provides general nutrition education, not diagnosis or medical treatment. It does not prescribe therapeutic diets or supplement doses for medical conditions. It does not assume allergies, intolerances, preferences or medical diagnoses because those fields are not in the current profile schema. Disease-specific or medically complex questions are directed to a registered dietitian or treating clinician.

Citation validation verifies that cited source labels exist; it does not independently prove that every generated sentence is entailed by the cited text. The retrieval corpus itself must be reviewed for authority, freshness and accuracy. Module 6 will add stronger policy and personalization checks.

## Local testing

```powershell
python -m pytest tests/test_nutrition_agent.py -q
```

The test suite mocks the LLM and retrieval boundary. It validates request/result contracts, evidence attribution, citation allow-listing, no-evidence behavior, exception handling and bounded retrieval. It does not claim medical accuracy or live Neon retrieval quality.
