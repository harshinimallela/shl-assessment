# SHL Assessment Recommender

> A production-grade conversational AI system for recommending SHL Individual Test Solutions.
> Multi-agent architecture · Semantic retrieval (FAISS + CrossEncoder) · Zero hallucination guarantee.

```
POST /chat  →  Supervisor
               ├── GuardrailAgent       keyword fast-path + LLM safety classifier
               ├── IntentAgent          structured HiringIntent extraction
               ├── FAISSCatalogStore    sentence-transformers → FAISS → CrossEncoder reranker
               └── routes to:
                   ├── ClarificationAgent
                   ├── RecommendationAgent
                   ├── ComparisonAgent
                   ├── RefinementAgent
                   └── RefuseAgent
```

---

## Quick Start

```bash
git clone <repo> && cd shl-recommender

# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: set GROQ_API_KEY=<YOUR_GROQ_API_KEY>

# Run
uvicorn app.main:app --reload --port 8000

# Verify
curl http://localhost:8000/health
# → {"status": "ok"}
```

### One-liner test

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hiring a Senior Java Engineer, 7 years experience, team lead"}]}'
```

---

## API Reference

### `GET /health`

Readiness probe. Returns HTTP 200 when service is ready.

```json
{ "status": "ok" }
```

### `POST /chat`

Stateless. Send the **full conversation history** on every request.

**Request:**

```json
{
  "messages": [
    { "role": "user", "content": "Hiring a mid-level Java developer" },
    {
      "role": "assistant",
      "content": "What seniority level are you targeting?"
    },
    { "role": "user", "content": "Around 4 years experience" }
  ]
}
```

**Response** (schema is non-negotiable):

```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/...",
      "test_type": "K"
    },
    { "name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P" },
    { "name": "Verify-G+", "url": "https://www.shl.com/...", "test_type": "A" }
  ],
  "end_of_conversation": false
}
```

| Field                 | Type           | Description                                        |
| --------------------- | -------------- | -------------------------------------------------- |
| `reply`               | `string`       | Conversational agent response                      |
| `recommendations`     | `array[0..10]` | Empty while clarifying. 1–10 items when committed. |
| `end_of_conversation` | `boolean`      | `true` only when user signals task complete        |

**Test type codes:** `A`=Ability, `P`=Personality, `K`=Knowledge, `S`=Simulation, `B`=Behavioral

---

## Architecture

### Agent Pipeline

```
HTTP Request
    │
    ▼
RequestLoggingMiddleware     ← UUID per request, latency, status code on every response
    │
    ▼
Supervisor.handle(messages)
    │
    ├─► GuardrailAgent                    ← fast-path keyword check → LLM safety classifier
    │     └── unsafe? → RefuseAgent
    │
    ├─► IntentAgent                       ← full conversation → typed HiringIntent struct
    │     {role, seniority, skills, action, personality_needed, cognitive_needed, ...}
    │
    ├─► FAISSCatalogStore.search()        ← query embedding → FAISS top-20 → CrossEncoder top-5
    │     (or TFIDFCatalogStore on free-tier deployments)
    │
    └─► route by intent.action:
          CLARIFY   → ClarificationAgent   one focused question
          RECOMMEND → RecommendationAgent  reranks retrieved entries, explains selection
          COMPARE   → ComparisonAgent      catalog-grounded diff of named assessments
          REFINE    → RefinementAgent      merges user edits into existing shortlist
          REFUSE    → RefuseAgent          polite redirect for off-topic queries
```

### Retrieval Pipeline

```
Query string
    │
    ▼
SentenceTransformer.encode()             all-MiniLM-L6-v2, 384-dim, L2-normalized
    │
    ▼
faiss.IndexFlatIP.search(top_k=20)       exact cosine similarity, sub-ms at 89 items
    │
    ▼
CrossEncoder.predict()                   ms-marco-MiniLM-L-6-v2, joint (query, doc) scoring
    │
    ▼
Metadata filter                          optional test_type, job_level, language constraints
    │
    ▼
Top-5 to Top-15 candidates               injected into agent system prompt
```

**Why two stages?**
Bi-encoder retrieval (FAISS) maximizes recall — cast a wide net. CrossEncoder reranking maximizes precision — it sees query and document together, enabling much richer relevance judgment. This is the industry-standard two-tower pattern for production RAG.

### Hallucination Prevention

Two independent layers ensure the LLM cannot return an invented assessment:

1. **Prompt layer** — each agent receives only retrieved entries. The system prompt says: _"Only recommend assessments from the RETRIEVED LIST."_
2. **Code layer** — `_validate()` in `recommend.py` cross-checks every returned URL against the retrieved set. Anything not in the pool is dropped before the response leaves the server.

Even if the LLM ignores the prompt, the code catches it. This is a hard API-boundary guarantee.

---

## Project Structure

```
shl-recommender/
├── app/
│   ├── main.py               FastAPI entrypoint (thin — no business logic)
│   ├── supervisor.py         Routing orchestrator
│   ├── config.py             pydantic-settings config, validated at startup
│   ├── dependencies.py       Composition root — all singletons built once
│   ├── prompt_manager.py     Prompt file loader and cache
│   ├── agents/
│   │   ├── base.py           Shared LLM call logic (retry, logging, JSON parsing)
│   │   ├── guardrail.py      Safety classifier (keyword + LLM)
│   │   ├── intent.py         Structured intent extraction → HiringIntent
│   │   ├── clarify.py        Single clarifying question
│   │   ├── recommend.py      Shortlist selection + validation
│   │   ├── compare.py        Assessment comparison
│   │   ├── refine.py         Constraint merging + re-retrieval
│   │   └── refuse.py         Off-topic polite redirect
│   ├── models/
│   │   ├── api.py            Request/response schemas (API contract)
│   │   ├── catalog.py        CatalogEntry domain model
│   │   └── intent.py         HiringIntent + AgentAction enum
│   ├── retrieval/
│   │   ├── __init__.py       Factory: selects backend from config
│   │   ├── base.py           Retriever protocol (structural typing)
│   │   ├── faiss_store.py    Semantic retrieval: embeddings + FAISS + CrossEncoder
│   │   └── tfidf_store.py    Zero-dependency fallback: TF-IDF cosine similarity
│   ├── middleware/
│   │   └── logging.py        Request ID + latency middleware
│   └── prompts/              One .txt per agent — edit without touching code
│       ├── guardrail.txt
│       ├── intent.txt
│       ├── clarify.txt
│       ├── recommend.txt
│       ├── compare.txt
│       ├── refine.txt
│       └── refuse.txt
├── tests/
│   ├── unit/                 No LLM, no network required
│   │   ├── test_catalog_store.py
│   │   └── test_models.py    Models, validation, prompt manager, hallucination guard
│   └── integration/          Requires running service
│       └── test_api.py       All evaluator behavioral probes
├── scripts/
│   └── evaluate.py           Recall@K, latency, routing accuracy, markdown report
├── catalog.json              89 SHL Individual Test Solutions
├── Dockerfile                Multi-stage production build
├── docker-compose.yml        Local dev
├── render.yaml               One-click Render deployment
├── Makefile                  All common commands
├── pyproject.toml            ruff, black, mypy, pytest config
├── .pre-commit-config.yaml   Pre-commit hooks
└── requirements.txt
```

---

## Running Tests

```bash
# Unit tests — no LLM, no network, runs in seconds
make test

# Unit tests with coverage report
make test-unit

# Integration tests — requires running service
make run &
make test-integration

# Full evaluation suite (Recall@K, latency, routing accuracy)
make eval
# Output: eval_report.md
```

### Unit test coverage

| Module                | Tests                                                              |
| --------------------- | ------------------------------------------------------------------ |
| `tfidf_store.py`      | Tokenizer, initialization, search, filters, validation, edge cases |
| `models/api.py`       | Request validation, response schema                                |
| `models/catalog.py`   | CatalogEntry, searchable_text                                      |
| `models/intent.py`    | HiringIntent, retrieval_query, AgentAction                         |
| `prompt_manager.py`   | Loading, caching, missing file errors                              |
| `agents/recommend.py` | `_validate()` hallucination guard, deduplication, URL correction   |
| `agents/base.py`      | JSON parsing (bare, fenced, embedded, invalid)                     |

---

## Deployment

### Render (recommended — free tier)

```bash
# 1. Push to GitHub
# 2. connect repo at render.com → New Web Service
# 3. Build: pip install -r requirements.txt
# 4. Start: uvicorn app.main:app --host 0.0.0.0 --port $PORT
# 5. Env var: GROQ_API_KEY = <YOUR_GROQ_API_KEY>
```

Or use the included `render.yaml` for one-click deployment.

> **Note:** `render.yaml` sets `RETRIEVAL_BACKEND=tfidf` for the free tier to avoid model download latency on cold starts. For a paid tier with persistent disk, set `RETRIEVAL_BACKEND=faiss`.

### Docker

```bash
# Build (downloads embedding models at build time)
docker build -t shl-recommender .

# Run
docker run -p 8000:8000 -e GROQ_API_KEY=<YOUR_GROQ_API_KEY> shl-recommender

# Or with compose
GROQ_API_KEY=<YOUR_GROQ_API_KEY> docker compose up
```

### Railway / Fly.io

```bash
# Railway
railway new && railway up
railway variables set GROQ_API_KEY=<YOUR_GROQ_API_KEY>

# Fly.io
fly launch
fly secrets set GROQ_API_KEY=<YOUR_GROQ_API_KEY>
```

---

## Environment Variables

| Variable               | Required | Default            | Description                         |
| ---------------------- | -------- | ------------------ | ----------------------------------- |
| `GROQ_API_KEY`         | ✓        | —                  | Groq API key                        |
| `RETRIEVAL_BACKEND`    |          | `faiss`            | `faiss` or `tfidf`                  |
| `EMBEDDING_MODEL`      |          | `all-MiniLM-L6-v2` | Sentence-transformer model          |
| `ENABLE_RERANKER`      |          | `true`             | CrossEncoder reranking              |
| `FAISS_TOP_K_RETRIEVE` |          | `20`               | Candidates before reranking         |
| `FAISS_TOP_K_RERANK`   |          | `5`                | Final results after reranking       |
| `LOG_LEVEL`            |          | `INFO`             | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `CORS_ORIGINS`         |          | `*`                | Comma-separated allowed origins     |

---

## Evaluation

```bash
# Against local service
python scripts/evaluate.py --url http://localhost:8000 --output eval_report.md

# Against production
python scripts/evaluate.py --url https://your-service.onrender.com
```

### Metrics

| Metric                 | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| **Schema compliance**  | All responses match `{reply, recommendations, end_of_conversation}` |
| **Routing accuracy**   | Vague→clarify, JD→recommend, injection→refuse                       |
| **Recall@K**           | Fraction of expected test types found in recommendations            |
| **Hallucination rate** | URLs returned not present in `catalog.json`                         |
| **Mean / P95 latency** | End-to-end response time                                            |
| **Clarification rate** | How often the system asks for more information                      |

---

## Design Decisions

### Why FAISS + CrossEncoder instead of just TF-IDF?

TF-IDF misses semantic relationships: a query for _"numerical reasoning"_ won't retrieve _"Verify-G+"_ even though that's the correct assessment — the overlap is semantic, not lexical. Sentence-transformers encode meaning, not just tokens.

The CrossEncoder reranking step (ms-marco-MiniLM-L-6-v2) sees the query and document together, enabling it to score relevance far more accurately than the bi-encoder. The two-stage pipeline (top-20 → rerank → top-5) gives recall of broad retrieval with precision of deep comparison.

Both models run fully locally — no API calls, no latency spikes, no cost.

The TF-IDF backend is retained as a fallback for environments without PyTorch (Render free tier, CI without GPU). A config line switches backends — the Supervisor is unaware.

### Why separate agents instead of one mega-prompt?

Each agent has one responsibility and one prompt file. The guardrail prompt is 15 lines; the recommend prompt is 30 lines. A single 200-line prompt handling all cases suffers from instruction interference at the edges and is impossible to test in isolation.

### Why structured intent extraction?

Routing based on `HiringIntent.action` is a deterministic switch on a typed field — explicit, unit-testable, debuggable. With implicit LLM routing, branching behavior is invisible and brittle on edge cases.

### Why stateless API?

The caller sends the full conversation on every request. This makes the service trivially horizontally scalable — any instance can handle any request. There is no session affinity requirement, no shared cache to synchronize, no session store to maintain.

### Why inject the full conversation into each agent?

The conversation history is the source of truth. Each agent reads what it needs from the full context rather than relying on inter-agent state passing, which would require shared memory and complicate horizontal scaling.

---

## Stack

| Component  | Choice                        | Rationale                                    |
| ---------- | ----------------------------- | -------------------------------------------- |
| LLM        | Llama 3.3 70B (Groq)          | Free tier, ~300 tok/s, OpenAI-compatible API |
| API        | FastAPI + Uvicorn             | Async, Pydantic-native, minimal overhead     |
| Retrieval  | FAISS + sentence-transformers | Semantic recall, fully local                 |
| Reranking  | CrossEncoder (ms-marco)       | Precision over recall                        |
| Config     | pydantic-settings             | Validated at startup, fails fast             |
| Deployment | Render / Docker               | Cold start within evaluator's 2-min window   |

---

## What Was Tried and Abandoned

**Few-shot examples in recommend prompt** — including 2 example conversations caused the model to mirror example phrasing on unusual inputs. Replaced with explicit decision rules.

**Two-LLM-call routing** — classify intent, then act — added 10–15s latency. Collapsed to single intent extraction call that returns `action` as part of structured output.

**Type-only metadata filtering before ranking** — filtering strictly by `test_type` before scoring excluded relevant entries when users hadn't specified type preferences. Changed to: apply type filter only when user explicitly requests it, otherwise retrieve across all types and let the reranker decide.

**Full catalog injection** — 89 entries × ~100 tokens ≈ 9K tokens per call. With retrieval, LLM sees only top-15 (~1.5K tokens). 83% token reduction, tighter focus, structurally harder hallucination.
