# Support Agent

An AI-powered customer support agent that routes queries to the appropriate knowledge source (company policies via RAG, customer data via SQL), generates answers with citations, and escalates to human support via Jira/Slack when needed.

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit │────▶│    FastAPI       │────▶│   LangGraph     │
│   Frontend  │     │    Backend       │     │   Agent Graph   │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    ▼                                 ▼                                 ▼
            ┌───────────────┐               ┌─────────────────┐              ┌──────────────────┐
            │ Knowledge Base│               │ Database Lookup │              │   Escalation     │
            │   (RAG)       │               │   (Text-to-SQL) │              │   Pipeline       │
            └───────────────┘               └─────────────────┘              └──────────────────┘
                    │                                 │                                 │
                    ▼                                 ▼                                 ▼
            ┌───────────────┐               ┌─────────────────┐              ┌──────────────────┐
            │  ChromaDB     │               │  PostgreSQL     │              │  Supabase +      │
            │  Vector Store │               │  Customer DB    │              │  Jira + Slack    │
            └───────────────┘               └─────────────────┘              └──────────────────┘
```

---

## Components

### 1. Routing Layer (`src/graph.py`)
- **LLM-based router** (via TypeSafe SDK) classifies queries into:
  - `knowledge_base` — Company policies, terms, general info
  - `database_lookup` — Customer-specific account/order data
  - `escalate` — Frustration, urgency, needs human
  - `irrelevant` — Out of scope
- **Urgency scoring** (low/medium/high) drives escalation logic

### 2. Knowledge Base Agent (`src/agents/knowledge_base_agent.py`)
- **Context sufficiency gate** — Reuses prior chunks if they fully answer the new query, otherwise retrieves
- **Query rewriting** — Resolves pronouns/references against history for standalone search
- **Retrieval + evaluation** — Vector search (ChromaDB) with LLM relevance check
- **Broadening + fallback** — Reformulates query on miss, gives up after retry
- **Token-bounded chunk store** — Maintains rolling context within `CHUNK_TOKEN_LIMIT`

### 3. Database Lookup Agent (`src/agents/database_lookup_agent.py`)
- **Text-to-SQL pipeline** with safety guardrails:
  - Schema retrieval (vector search over indexed tables)
  - SQL generation with dialect awareness
  - **Safety check** (via LLM) — Read-only, tenant-scoped via `:tenant_user_pk`
  - Execution with automatic retry on error (broaden query → fallback)

### 4. Escalation Pipeline
```
answer ─┐
        ├──▶ combine (sync barrier) ▶ finalize ▶ user
ticket ─┼──▶ jira (parallel) ──▶
        └──▶ slack (parallel) ──▶
```
- **ticket** — LLM drafts ticket → inserts into Supabase `tickets` table
- **jira** — Creates Jira issue (parallel)
- **slack** — Posts notification to Slack channel (parallel)
- **finalize** — Combines answer + ticket reference for user

### 5. Auth & Multi-tenancy (`src/routers/auth.py`, `src/dependencies.py`)
- **Admin flow**: Signup → receives `admin_code` → shares with customers
- **Customer flow**: Signup with `admin_code` → linked to admin
- **Per-request context**: `resolve_customer_context` → `get_chat_context` builds `AgentContext` with admin-scoped config (Jira/Slack credentials)

### 6. Admin Dashboard (`src/frontend/main.py`)
- Database connection (URL, identity table, PK/email columns) + schema indexing
- Jira integration config (base URL, email, token, project key, issue type)
- Slack integration config (bot token, channel ID)
- Knowledge base document upload (PDF, DOCX, TXT, MD, CSV, XLSX)

---

## Data Flow

```
User Query
    │
    ▼
/chat/chat (FastAPI)
    │
    ▼
resolve_customer_context (validates JWT, gets admin_id, customer_pk)
    │
    ▼
get_chat_context (builds AgentContext with admin-scoped Jira/Slack config)
    │
    ▼
LangGraph Agent (MemorySaver checkpointer per thread_id)
    │
    ├──▶ summarize_history (if token limit exceeded)
    │
    ├──▶ routing_decision_node (LLM classifies + urgency)
    │
    ├──▶ Conditional: knowledge_base / database_lookup / irrelevant / escalate
    │       │
    │       ├──▶ knowledge_base → RAG subgraph → retrieved_context
    │       ├──▶ database_lookup → SQL subgraph → db_rows
    │       └──▶ irrelevant → static response
    │
    ├──▶ retrieval_join (no-op sync)
    │
    ├──▶ route_after_retrieval → answer (+ ticket if should_escalate)
    │
    ├──▶ answer_node (LLM synthesizes answer from context)
    │
    ├──▶ ticket_node (drafts + Supabase insert)
    │       │
    │       ├──▶ jira_node (parallel)
    │       └──▶ slack_node (parallel)
    │
    ├──▶ combine (sync barrier, 3 incoming edges)
    │
    └──▶ finalize_response_node (combines answer + ticket_display)
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | Service role key (backend) |
| `SUPABASE_ANON_KEY` | Anon key (auth) |
| `OPENROUTER_API_KEY` | TypeSafe/JeV API key |
| `GROQ_API_KEY` | Groq API key for LLM endpoints |

---

## Running Locally

```bash
# Backend
cd support_agent
pip install -e .
uvicorn src.server:app --reload --port 8000

# Frontend (separate terminal)
streamlit run src/frontend/main.py
```

---

## API Endpoints

### Auth
- `POST /auth/admin/signup` — Admin registration
- `POST /auth/admin/signin` — Admin login (returns config + access_token)
- `POST /auth/user/signup` — Customer registration (requires admin_code)
- `POST /auth/user/signin` — Customer login

### Chat
- `POST /chat/chat` — Main chat endpoint (Bearer token required)

### Upload (Admin only, Bearer or admin_id)
- `POST /upload/database` — Save DB URL + index schema
- `POST /upload/integrations` — Update Jira/Slack config (partial)
- `POST /upload/knowledge_base` — Upload & index documents

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **LangGraph + Runtime Context** | Explicit dependency injection, per-request config via `AgentContext` |
| **TypeSafe/JeV for routing** | Structured output (Choice/Score) with probability distributions |
| **Dual subgraph agents** | Separation of concerns: RAG vs SQL have different retrieval/eval logic |
| **Reducer-annotated State** | Handles concurrent writes from parallel branches (`combine` sync) |
| **Per-request Deps clone** | Admin-specific Jira/Slack config without mutating shared state |
| **Tenant-scoped SQL** | Every query bound to `:tenant_user_pk` at generation + guard |
| **Streamlit frontend** | Rapid admin dashboard + customer chat without separate React build |

---

## Project Structure

```
src/
├── agents/
│   ├── context.py              # Deps, JiraConfig, SlackConfig, AgentContext
│   ├── knowledge_base_agent.py # RAG subgraph
│   └── database_lookup_agent.py# Text-to-SQL subgraph
├── database_lookup/
│   ├── embedding.py            # SchemeEmbedding (index schema → Chroma)
│   ├── retrieval.py            # SchemeRetriever (vector search)
│   └── scheme.py               # Scheme dataclass
├── knowledge_base/
│   ├── embedding.py            # Embedding (index docs → Chroma)
│   ├── ingestion_and_chunking.py
│   └── retrieval.py            # Retriever (vector search)
├── routers/
│   ├── auth.py                 # Admin/customer auth
│   ├── chat.py                 # Chat endpoint
│   └── upload.py               # DB, integrations, KB upload
├── dependencies.py             # FastAPI deps (auth, context, config)
├── graph.py                    # Main LangGraph agent
├── server.py                   # FastAPI app + lifespan
└── frontend/
    └── main.py                 # Streamlit admin dashboard + customer chat
```