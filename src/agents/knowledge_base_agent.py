import os
import asyncio
from typing import TypedDict, Optional, Literal, List
from dotenv import load_dotenv

import tiktoken

from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

from typesafe_sdk import TypeSafeClient, Choice

from src.agents.context import AgentContext
from src.knowledge_base.retrieval import Retriever

load_dotenv()

FALLBACK_RESPONSE = "Your question is irrelevant to the given file."
encoder = tiktoken.get_encoding("o200k_base")

CHUNK_TOKEN_LIMIT = 15000


class RAGWorkflowState(TypedDict):
    messages: List[AnyMessage]
    retrieved_context: Optional[str]
    retrieval_status: Optional[Literal["relevant", "irrelevant", "no_context"]]
    chunk_store: List[dict] 
    new_chunks: List[dict]  
    current_query: str
    rewritten_query: str
    decision: Optional[Literal["retrieve", "reuse"]]
    retry_count: int


def preprocess_node(state: RAGWorkflowState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {"current_query": "", "rewritten_query": "", "retry_count": 0, "decision": None, "new_chunks": []}

    latest_msg = messages[-1]
    cleaned_query = latest_msg.content if hasattr(latest_msg, "content") else str(latest_msg)

    return {
        "current_query": cleaned_query,
        "rewritten_query": cleaned_query,
        "retry_count": 0,
        "decision": None,
        "new_chunks": [],
    }


def context_sufficiency_gate_node(state: RAGWorkflowState, runtime: Runtime[AgentContext]) -> dict:
    messages = state.get("messages", [])
    current_query = state.get("current_query", "").strip()
    store = state.get("chunk_store") or []

    if not store:
        return {"decision": "retrieve"}

    history = "\n".join(f"{m.type}: {m.content}" for m in messages[:-1])

    stored_chunks = "".join(
        f"\n--- Document Chunk {n} ---\n"
        f"Source: {c['source']}\n"
        f"Pages: {c['pages']}\n"
        f"Section: {c['section']}\n"
        f"Content: {c['content']}\n"
        for n, c in enumerate(reversed(store), start=1)
    )

    prompt_state = f"""<role>
You are an enterprise RAG context sufficiency evaluator. You must decide whether to RETRIEVE fresh documents or REUSE the already retrieved document chunks.
</role>

<critical_rule id="context_sufficiency_gate">
Default bias is RETRIEVE — you may only choose REUSE if ALL 4 criteria are strictly met:
1. COVERAGE: every part of the current question is directly addressed by those chunks — not just topically related.
2. NO NEW ENTITIES: the question introduces no new named entity, section, metric, date range, or sub-topic that was not already covered by prior retrieval.
3. NOT COMPARATIVE/EXPANSIVE: the question doesn't ask for something broader, deeper, or differently-scoped than what was already retrieved (e.g., 'give more examples', 'what about X instead', 'is there an exception to that').
4. NOT STALE-SENSITIVE: the question doesn't hinge on information that could exist elsewhere in the documents but wasn't part of the earlier retrieval.

If ANY of the four fails, or you are uncertain, you MUST RETRIEVE.
</critical_rule>

<messages>
{history}
</messages>

<retrieved_chunks>
{stored_chunks}
</retrieved_chunks>

<current_query>
{current_query}
</current_query>"""

    try:
        client = runtime.context.deps.jev
        response = client.system_one(
            model="jev-1.13",
            state=prompt_state,
            questions={
                "decision": Choice(
                    instructions="Based strictly on the 4 sufficiency rules, should the assistant RETRIEVE new documents or REUSE the visible chunks?",
                    criteria={
                        "retrieve": "Fails one or more sufficiency tests: query asks for new entities, comparative details, broader scope, or requires unseen doc info.",
                        "reuse": "Passes all 4 sufficiency tests: the query is a direct reformatting, clarification, or summary of what is ALREADY visible in the chunks without expanding scope."
                    }
                )
            }
        )
        decision = response.answers["decision"].choice
        if decision not in ("retrieve", "reuse"):
            decision = "retrieve"
    except Exception as e:
        print(f"JEV sufficiency decision error: {e}. Defaulting to 'retrieve'")
        decision = "retrieve"

    if decision == "reuse":
        return {"decision": "reuse", "retrieved_context": stored_chunks, "retrieval_status": "relevant"}
    return {"decision": "retrieve"}


def query_rewriter_node(state: RAGWorkflowState, runtime: Runtime[AgentContext]) -> dict:
    current_query = state.get("current_query", "")
    messages = state.get("messages", [])

    history = "\n".join(f"{m.type}: {m.content}" for m in messages[:-1])

    system_instruction = """You are a search query reformulation expert for an enterprise document search engine.
Your task is to rewrite the user's latest query into a single, fully self-contained search query.

Rules:
1. Resolve all pronouns and ambiguous references ("it", "that", "this one", "they", "the previous one") against the conversation history.
   Example: prior topic "attention mechanism" + latest query "give examples of it" -> "examples of attention mechanism".
2. Do NOT answer the question.
3. Keep the query concise, keyword-rich, and directly suitable for vector similarity search.
4. Output ONLY the standalone search query string with no explanation or punctuation prefix."""

    try:
        response = runtime.context.deps.llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=(
                f"Conversation so far:\n{history}\n\n"
                f"Latest user message:\n{current_query}\n\n"
                "Rewrite the latest message as a standalone search query."
            )),
        ])
        rewritten = response.content.strip().strip('"').strip("'")
        if not rewritten:
            rewritten = current_query
    except Exception as e:
        print(f"Query rewrite error: {e}")
        rewritten = current_query

    return {"rewritten_query": rewritten}


async def retrieval_node(state: RAGWorkflowState, runtime: Runtime[AgentContext]) -> dict:
    query = state.get("rewritten_query") or state.get("current_query", "")

    admin_id = runtime.context.admin_id
    if not admin_id:
        print("Retrieval error: admin_id missing in context")
        return {"new_chunks": []}

    retriever_inst = runtime.context.deps.kb_retriever
    results = await asyncio.to_thread(retriever_inst.retrieve, query, admin_id)

    new_chunks = [
        {
            "chunk_id": chunk.metadata.get("chunk_id") or chunk.page_content,
            "source": chunk.metadata.get("source", "Unknown"),
            "pages": chunk.metadata.get("pages", "N/A"),
            "section": chunk.metadata.get("section", "N/A"),
            "content": chunk.page_content,
            "tokens": len(encoder.encode(chunk.page_content)),
        }
        for chunk in results
    ]

    capped, budget = [], 0
    for chunk in new_chunks:
        if capped and budget + chunk["tokens"] > CHUNK_TOKEN_LIMIT:
            break
        capped.append(chunk)
        budget += chunk["tokens"]
    new_chunks = capped

    return {"new_chunks": new_chunks}


def retrieval_evaluation_node(state: RAGWorkflowState, runtime: Runtime[AgentContext]) -> dict:
    new_chunks = state.get("new_chunks") or []
    current_query = state.get("current_query", "")
    rewritten_query = state.get("rewritten_query", current_query)

    if not new_chunks:
        return {"retrieval_status": "no_context"}

    new_chunks_text = "".join(
        f"\n--- Document Chunk {n} ---\n"
        f"Source: {c['source']}\n"
        f"Pages: {c['pages']}\n"
        f"Section: {c['section']}\n"
        f"Content: {c['content']}\n"
        for n, c in enumerate(new_chunks, start=1)
    )

    prompt_state = f"""User Query:
{current_query}

Search Query Used:
{rewritten_query}

Retrieved Document Chunks:
{new_chunks_text}"""

    try:
        client = runtime.context.deps.jev
        response = client.system_one(
            model="jev-1.13",
            state=prompt_state,
            questions={
                "relevance": Choice(
                    instructions="Do the retrieved document chunks contain relevant information that directly addresses or helps answer the user query?",
                    criteria={
                        "relevant": "The chunks contain relevant facts, figures, policies, or explanations that help answer the user query.",
                        "irrelevant": "The chunks are off-topic, unrelated, or do not contain information to address the user query."
                    }
                )
            }
        )
        relevance_choice = response.answers["relevance"].choice
        status = "relevant" if relevance_choice == "relevant" else "irrelevant"
    except Exception as e:
        print(f"JEV retrieval evaluation error: {e}. Defaulting to 'relevant'")
        status = "relevant"

    return {"retrieval_status": status}


def broaden_query_node(state: RAGWorkflowState, runtime: Runtime[AgentContext]) -> dict:
    current_query = state.get("current_query", "")
    previous_query = state.get("rewritten_query", current_query)
    retry_count = state.get("retry_count", 0)

    prompt = f"""The previous document search for the user question failed to retrieve relevant documents.
Formulate a BROADER, differently-worded search query that uses broader synonyms, related technical terms, or higher-level keywords to search the user's uploaded files.

Original User Question: {current_query}
Previous Failed Search: {previous_query}

Provide ONLY the reformulated broader search query:"""

    try:
        response = runtime.context.deps.llm.invoke(prompt)
        broader_query = response.content.strip().strip('"').strip("'")
        if not broader_query:
            broader_query = f"{current_query} overview details"
    except Exception as e:
        print(f"Broaden query error: {e}")
        broader_query = f"{current_query} overview details"

    return {
        "rewritten_query": broader_query,
        "retry_count": retry_count + 1,
    }


def fallback_node(state: RAGWorkflowState) -> dict:
    return {"retrieved_context": ""}


def add_chunks_node(state: RAGWorkflowState) -> dict:
    store = state.get("chunk_store") or []
    new_chunks = state.get("new_chunks") or []

    new_ids = {c["chunk_id"] for c in new_chunks}
    store = [c for c in store if c["chunk_id"] not in new_ids]

    return {"chunk_store": store + new_chunks}


def trim_oldest_node(state: RAGWorkflowState) -> dict:
    store = state.get("chunk_store") or []
    return {"chunk_store": store[1:]}


def build_context_node(state: RAGWorkflowState) -> dict:
    store = state.get("chunk_store") or []

    context = "".join(
        f"\n--- Document Chunk {n} ---\n"
        f"Source: {c['source']}\n"
        f"Pages: {c['pages']}\n"
        f"Section: {c['section']}\n"
        f"Content: {c['content']}\n"
        for n, c in enumerate(reversed(store), start=1)
    )
    return {"retrieved_context": context}

def route_after_sufficiency(state: RAGWorkflowState) -> Literal["query_rewriter", "__end__"]:
    if state.get("decision") == "reuse":
        return END
    return "query_rewriter"


def route_after_retrieval_eval(state: RAGWorkflowState) -> Literal["add_chunks", "broaden_query", "fallback"]:
    status = state.get("retrieval_status", "no_context")
    retry_count = state.get("retry_count", 0)

    if status == "relevant":
        return "add_chunks"

    if retry_count < 1:
        return "broaden_query"
    return "fallback"


def route_after_add(state: RAGWorkflowState) -> Literal["trim_oldest", "build_context"]:
    store = state.get("chunk_store") or []
    new_count = len(state.get("new_chunks") or [])
    total_tokens = sum(c["tokens"] for c in store)

    if total_tokens > CHUNK_TOKEN_LIMIT and len(store) > new_count:
        return "trim_oldest"
    return "build_context"


def create_workflow_graph() -> StateGraph:
    workflow = StateGraph(RAGWorkflowState, context_schema=AgentContext)

    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("context_sufficiency_gate", context_sufficiency_gate_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("retrieval_eval", retrieval_evaluation_node)
    workflow.add_node("broaden_query", broaden_query_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("add_chunks", add_chunks_node)
    workflow.add_node("trim_oldest", trim_oldest_node)
    workflow.add_node("build_context", build_context_node)

    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "context_sufficiency_gate")

    workflow.add_conditional_edges(
        "context_sufficiency_gate",
        route_after_sufficiency,
        {
            "query_rewriter": "query_rewriter",
            END: END,
        }
    )

    workflow.add_edge("query_rewriter", "retrieval")
    workflow.add_edge("retrieval", "retrieval_eval")

    workflow.add_conditional_edges(
        "retrieval_eval",
        route_after_retrieval_eval,
        {
            "add_chunks": "add_chunks",
            "broaden_query": "broaden_query",
            "fallback": "fallback",
        }
    )

    workflow.add_edge("broaden_query", "retrieval")
    workflow.add_edge("fallback", END)

    workflow.add_conditional_edges(
        "add_chunks",
        route_after_add,
        {
            "trim_oldest": "trim_oldest",
            "build_context": "build_context",
        }
    )
    workflow.add_conditional_edges(
        "trim_oldest",
        route_after_add,
        {
            "trim_oldest": "trim_oldest",
            "build_context": "build_context",
        }
    )
    workflow.add_edge("build_context", END)

    return workflow

def build_agent():
    workflow = create_workflow_graph()
    return workflow.compile(checkpointer=True)


build_workflow = build_agent