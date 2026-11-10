import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Literal, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, AIMessage
from typesafe_sdk import TypeSafeClient, Choice,Score

from src.agents.knowledge_base_agent import build_agent as build_kb_agent, KBContext
from src.agents.database_lookup_agent import build_agent as build_db_agent, AgentContext
import json
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.runtime import Runtime
import asyncio
from functools import lru_cache
from pydantic import BaseModel, Field
from supabase import create_client, Client

load_dotenv()

kb_subgraph = build_kb_agent()
db_subgraph = build_db_agent()

jev_client = TypeSafeClient(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api",
)

answer_llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0,
)

class State(TypedDict) :
    messages : Annotated[List[AnyMessage], add_messages]
    route: Optional[Literal["knowledge_base", "database_lookup", "human_escalate", "Irrelevant"]]
    urgency: Optional[Literal["low", "medium", "high"]]
    urgency_score: Optional[int]
    sources: Optional[List[Literal["knowledge_base", "database_lookup"]]]
    retrieved_context: Optional[str]
    db_rows: List[dict]
    db_status: Optional[Literal["ok", "failed"]]


# A retrieval source is included when JEV assigns it at least this much probability.
# Below the threshold the query is treated as single-source.
ROUTE_PROB_THRESHOLD = 0.65


def decide_sources(probs) -> List[str]:
    """Return the set of retrieval sources JEV assigns meaningful probability to."""
    sources = []
    for label in ("knowledge_base", "database_lookup"):
        if (probs.get(label) or 0.0) >= ROUTE_PROB_THRESHOLD:
            sources.append(label)
    if not sources:
        winner = max(probs, key=lambda k: probs[k]) if probs else None
        if winner in ("knowledge_base", "database_lookup"):
            sources = [winner]
    return sources

def routing_decision_node(state : State) :
    messages = state.get("messages")
    current_query = messages[-1].content
    history = "\n".join(f"{m.type}: {m.content}"for m in messages[:-1])
    PROMPT=f"""
<role>
You are a decision model working in a customer support agent designed explicitly for routing between different supporting use cases based on the user query
</role>
<routing_decision>
According to the given user query, you have to decide between the given 4 different routing decisions :
1. Knowledge Base : If the user query is related to the company's knowledge base like policies, terms and conditions etc.
2. Database Lookup : If the user query can be answered by providing the details to the user according to its user data from the company's database
3. Human Escalation : If the user query involves frustration/dissatisfaction and requires urgent resolution by a human support.
4. Irrelvant : If the user query is completely irrelevant to your role kindly route it as irrelvant.
"""
    try :
        response = jev_client.system_one(
            model="jev-1.13",
            state=f"{PROMPT}\n\nConversation History:\n{history}\n\nCurrent Query:\n{current_query}",
            questions={
                "route": Choice(
                    instructions="Based on the current query and conversation history, which routing decision applies?",
                    criteria={
                        "knowledge_base": "Answerable from company policies, terms, or general info.",
                        "database_lookup": "Needs the user's own account/order data.",
                        "human_escalate": "Frustration, dissatisfaction, or needs urgent human resolution.",
                        "Irrelevant": "Unrelated to customer support at all.",
                        },
                    ),
                "urgency": Score(
                    instructions="Determine the urgency of the customer's request based on frustration, potential harm, security concerns, or time sensitivity.",
                    criteria=[
                        "low: Normal customer request with no immediate risk or significant urgency.",
                        "medium: Important issue that should be resolved promptly, but there is no immediate serious risk.",
                        "high: Requires urgent attention because of security concerns, financial harm, serious dissatisfaction, or significant time sensitivity."
                    ]
                )
            }
        )
        route=response.answers['route'].choice
        urgency_score=response.answers['urgency'].score
        urgency = {0: "low", 1: "medium", 2: "high"}.get(urgency_score)
        sources = decide_sources(response.answers['route'].probabilities)
        return {"route" : route , "urgency" : urgency, "urgency_score" : urgency_score, "sources" : sources}
    except Exception as e :
        print(f"{e}")
        return {"route": None, "urgency": None}


def route_after_router(state: State) -> Literal["knowledge_base", "database_lookup", "both_sources", "human_escalate", "irrelevant"]:
    if state.get("urgency") == "high":
        return "human_escalate"
    sources = state.get("sources") or []
    if len(sources) == 2:
        return "both_sources"
    if state.get("route") == "database_lookup":
        return "database_lookup"
    if state.get("route") == "human_escalate":
        return "human_escalate"
    if state.get("route") == "Irrelevant":
        return "irrelevant"
    return "knowledge_base"


async def database_lookup_node(state: State, runtime: Runtime[AgentContext], config: RunnableConfig) -> dict:
    result = await db_subgraph.ainvoke(
        {"messages": state["messages"]},
        config,
        context=runtime.context,
    )
    return {"db_rows": result.get("db_rows", []), "db_status": result.get("db_status")}


async def knowledge_base_node(state: State, runtime: Runtime[AgentContext], config: RunnableConfig) -> dict:
    result = await kb_subgraph.ainvoke(
        {"messages": state["messages"]},
        config,
        context=KBContext(admin_id=runtime.context.admin_id),
    )
    return {"retrieved_context": result.get("retrieved_context")}


async def both_sources_node(state: State, runtime: Runtime[AgentContext], config: RunnableConfig) -> dict:
    """Run the KB and DB subgraphs sequentially and merge their results.

    Sequential (not parallel) on purpose: both subgraphs use per-thread
    checkpointers, so invoking them concurrently under the same thread_id
    would cause checkpoint namespace conflicts.
    """
    kb_result = await kb_subgraph.ainvoke(
        {"messages": state["messages"]},
        config,
        context=KBContext(admin_id=runtime.context.admin_id),
    )
    db_result = await db_subgraph.ainvoke(
        {"messages": state["messages"]},
        config,
        context=runtime.context,
    )
    return {
        "retrieved_context": kb_result.get("retrieved_context"),
        "db_rows": db_result.get("db_rows", []),
        "db_status": db_result.get("db_status"),
    }


async def answer_node(state: State) -> dict:
    sources = state.get("sources") or []
    has_kb = "knowledge_base" in sources
    has_db = "database_lookup" in sources

    db_rows = state.get("db_rows") or []
    db_ok = state.get("db_status") == "ok"
    context = (state.get("retrieved_context") or "").strip()

    # Neither source produced anything usable.
    if not has_kb and not has_db:
        return {"messages": [AIMessage(content="I couldn't find any information to answer that. Please try rephrasing.")]}
    if has_kb and not context:
        return {"messages": [AIMessage(content="I couldn't find anything about that in the company's documents.")]}
    if has_db and (not db_ok or not db_rows):
        return {"messages": [AIMessage(content="I couldn't retrieve that information right now. Please try again or rephrase your request.")]}

    parts = []
    if has_kb:
        parts.append(f"<documents>{context}</documents>")
    if has_db:
        parts.append(f"<database_rows>{json.dumps(db_rows)}</database_rows>")
    source = "\n".join(parts)

    messages = state.get("messages")
    response = await answer_llm.ainvoke([
        SystemMessage(content=(
            "You are a customer support assistant. Answer the user's latest question using only the "
            "information below. If it does not contain the answer, say you could not find it. "
            "Do not invent values. Mention the source and page when it helps.\n\n"
            f"{source}"
        )),
        *messages,
    ])
    return {"messages": [AIMessage(content=response.content)]}


class TicketDraft(BaseModel):
    subject: str = Field(description="Short title of the issue, at most 120 characters")
    description: str = Field(description="Clear summary of the customer's issue and what they need, based on the conversation")
    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high"]


ticket_llm = answer_llm.with_structured_output(TicketDraft)


@lru_cache(maxsize=1)
def get_supabase() -> Client:
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])


async def human_escalate_node(state: State, runtime: Runtime[AgentContext], config: RunnableConfig) -> dict:
    messages = state.get("messages")

    draft = await ticket_llm.ainvoke([
        SystemMessage(content=(
            "You create support tickets. From the conversation, write a short subject, a clear "
            "description of the customer's issue and what they need, a category, and a priority "
            "based on urgency, frustration, and any security or financial risk."
        )),
        *messages,
    ])

    ticket = {
        "user_id": runtime.context.user_id,
        "admin_id": runtime.context.admin_id,
        "thread_id": config["configurable"]["thread_id"],
        **draft.model_dump(),
    }

    try:
        result = await asyncio.to_thread(get_supabase().table("tickets").insert(ticket).execute)
        ticket_id = result.data[0]["id"]
    except Exception as e:
        print(f"Ticket insert error: {e}")
        return {"messages": [AIMessage(content="I'm sorry, I wasn't able to log your request just now. Please try again in a moment, or contact our support team directly.")]}

    return {"messages": [AIMessage(content=(
        "Thank you for reaching out, and I'm sorry for the trouble. I've escalated your request to our "
        f"support team under ticket #{ticket_id}. A member of our team will contact you as soon as possible."
    ))]}


def irrelevant_node(state: State) -> dict:
    return {"messages": [AIMessage(content="I can only help with questions about our company's policies, your account, or support issues.")]}


def create_graph() -> StateGraph:
    workflow = StateGraph(State, context_schema=AgentContext)

    workflow.add_node("router", routing_decision_node)
    workflow.add_node("knowledge_base", knowledge_base_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("database_lookup", database_lookup_node)
    workflow.add_node("both_sources", both_sources_node)
    workflow.add_node("human_escalate", human_escalate_node)
    workflow.add_node("irrelevant", irrelevant_node)

    workflow.add_edge(START, "router")

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "knowledge_base": "knowledge_base",
            "database_lookup": "database_lookup",
            "both_sources": "both_sources",
            "human_escalate": "human_escalate",
            "irrelevant": "irrelevant",
        }
    )

    workflow.add_edge("knowledge_base", "answer")
    workflow.add_edge("database_lookup", "answer")
    workflow.add_edge("both_sources", "answer")
    workflow.add_edge("answer", END)
    workflow.add_edge("human_escalate", END)
    workflow.add_edge("irrelevant", END)

    return workflow


def build_agent(checkpointer):
    return create_graph().compile(checkpointer=checkpointer)