from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List, Literal, Optional, Union
from langgraph.graph.message import add_messages, RemoveMessage
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from langgraph.types import Command
from typesafe_sdk import TypeSafeClient, Choice,Score
import tiktoken

from src.agents.knowledge_base_agent import build_agent as build_kb_agent
from src.agents.database_lookup_agent import build_agent as build_db_agent, AgentContext
import json
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

load_dotenv()

kb_subgraph = build_kb_agent()
db_subgraph = build_db_agent()

SUMMARY_TOKEN_LIMIT = 4000
SUMMARY_KEEP_LAST = 2

encoder = tiktoken.get_encoding("o200k_base")

class State(TypedDict) :
    messages : Annotated[List[AnyMessage], add_messages]
    route: Optional[Literal["knowledge_base", "database_lookup", "human_escalate", "Irrelevant"]]
    urgency: Optional[Literal["low", "medium", "high"]]
    urgency_score: Optional[int]
    sources: Optional[List[Literal["knowledge_base", "database_lookup"]]]
    retrieved_context: Optional[str]
    db_rows: List[dict]
    db_status: Optional[Literal["ok", "failed"]]

def message_tokens(messages) -> int:
    return sum(len(encoder.encode(str(m.content))) for m in messages)

def summarize_history_node(state: State, runtime: Runtime[AgentContext]) -> Command:
    messages = state.get("messages") or []
    if len(messages) <= SUMMARY_KEEP_LAST:
        return Command(update={"messages": []})

    to_summarize, keep = messages[:-SUMMARY_KEEP_LAST], messages[-SUMMARY_KEEP_LAST:]

    summary = runtime.context.deps.summarizer_llm.invoke([
        SystemMessage(content=(
            "Summarize the conversation so far for a customer support agent. "
            "Preserve the customer's issue, any account/order details, what was "
            "decided, and any open questions. Be concise but complete."
        )),
        *to_summarize,
    ]).content

    return Command(update={
        "messages": [RemoveMessage(id=m.id) for m in messages]
                     + [AIMessage(content=f"Summary of the earlier conversation: {summary}")]
                     + [type(m)(content=m.content) for m in keep],
    })


def route_before_router(state: State) -> str:
    if message_tokens(state.get("messages") or []) > SUMMARY_TOKEN_LIMIT:
        return "summarize"
    return "router"

ROUTE_PROB_THRESHOLD = 0.375

def decide_sources(probs) -> List[str]:
    sources = []
    for label in ("knowledge_base", "database_lookup"):
        if (probs.get(label) or 0.0) >= ROUTE_PROB_THRESHOLD:
            sources.append(label)
    if not sources:
        winner = max(probs, key=lambda k: probs[k]) if probs else None
        if winner in ("knowledge_base", "database_lookup"):
            sources = [winner]
    return sources

def routing_decision_node(state : State, runtime: Runtime[AgentContext]) :
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
        response = runtime.context.deps.jev.system_one(
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
        return {"route": None, "urgency": None, "urgency_score": None, "sources": None}


def route_after_router(state: State) -> Union[str, List[str]]:
    if state.get("urgency") == "high":
        return "human_escalate"
    sources = state.get("sources") or []
    if len(sources) == 2:
        return ["knowledge_base", "database_lookup"]
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
        context=runtime.context,
    )
    return {"retrieved_context": result.get("retrieved_context")}


async def answer_node(state: State, runtime: Runtime[AgentContext]) -> dict:
    sources = state.get("sources") or []
    has_kb = "knowledge_base" in sources
    has_db = "database_lookup" in sources

    db_rows = state.get("db_rows") or []
    db_ok = state.get("db_status") == "ok"
    context = (state.get("retrieved_context") or "").strip()

    parts = []
    if has_kb and context:
        parts.append(f"<documents>{context}</documents>")
    if has_db and db_ok and db_rows:
        parts.append(f"<database_rows>{json.dumps(db_rows)}</database_rows>")

    if not parts:
        if has_kb and not context:
            return {"messages": [AIMessage(content="I couldn't find anything about that in the company's documents.")]}
        if has_db and not db_ok:
            return {"messages": [AIMessage(content="I couldn't retrieve that information right now. Please try again or rephrase your request.")]}
        if has_db and not db_rows:
            return {"messages": [AIMessage(content="I couldn't find any matching records for your account.")]}
        return {"messages": [AIMessage(content="I couldn't find any information to answer that. Please try rephrasing.")]}

    source = "\n".join(parts)

    messages = state.get("messages")
    response = await runtime.context.deps.llm.ainvoke([
        SystemMessage(content=(
            "You are a customer support assistant. Answer the user's latest question using only the "
            "information in the <context> block below. If it does not contain the answer, say you "
            "could not find it. Do not invent values. Mention the source and page when it helps."
        )),

        HumanMessage(content=f"<context>\n{source}\n</context>"),
        *messages,
    ])
    return {"messages": [AIMessage(content=response.content)]}


class TicketDraft(BaseModel):
    subject: str = Field(description="Short title of the issue, at most 120 characters")
    description: str = Field(description="Clear summary of the customer's issue and what they need, based on the conversation")
    category: Literal["billing", "technical", "account", "other"]
    priority: Optional[Literal["low", "medium", "high"]] = None




async def human_escalate_node(state: State, runtime: Runtime[AgentContext], config: RunnableConfig) -> dict:
    messages = state.get("messages")
    priority = state.get("urgency") or "medium"

    ticket_llm = runtime.context.deps.llm.with_structured_output(TicketDraft)
    draft = await ticket_llm.ainvoke([
        SystemMessage(content=(
            "You create support tickets. From the conversation, write a short subject, a clear "
            "description of the customer's issue and what they need, and a category "
            "(billing, technical, account, or other). Do not set a priority."
        )),
        *messages,
    ])

    ticket = {
        "user_id": runtime.context.user_id,
        "admin_id": runtime.context.admin_id,
        "thread_id": config["configurable"]["thread_id"],
        **draft.model_dump(),
        "priority": priority,   # router's urgency is authoritative; placed last so it wins
    }

    try:
        result = await runtime.context.deps.supabase.table("tickets").insert(ticket).execute()
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
    workflow.add_node("human_escalate", human_escalate_node)
    workflow.add_node("irrelevant", irrelevant_node)
    workflow.add_node("summarize", summarize_history_node)

    workflow.add_conditional_edges(
        START,
        route_before_router,
        {
            "summarize": "summarize",
            "router": "router",
        }
    )

    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "knowledge_base": "knowledge_base",
            "database_lookup": "database_lookup",
            "human_escalate": "human_escalate",
            "irrelevant": "irrelevant",
        }
    )

    workflow.add_edge("summarize", "router")
    workflow.add_edge("knowledge_base", "answer")
    workflow.add_edge("database_lookup", "answer")
    workflow.add_edge("answer", END)
    workflow.add_edge("human_escalate", END)
    workflow.add_edge("irrelevant", END)

    return workflow


def build_agent(checkpointer):
    return create_graph().compile(checkpointer=checkpointer)