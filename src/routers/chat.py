import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import create_engine

from src.graph import AgentContext

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    admin_id: str
    user_id: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


def get_engine(request: Request, admin_id: str):
    """Create the SQLAlchemy engine from the URL stored in the admins table."""
    supabase = request.app.state.supabase
    admin = supabase.table("admins").select("database_url").eq(
        "id", admin_id
    ).maybe_single().execute()
    if admin is None or admin.data is None or not admin.data.get("database_url"):
        raise HTTPException(
            status_code=400,
            detail="No database uploaded for this admin. Upload a database first.",
        )
    return create_engine(admin.data["database_url"])


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest, request: Request):
    engine = get_engine(request, body.admin_id)

    context = AgentContext(
        admin_id=body.admin_id,
        user_id=body.user_id,
        engine=engine,
    )

    agent = request.app.state.agent

    thread_id = body.thread_id or os.urandom(8).hex()
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config=config,
            context=context,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    reply = result["messages"][-1].content
    return ChatResponse(reply=reply, thread_id=thread_id)