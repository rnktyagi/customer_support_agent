import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Header
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from src.dependencies import (
    resolve_customer_context,
    get_chat_context,
    get_agent,
)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    customer_ctx=Depends(resolve_customer_context),
    context=Depends(get_chat_context),
    agent=Depends(get_agent),
):
    supabase_user_id, admin_id, email = customer_ctx

    thread_id = body.thread_id or os.urandom(8).hex()
    config_dict = {"configurable": {"thread_id": thread_id}}

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config=config_dict,
            context=context,
        )
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    reply = result["messages"][-1].content
    return ChatResponse(reply=reply, thread_id=thread_id)