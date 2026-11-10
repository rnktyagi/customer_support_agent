import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, Header
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text

from src.graph import AgentContext

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


async def get_engine(request: Request, admin_id: str):
    supabase = request.app.state.supabase
    admin = await supabase.table("admins").select("database_url").eq(
        "id", admin_id
    ).maybe_single().execute()
    if admin is None or admin.data is None or not admin.data.get("database_url"):
        raise HTTPException(
            status_code=400,
            detail="No database uploaded for this admin. Upload a database first.",
        )
    return create_engine(admin.data["database_url"])


async def get_admin_config(request: Request, admin_id: str):
    supabase = request.app.state.supabase
    admin = await supabase.table("admins").select(
        "database_url, identity_table, pk_column, email_column"
    ).eq("id", admin_id).maybe_single().execute()
    if admin is None or admin.data is None:
        raise HTTPException(status_code=403, detail="Invalid admin_id.")
    return admin.data


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    token = authorization.split(" ", 1)[1].strip()

    user = await request.app.state.supabase_auth.auth.get_user(token)
    if user is None or user.user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    supabase_user_id = user.user.id
    email = user.user.email

    customer = await request.app.state.supabase.table("customers").select("admin_id").eq(
        "user_id", supabase_user_id
    ).maybe_single().execute()
    if customer is None or customer.data is None:
        raise HTTPException(status_code=403, detail="This is not a user account.")

    admin_id = customer.data["admin_id"]

    config = await get_admin_config(request, admin_id)
    identity_table = config.get("identity_table")
    pk_column = config.get("pk_column")
    email_column = config.get("email_column")

    if not identity_table or not pk_column or not email_column:
        raise HTTPException(
            status_code=400,
            detail="Admin has not configured identity table, PK column, or email column.",
        )

    engine = await get_engine(request, admin_id)

    pk = None
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT {pk_column} FROM {identity_table} WHERE {email_column} = :email"),
            {"email": email},
        ).scalar()
        pk = str(result) if result else None

    if not pk:
        raise HTTPException(status_code=404, detail="Customer email not found in admin's database.")

    context = AgentContext(
        admin_id=admin_id,
        user_id=supabase_user_id,
        customer_pk=pk,
        tenant_pk_column=pk_column,
        engine=engine,
    )

    agent = request.app.state.agent

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