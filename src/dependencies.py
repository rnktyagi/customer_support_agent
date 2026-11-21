from typing import Tuple, Optional
from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy import create_engine, text

import os
import httpx



def get_supabase(request: Request):
    return getattr(request.app.state, "supabase", None)

def get_supabase_auth(request: Request):
    return getattr(request.app.state, "supabase_auth", None)

def get_scheme_embedding(request: Request):
    return getattr(request.app.state, "scheme_embedding", None)

def get_kb_embedding(request: Request):
    return getattr(request.app.state, "kb_embedding", None)

def get_deps(request: Request):
    return getattr(request.app.state, "deps", None)

def get_agent(request: Request):
    return getattr(request.app.state, "agent", None)


async def get_current_user(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    supabase_auth=Depends(get_supabase_auth),
):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    try:
        user = await supabase_auth.auth.get_user(token)
    except Exception:
        raise HTTPException(401, "Invalid or expired token.")
    if not user or not user.user:
        raise HTTPException(401, "Invalid or expired token.")
    return user.user


async def resolve_customer_context(
    user=Depends(get_current_user),
    supabase=Depends(get_supabase),
) -> Tuple[str, str, str]:
    supabase_user_id = user.id
    email = user.email
    customer = await supabase.table("customers").select("admin_id").eq(
        "user_id", supabase_user_id
    ).maybe_single().execute()
    if not customer or not customer.data:
        raise HTTPException(403, "This is not a user account.")
    return supabase_user_id, customer.data["admin_id"], email


async def resolve_admin_context(
    request: Request,
    authorization: Optional[str] = Header(None, alias="Authorization"),
    supabase=Depends(get_supabase),
    supabase_auth=Depends(get_supabase_auth),
) -> Tuple[str, str]:
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        try:
            user_resp = await supabase_auth.auth.get_user(token)
            if user_resp and user_resp.user:
                user = user_resp.user
                admin = await supabase.table("admins").select("id").eq("user_id", user.id).maybe_single().execute()
                if admin and admin.data:
                    return admin.data["id"], user.email or ""
        except Exception:
            pass

    admin_id = None
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            if isinstance(body, dict):
                admin_id = body.get("admin_id")
        elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            form = await request.form()
            admin_id = form.get("admin_id")
    except Exception:
        pass

    if not admin_id:
        admin_id = request.query_params.get("admin_id")

    if admin_id:
        admin = await supabase.table("admins").select("id").eq("id", admin_id).maybe_single().execute()
        if admin and admin.data:
            return admin.data["id"], ""

    raise HTTPException(status_code=401, detail="Authentication failed: Missing or invalid credentials.")


async def get_admin_config(
    admin_id: str,
    supabase=Depends(get_supabase),
) -> dict:
    admin = await supabase.table("admins").select(
        "database_url, identity_table, pk_column, email_column, jira_base_url, jira_email, jira_token, jira_project_key, jira_issue_type, slack_bot_token, slack_channel_id"
    ).eq("id", admin_id).maybe_single().execute()
    if not admin or not admin.data:
        raise HTTPException(403, "Invalid admin_id.")
    config = admin.data
    if not all(config.get(k) for k in ("identity_table", "pk_column", "email_column")):
        raise HTTPException(400, "Admin has not configured identity table, PK column, or email column.")
    return config


async def get_db_engine(
    admin_id: str,
    supabase=Depends(get_supabase),
):
    admin = await supabase.table("admins").select("database_url").eq("id", admin_id).maybe_single().execute()
    if not admin or not admin.data or not admin.data.get("database_url"):
        raise HTTPException(400, "No database uploaded for this admin.")
    engine = create_engine(admin.data["database_url"])
    try:
        yield engine
    finally:
        engine.dispose()


async def get_customer_pk(
    engine,
    admin_config: dict,
    email: str,
) -> str:
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT {admin_config['pk_column']} FROM {admin_config['identity_table']} WHERE {admin_config['email_column']} = :email"),
            {"email": email},
        ).scalar()
    if not result:
        raise HTTPException(404, "Customer email not found in admin's database.")
    return str(result)


async def get_chat_context(
    customer_ctx=Depends(resolve_customer_context),
    admin_config=Depends(get_admin_config),
    engine=Depends(get_db_engine),
    deps=Depends(get_deps),
):
    from src.agents.context import AgentContext, Deps, JiraConfig, SlackConfig

    supabase_user_id, admin_id, email = customer_ctx
    customer_pk = await get_customer_pk(engine, admin_config, email)

    jira_cfg = JiraConfig(
        base_url=admin_config.get("jira_base_url", ""),
        email=admin_config.get("jira_email", ""),
        api_token=admin_config.get("jira_token", ""),
        project_key=admin_config.get("jira_project_key", ""),
        issue_type=admin_config.get("jira_issue_type", "Task"),
    )
    slack_cfg = SlackConfig(
        bot_token=admin_config.get("slack_bot_token", ""),
        channel_id=admin_config.get("slack_channel_id", ""),
    )

    request_deps = Deps(
        kb_retriever=deps.kb_retriever,
        scheme_retriever=deps.scheme_retriever,
        jev=deps.jev,
        llm=deps.llm,
        sql_llm=deps.sql_llm,
        summarizer_llm=deps.summarizer_llm,
        supabase=deps.supabase,
        http_client=deps.http_client,
        jira_config=jira_cfg,
        slack_config=slack_cfg,
    )
    return AgentContext(
        deps=request_deps,
        admin_id=admin_id,
        user_id=supabase_user_id,
        customer_pk=customer_pk,
        tenant_pk_column=admin_config["pk_column"],
        engine=engine,
    )