from typing import Tuple
from fastapi import Depends, Header, HTTPException
from sqlalchemy import create_engine, text


async def get_current_user(
    authorization: str = Header(..., alias="Authorization"),
    supabase_auth=Depends(lambda r: r.app.state.supabase_auth),
):
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Bearer token.")
    token = authorization.split(" ", 1)[1].strip()
    user = await supabase_auth.auth.get_user(token)
    if not user or not user.user:
        raise HTTPException(401, "Invalid or expired token.")
    return user.user


async def resolve_customer_context(
    user=Depends(get_current_user),
    supabase=Depends(lambda r: r.app.state.supabase),
) -> Tuple[str, str, str]:
    supabase_user_id = user.id
    email = user.email
    customer = await supabase.table("customers").select("admin_id").eq(
        "user_id", supabase_user_id
    ).maybe_single().execute()
    if not customer or not customer.data:
        raise HTTPException(403, "This is not a user account.")
    return supabase_user_id, customer.data["admin_id"], email


async def get_supabase_user_id(customer_ctx=Depends(resolve_customer_context)) -> str:
    return customer_ctx[0]


async def get_admin_id(customer_ctx=Depends(resolve_customer_context)) -> str:
    return customer_ctx[1]


async def get_user_email(customer_ctx=Depends(resolve_customer_context)) -> str:
    return customer_ctx[2]


async def resolve_admin_context(
    user=Depends(get_current_user),
    supabase=Depends(lambda r: r.app.state.supabase),
) -> Tuple[str, str]:
    admin_id = user.id
    email = user.email
    admin = await supabase.table("admins").select("id").eq("id", admin_id).maybe_single().execute()
    if not admin or not admin.data:
        raise HTTPException(403, "This is not an admin account.")
    return admin_id, email


async def get_admin_config(
    admin_id: str,
    supabase=Depends(lambda r: r.app.state.supabase),
) -> dict:
    admin = await supabase.table("admins").select(
        "database_url, identity_table, pk_column, email_column"
    ).eq("id", admin_id).maybe_single().execute()
    if not admin or not admin.data:
        raise HTTPException(403, "Invalid admin_id.")
    config = admin.data
    if not all(config.get(k) for k in ("identity_table", "pk_column", "email_column")):
        raise HTTPException(400, "Admin has not configured identity table, PK column, or email column.")
    return config


async def get_db_engine(
    admin_id: str,
    supabase=Depends(lambda r: r.app.state.supabase),
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
    engine=Depends(get_db_engine),
    admin_config=Depends(get_admin_config),
    email: str = None,
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
    supabase_user_id=Depends(get_supabase_user_id),
    admin_id=Depends(get_admin_id),
    customer_pk=Depends(get_customer_pk),
    admin_config=Depends(get_admin_config),
    engine=Depends(get_db_engine),
    deps=Depends(lambda r: r.app.state.deps),
):
    from src.agents.context import AgentContext
    return AgentContext(
        deps=deps,
        admin_id=admin_id,
        user_id=supabase_user_id,
        customer_pk=customer_pk,
        tenant_pk_column=admin_config["pk_column"],
        engine=engine,
    )