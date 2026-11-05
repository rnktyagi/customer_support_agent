import secrets
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])


class Credentials(BaseModel):
    email: str
    password: str


class CustomerSignup(BaseModel):
    email: str
    password: str
    admin_code: str


@router.post("/admin/signup")
async def admin_signup(body: Credentials, request: Request):
    code = secrets.token_hex(4).upper()
    try:
        result = await request.app.state.supabase_auth.auth.sign_up(
            {"email": body.email, "password": body.password}
        )
        await request.app.state.supabase.table("admins").insert(
            {"user_id": result.user.id, "code": code}
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "message": "Sign up successful. Confirm your email if asked, then sign in.",
        "admin_code": code,
    }


@router.post("/admin/signin")
async def admin_signin(body: Credentials, request: Request):
    try:
        result = await request.app.state.supabase_auth.auth.sign_in_with_password(
            {"email": body.email, "password": body.password}
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

    admin = await request.app.state.supabase.table("admins").select("id, code, database_url").eq(
        "user_id", result.user.id
    ).maybe_single().execute()
    if admin is None or admin.data is None:
        raise HTTPException(status_code=403, detail="This is not an admin account.")

    return {
        "admin_id": admin.data["id"],
        "admin_code": admin.data["code"],
        "database_url": admin.data.get("database_url"),
    }


@router.post("/user/signup")
async def user_signup(body: CustomerSignup, request: Request):
    admin = await request.app.state.supabase.table("admins").select("id").eq(
        "code", body.admin_code.strip().upper()
    ).maybe_single().execute()
    if admin is None or admin.data is None:
        raise HTTPException(status_code=400, detail="Invalid admin code.")

    try:
        result = await request.app.state.supabase_auth.auth.sign_up(
            {"email": body.email, "password": body.password}
        )
        await request.app.state.supabase.table("customers").insert(
            {"user_id": result.user.id, "admin_id": admin.data[0]["id"]}
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": "Sign up successful. Confirm your email if asked, then sign in."}


@router.post("/user/signin")
async def user_signin(body: Credentials, request: Request):
    try:
        result = await request.app.state.supabase_auth.auth.sign_in_with_password(
            {"email": body.email, "password": body.password}
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

    customer = await request.app.state.supabase.table("customers").select("admin_id").eq(
        "user_id", result.user.id
    ).maybe_single().execute()
    if customer is None or customer.data is None:
        raise HTTPException(status_code=403, detail="This is not a user account.")

    return {
        "admin_id": customer.data["admin_id"],
        "access_token": result.session.access_token,
    }