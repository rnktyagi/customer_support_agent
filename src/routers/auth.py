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
def admin_signup(body: Credentials, request: Request):
    code = secrets.token_hex(4).upper()
    try:
        result = request.app.state.supabase_auth.auth.sign_up(
            {"email": body.email, "password": body.password}
        )
        request.app.state.supabase.table("admins").insert(
            {"user_id": result.user.id, "code": code}
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "message": "Sign up successful. Confirm your email if asked, then sign in.",
        "admin_code": code,
    }


@router.post("/admin/signin")
def admin_signin(body: Credentials, request: Request):
    try:
        result = request.app.state.supabase_auth.auth.sign_in_with_password(
            {"email": body.email, "password": body.password}
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

    admin = request.app.state.supabase.table("admins").select("id, code").eq(
        "user_id", result.user.id
    ).execute()
    if not admin.data:
        raise HTTPException(status_code=403, detail="This is not an admin account.")

    return {
        "access_token": result.session.access_token,
        "refresh_token": result.session.refresh_token,
        "admin_code": admin.data[0]["code"],
    }


@router.post("/user/signup")
def user_signup(body: CustomerSignup, request: Request):
    admin = request.app.state.supabase.table("admins").select("id").eq(
        "code", body.admin_code.strip().upper()
    ).execute()
    if not admin.data:
        raise HTTPException(status_code=400, detail="Invalid admin code.")

    try:
        result = request.app.state.supabase_auth.auth.sign_up(
            {"email": body.email, "password": body.password}
        )
        request.app.state.supabase.table("customers").insert(
            {"user_id": result.user.id, "admin_id": admin.data[0]["id"]}
        ).execute()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"message": "Sign up successful. Confirm your email if asked, then sign in."}


@router.post("/user/signin")
def user_signin(body: Credentials, request: Request):
    try:
        result = request.app.state.supabase_auth.auth.sign_in_with_password(
            {"email": body.email, "password": body.password}
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

    customer = request.app.state.supabase.table("customers").select("admin_id").eq(
        "user_id", result.user.id
    ).execute()
    if not customer.data:
        raise HTTPException(status_code=403, detail="This is not a user account.")

    return {
        "admin_id": customer.data[0]["admin_id"]
    }