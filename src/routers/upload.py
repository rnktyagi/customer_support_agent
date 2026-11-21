import asyncio
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, Body
from pydantic import BaseModel

from src.dependencies import (
    resolve_admin_context,
    get_scheme_embedding,
    get_kb_embedding,
    get_supabase,
)

router = APIRouter(prefix="/upload", tags=["upload"])


class UploadDBResponse(BaseModel):
    table_ids: List[str]


class UploadKBResponse(BaseModel):
    chunk_ids: List[str]


class UploadDBRequest(BaseModel):
    admin_id: str
    database_url: str
    identity_table: str
    pk_column: str
    email_column: str


@router.post("/database", response_model=UploadDBResponse)
async def upload_database(
    request: Request,
    body: UploadDBRequest,
    admin_context=Depends(resolve_admin_context),
    scheme_embedding=Depends(get_scheme_embedding),
    supabase=Depends(get_supabase),
):
    admin_id, _ = admin_context

    await supabase.table("admins").update({
        "database_url": body.database_url,
        "identity_table": body.identity_table,
        "pk_column": body.pk_column,
        "email_column": body.email_column,
    }).eq("id", admin_id).execute()

    ids = await asyncio.to_thread(
        scheme_embedding.create_embeddings, body.database_url, admin_id
    )
    return UploadDBResponse(table_ids=ids)


class IntegrationConfigRequest(BaseModel):
    admin_id: Optional[str] = None
    jira_base_url: Optional[str] = None
    jira_email: Optional[str] = None
    jira_token: Optional[str] = None
    jira_project_key: Optional[str] = None
    jira_issue_type: Optional[str] = None
    slack_bot_token: Optional[str] = None
    slack_channel_id: Optional[str] = None

    class Config:
        extra = "ignore"


@router.post("/integrations", response_model=dict)
async def update_integrations(
    request: Request,
    body: IntegrationConfigRequest = Body(default=None),
    admin_context=Depends(resolve_admin_context),
    supabase=Depends(get_supabase),
):
    admin_id, _ = admin_context

    data = body.model_dump(exclude_unset=True) if body else {}
    data.pop("admin_id", None)
    update_data = {k: v for k, v in data.items() if v is not None}

    if update_data:
        await supabase.table("admins").update(update_data).eq("id", admin_id).execute()

    return {"message": "Integration settings updated successfully"}


@router.post("/knowledge_base", response_model=UploadKBResponse)
async def upload_knowledge_base(
    admin_context=Depends(resolve_admin_context),
    files: List[UploadFile] = File(...),
    kb_embedding=Depends(get_kb_embedding),
):
    admin_id, _ = admin_context
    files = [(f.file, f.filename) for f in files]
    ids = await asyncio.to_thread(kb_embedding.upload_files, files, admin_id)
    return UploadKBResponse(chunk_ids=ids)