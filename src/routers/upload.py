import asyncio
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel

from src.dependencies import resolve_admin_context, get_scheme_embedding, get_kb_embedding

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
    jira_base_url: str = ""
    jira_email: str = ""
    jira_token: str = ""
    jira_project_key: str = ""
    jira_issue_type: str = "Task"
    slack_bot_token: str = ""
    slack_channel_id: str = ""


@router.post("/database", response_model=UploadDBResponse)
async def upload_database(
    body: UploadDBRequest,
    admin_id=Depends(lambda: Depends(lambda r: resolve_admin_context(r))),
    scheme_embedding=Depends(lambda r: r.app.state.scheme_embedding),
):
    if body.admin_id != admin_id:
        raise HTTPException(403, "admin_id mismatch")

    supabase = request.app.state.supabase
    await supabase.table("admins").update({
        "database_url": body.database_url,
        "identity_table": body.identity_table,
        "pk_column": body.pk_column,
        "email_column": body.email_column,
        "jira_base_url": body.jira_base_url,
        "jira_email": body.jira_email,
        "jira_token": body.jira_token,
        "jira_project_key": body.jira_project_key,
        "jira_issue_type": body.jira_issue_type,
        "slack_bot_token": body.slack_bot_token,
        "slack_channel_id": body.slack_channel_id,
    }).eq("id", admin_id).execute()

    ids = await asyncio.to_thread(
        scheme_embedding.create_embeddings, body.database_url, admin_id
    )
    return UploadDBResponse(table_ids=ids)


@router.post("/knowledge_base", response_model=UploadKBResponse)
async def upload_knowledge_base(
    admin_id=Depends(lambda: Depends(lambda r: resolve_admin_context(r))),
    files: List[UploadFile] = File(...),
    kb_embedding=Depends(lambda r: r.app.state.kb_embedding),
):
    files = [(f.file, f.filename) for f in files]
    ids = await asyncio.to_thread(kb_embedding.upload_files, files, admin_id)
    return UploadKBResponse(chunk_ids=ids)