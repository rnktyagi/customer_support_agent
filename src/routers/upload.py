import asyncio
from typing import List

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel

router = APIRouter(prefix="/upload", tags=["upload"])


class UploadDBResponse(BaseModel):
    table_ids: List[str]


class UploadKBResponse(BaseModel):
    chunk_ids: List[str]


@router.post("/database", response_model=UploadDBResponse)
async def upload_database(
    request: Request,
    admin_id: str = Form(...),
    database_url: str = Form(...),
):
    supabase = request.app.state.supabase

    admin = await supabase.table("admins").select("id").eq("id", admin_id).maybe_single().execute()
    if admin is None or admin.data is None:
        raise HTTPException(status_code=403, detail="Invalid admin_id.")

    await supabase.table("admins").update({"database_url": database_url}).eq(
        "id", admin_id
    ).execute()

    try:
        ids = await asyncio.to_thread(
            request.app.state.scheme_embedding.create_embeddings, database_url, admin_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return UploadDBResponse(table_ids=ids)


@router.post("/knowledge_base", response_model=UploadKBResponse)
async def upload_knowledge_base(
    request: Request,
    admin_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    supabase = request.app.state.supabase

    admin = await supabase.table("admins").select("id").eq("id", admin_id).maybe_single().execute()
    if admin is None or admin.data is None:
        raise HTTPException(status_code=403, detail="Invalid admin_id.")

    # Streamlit sends raw bytes via multipart; FastAPI wraps them as UploadFile.
    # We extract the bytes back for the ingestion layer.
    file_bytes = [(f.file.read(), f.filename) for f in files]

    try:
        ids = await asyncio.to_thread(
            request.app.state.kb_embedding.upload_files, file_bytes, admin_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return UploadKBResponse(chunk_ids=ids)