import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver
from supabase import AsyncClient, acreate_client

from src.database_lookup.embedding import SchemeEmbedding
from src.graph import build_agent
from src.knowledge_base.embedding import Embedding
from src.routers.auth import router as auth_router
from src.routers.chat import router as chat_router
from src.routers.upload import router as upload_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Supabase clients (created once, shared by all requests)
    app.state.supabase = await acreate_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )
    app.state.supabase_auth = await acreate_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_ANON_KEY"],
    )

    # Embedding clients for schema / document uploads
    app.state.scheme_embedding = SchemeEmbedding()
    app.state.kb_embedding = Embedding()

    # Compile the support graph once with a single checkpointer
    app.state.agent = build_agent(MemorySaver())

    yield


app = FastAPI(title="Support Agent", lifespan=lifespan)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(upload_router)