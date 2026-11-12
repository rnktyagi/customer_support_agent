import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver
from supabase import AsyncClient, acreate_client
from langchain_openai import ChatOpenAI
from typesafe_sdk import TypeSafeClient

from src.agents.context import Deps
from src.database_lookup.embedding import SchemeEmbedding
from src.database_lookup.retrieval import SchemeRetriever
from src.graph import build_agent
from src.knowledge_base.embedding import Embedding
from src.knowledge_base.retrieval import Retriever
from src.routers.auth import router as auth_router
from src.routers.chat import router as chat_router
from src.routers.upload import router as upload_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.supabase = await acreate_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
    )
    app.state.supabase_auth = await acreate_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_ANON_KEY"),
    )

    app.state.scheme_embedding = SchemeEmbedding()
    app.state.kb_embedding = Embedding()

    jev = TypeSafeClient(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("https://openrouter.ai/api"),
    )
    llm = ChatOpenAI(
        model="openai/gpt-oss-120b",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0,
    )
    sql_llm = ChatOpenAI(
        model="openai/gpt-oss-20b",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0,
    )
    summarizer_llm = ChatOpenAI(
        model="llama-3.1-8b-instant",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0,
    )
    deps = Deps(
        kb_retriever=Retriever(),
        scheme_retriever=SchemeRetriever(),
        jev=jev,
        llm=llm,
        sql_llm=sql_llm,
        summarizer_llm=summarizer_llm,
        supabase=app.state.supabase,
    )
    app.state.deps = deps
    app.state.agent = build_agent(MemorySaver())

    yield


app = FastAPI(title="Support Agent", lifespan=lifespan)

app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(upload_router)