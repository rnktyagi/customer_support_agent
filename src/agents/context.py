from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from supabase import Client
from sqlalchemy.engine import Engine
from typesafe_sdk import TypeSafeClient

from src.database_lookup.retrieval import SchemeRetriever
from src.knowledge_base.retrieval import Retriever


@dataclass
class Deps:
    kb_retriever: Retriever
    scheme_retriever: SchemeRetriever
    jev: TypeSafeClient
    llm: ChatOpenAI
    sql_llm: ChatOpenAI
    summarizer_llm: ChatOpenAI
    supabase: Client


@dataclass
class AgentContext:
    deps: Deps
    admin_id: str
    user_id: str
    customer_pk: str
    tenant_pk_column: str
    engine: Engine