from dataclasses import dataclass, field
from typing import Optional

import httpx
from langchain_openai import ChatOpenAI
from supabase import Client
from sqlalchemy.engine import Engine
from typesafe_sdk import TypeSafeClient

from src.database_lookup.retrieval import SchemeRetriever
from src.knowledge_base.retrieval import Retriever


@dataclass
class JiraConfig:
    base_url: str = ""
    email: str = ""
    api_token: str = ""
    project_key: str = ""
    issue_type: str = "Task"


@dataclass
class SlackConfig:
    bot_token: str = ""
    channel_id: str = ""


@dataclass
class Deps:
    kb_retriever: Retriever
    scheme_retriever: SchemeRetriever
    jev: TypeSafeClient
    llm: ChatOpenAI
    sql_llm: ChatOpenAI
    summarizer_llm: ChatOpenAI
    supabase: Client
    jira_config: Optional[JiraConfig] = None
    slack_config: Optional[SlackConfig] = None
    http_client: Optional[httpx.AsyncClient] = None


@dataclass
class AgentContext:
    deps: Deps
    admin_id: str
    user_id: str
    customer_pk: str
    tenant_pk_column: str
    engine: Engine