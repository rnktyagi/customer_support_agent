import os
import re
import asyncio
from dataclasses import dataclass
from datetime import date
from typing import TypedDict, Optional, Literal, List
from dotenv import load_dotenv

import tiktoken

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime

from sqlalchemy import text
from sqlalchemy.engine import Engine

from typesafe_sdk import TypeSafeClient, Choice

from src.database_lookup.retrieval import SchemeRetriever

load_dotenv()

sql_llm = ChatOpenAI(
    model="openai/gpt-oss-20b",
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1",
    temperature=0,
)

scheme_retriever: Optional[SchemeRetriever] = None


def get_scheme_retriever() -> SchemeRetriever:
    global scheme_retriever
    if scheme_retriever is None:
        scheme_retriever = SchemeRetriever()
    return scheme_retriever


jev_client: Optional[TypeSafeClient] = None


def get_jev_client() -> TypeSafeClient:
    global jev_client
    if jev_client is None:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("TYPESAFE_API_KEY", "")
        base_url = os.getenv("TYPESAFE_BASE_URL", "https://openrouter.ai/api")
        jev_client = TypeSafeClient(api_key=api_key, base_url=base_url)
    return jev_client


encoder = tiktoken.get_encoding("o200k_base")

TABLE_TOKEN_LIMIT = 8000
MAX_ROWS = 50


@dataclass
class AgentContext:
    admin_id: str
    user_id: str
    engine: Engine


class DBLookupState(TypedDict):
    messages: List[AnyMessage]
    table_store: List[dict]
    new_tables: List[dict]
    current_query: str
    rewritten_query: str
    decision: Optional[Literal["retrieve", "reuse"]]
    retrieval_status: Optional[Literal["relevant", "irrelevant", "no_context"]]
    retry_count: int
    schema: Optional[str]
    sql_query: Optional[str]
    sql_error: Optional[str]
    sql_retry_count: int
    db_rows: List[dict]
    db_status: Optional[Literal["ok", "failed"]]


def preprocess_node(state: DBLookupState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        cleaned_query = ""
    else:
        latest_msg = messages[-1]
        cleaned_query = latest_msg.content if hasattr(latest_msg, "content") else str(latest_msg)

    return {
        "current_query": cleaned_query,
        "rewritten_query": cleaned_query,
        "retry_count": 0,
        "sql_retry_count": 0,
        "decision": None,
        "new_tables": [],
        "schema": None,
        "sql_query": None,
        "sql_error": None,
        "db_rows": [],
        "db_status": None,
    }


def context_sufficiency_gate_node(state: DBLookupState) -> dict:
    messages = state.get("messages", [])
    current_query = state.get("current_query", "").strip()
    store = state.get("table_store") or []

    if not store:
        return {"decision": "retrieve"}

    history = "\n".join(f"{m.type}: {m.content}" for m in messages[:-1])

    stored_tables = "".join(
        f"\n--- Table {n} ---\n"
        f"Table name: {t['table_name']}\n"
        f"{t['table_info']}\n"
        for n, t in enumerate(reversed(store), start=1)
    )

    prompt_state = f"""<role>
You are an enterprise Text-to-SQL context sufficiency evaluator. You must decide whether to RETRIEVE fresh database tables or REUSE the already retrieved tables to write the SQL query for the current question.
</role>

<critical_rule id="table_sufficiency_gate">
Default bias is RETRIEVE — you may only choose REUSE if ALL 4 criteria are strictly met:
1. COVERAGE: every entity, metric and column the current question needs exists in the retrieved tables (including the tables needed to join them) — not just topically related.
2. NO NEW ENTITIES: the question introduces no new entity, metric, date field or sub-topic that is not present in the retrieved tables.
3. NOT EXPANSIVE: the question doesn't ask for something broader, deeper, or differently-scoped than what the retrieved tables can answer.
4. NO ASSUMPTIONS: you can point to the exact tables and columns for every part of the question. Do not assume.

If ANY of the four fails, or you are uncertain, you MUST RETRIEVE.
</critical_rule>

<examples>
<example type="correct_reuse">
Turn 1: "Give me all the users and their expenses" -> tables users and expenses were retrieved.
Turn 2: "Give me the user with the highest expenses." -> the same tables are enough to write the SQL. Decision: REUSE.
</example>
<example type="incorrect_reuse">
Turn 1: "Give me all the products listed for more than 100" -> products table retrieved.
Turn 2: "Give me the employee with the lowest salary." -> completely new and unrelated to the retrieved tables. Decision: RETRIEVE.
</example>
<example type="correct_retrieve_despite_relevancy">
Turn 1: "Give me all the products listed for more than 100" -> products table retrieved.
Turn 2: "Fetch me the top 5 products which are sold the most." -> related to products, but "sold the most" may need sales data that is not in the retrieved tables. Do not assume. Decision: RETRIEVE.
</example>
</examples>

<messages>
{history}
</messages>

<retrieved_tables>
{stored_tables}
</retrieved_tables>

<current_query>
{current_query}
</current_query>"""

    try:
        client = get_jev_client()
        response = client.system_one(
            model="jev-1.13",
            state=prompt_state,
            questions={
                "decision": Choice(
                    instructions="Based strictly on the 4 sufficiency rules, should the assistant RETRIEVE new tables or REUSE the already retrieved tables?",
                    criteria={
                        "retrieve": "Fails one or more sufficiency tests: the question needs new entities, metrics or columns, is broader in scope, or requires tables not visible in the retrieved tables.",
                        "reuse": "Passes all 4 sufficiency tests: every part of the question can be answered with the exact tables and columns already retrieved."
                    }
                )
            }
        )
        decision = response.answers["decision"].choice
        if decision not in ("retrieve", "reuse"):
            decision = "retrieve"
    except Exception as e:
        print(f"JEV sufficiency decision error: {e}. Defaulting to 'retrieve'")
        decision = "retrieve"

    return {"decision": decision}


def query_rewriter_node(state: DBLookupState) -> dict:
    current_query = state.get("current_query", "")
    messages = state.get("messages", [])

    history = "\n".join(f"{m.type}: {m.content}" for m in messages[:-1])

    system_instruction = """You rewrite the user's latest message into a query used to retrieve the relevant database tables from a vector database.

Rules:
1. The query must be a fully self-contained search string. Resolve pronouns and vague references ("it", "that", "this one", "the previous one") against the conversation history.
   Example: prior topic "employees with their salaries" + latest message "give me the highest among them" -> "employee with the highest salary", NOT "highest among them".
2. Do NOT answer the question and do NOT write SQL.
3. Keep it concise and keep the key nouns (entities, metrics, time ranges), since it is matched against table and column names.
4. Output ONLY the query with no explanation."""

    try:
        response = sql_llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=(
                f"Conversation so far:\n{history}\n\n"
                f"Latest user message:\n{current_query}\n\n"
                "Rewrite the latest message as a standalone query."
            )),
        ])
        rewritten = response.content.strip().strip('"').strip("'")
        if not rewritten:
            rewritten = current_query
    except Exception as e:
        print(f"Query rewrite error: {e}")
        rewritten = current_query

    return {"rewritten_query": rewritten}


async def retrieval_node(state: DBLookupState, runtime: Runtime[AgentContext]) -> dict:
    query = state.get("rewritten_query") or state.get("current_query", "")
    admin_id = runtime.context.admin_id

    try:
        results = await asyncio.to_thread(get_scheme_retriever().retrieve, query, admin_id)
    except Exception as e:
        print(f"Schema retrieval error: {e}")
        return {"new_tables": []}

    new_tables = [
        {
            "table_id": doc.metadata.get("table_id") or doc.metadata["table_name"],
            "table_name": doc.metadata["table_name"],
            "dialect": doc.metadata.get("dialect", ""),
            "table_info": doc.metadata["table_info"],
            "tokens": len(encoder.encode(doc.metadata["table_info"])),
        }
        for doc in results
    ]
    return {"new_tables": new_tables}


def retrieval_evaluation_node(state: DBLookupState) -> dict:
    new_tables = state.get("new_tables") or []
    current_query = state.get("current_query", "")
    rewritten_query = state.get("rewritten_query", current_query)

    if not new_tables:
        return {"retrieval_status": "no_context"}

    new_tables_text = "".join(
        f"\n--- Table {n} ---\n"
        f"Table name: {t['table_name']}\n"
        f"{t['table_info']}\n"
        for n, t in enumerate(new_tables, start=1)
    )

    prompt_state = f"""User Query:
{current_query}

Search Query Used:
{rewritten_query}

Retrieved Tables:
{new_tables_text}"""

    try:
        client = get_jev_client()
        response = client.system_one(
            model="jev-1.13",
            state=prompt_state,
            questions={
                "relevance": Choice(
                    instructions="Do the retrieved tables contain the tables and columns needed to write a SQL query that answers the user query?",
                    criteria={
                        "relevant": "The tables contain the entities, columns and relationships needed to answer the user query.",
                        "irrelevant": "The tables are unrelated or are missing what is needed to answer the user query."
                    }
                )
            }
        )
        relevance_choice = response.answers["relevance"].choice
        status = "relevant" if relevance_choice == "relevant" else "irrelevant"
    except Exception as e:
        print(f"JEV retrieval evaluation error: {e}. Defaulting to 'relevant'")
        status = "relevant"

    return {"retrieval_status": status}


def broaden_query_node(state: DBLookupState) -> dict:
    current_query = state.get("current_query", "")
    previous_query = state.get("rewritten_query", current_query)
    retry_count = state.get("retry_count", 0)
    sql_error = state.get("sql_error")

    prompt = f"""The previous search for database tables did not lead to a working SQL query.
Formulate a differently-worded search query that uses synonyms, related entities or higher-level keywords to find the relevant tables and columns.

Original User Question: {current_query}
Previous Search: {previous_query}
"""
    if sql_error:
        prompt += f"SQL error from the last attempt: {sql_error}\n"

    prompt += "\nProvide ONLY the reformulated search query:"

    try:
        response = sql_llm.invoke(prompt)
        broader_query = response.content.strip().strip('"').strip("'")
        if not broader_query:
            broader_query = f"{current_query} details"
    except Exception as e:
        print(f"Broaden query error: {e}")
        broader_query = f"{current_query} details"

    return {
        "rewritten_query": broader_query,
        "retry_count": retry_count + 1,
    }


def fallback_node(state: DBLookupState) -> dict:
    return {"db_rows": [], "db_status": "failed"}


def add_tables_node(state: DBLookupState) -> dict:
    store = state.get("table_store") or []
    new_tables = state.get("new_tables") or []

    new_ids = {t["table_id"] for t in new_tables}
    store = [t for t in store if t["table_id"] not in new_ids]

    return {"table_store": store + new_tables}


def trim_oldest_node(state: DBLookupState) -> dict:
    store = state.get("table_store") or []
    return {"table_store": store[1:]}


def build_schema_node(state: DBLookupState) -> dict:
    store = state.get("table_store") or []

    dialect = store[0]["dialect"] if store else ""
    tables = "\n".join(t["table_info"] for t in store)

    return {"schema": f"Dialect : {dialect}\n{tables}\n"}


def sql_generator_node(state: DBLookupState) -> dict:
    schema = state.get("schema", "")
    question = state.get("current_query", "")
    standalone_question = state.get("rewritten_query", "")
    previous_sql = state.get("sql_query")
    error = state.get("sql_error")
    sql_retry_count = state.get("sql_retry_count", 0)

    system_instruction = f"""You are an enterprise Text-to-SQL generator. Write ONE read-only SQL query that answers the user's question from the given database schema.

Today's date is {date.today().isoformat()}. Use it to resolve relative time expressions ("last month", "this quarter", "yesterday", "year to date") into concrete date ranges in the WHERE clause.

Rules:
1. The query must only GET data: a single SELECT statement (a WITH clause is allowed). Never write anything that deletes or manipulates data or the schema.
2. Use ONLY the tables and columns present in the given schema, with their exact names. Never hallucinate a column, table or join.
3. Write the query in the exact SQL dialect stated in the schema (the "Dialect : ..." line). Do not default to another dialect's syntax.
4. The current user's id is available as the bind parameter :user_id. Every query MUST restrict the results to this user with :user_id, either on the user's column directly or through joins. Never write a literal user id and never return other users' data.
5. For every selected column, use a clear alias (e.g. SUM(amount) AS total_amount) so the result can be understood without guessing from column position.
6. When the question is about a person, customer, company, product or other entity, return BOTH its readable name field (if one exists) and its matching ID field, joining to another table to fetch the name if needed. Priority: (1) name and id together, (2) name only if the id cannot be included, (3) id only if no readable name exists anywhere.
7. When listing records, also include the other relevant identifying columns available in the schema (for example id, name and designation for an employee).
8. For aggregate queries, include a label column where possible so the result is self-explanatory without the original question.
9. If the question is ambiguous, use the most natural interpretation.
10. Limit results to at most {MAX_ROWS} rows unless the query is an aggregate.
11. Output ONLY the SQL, with no explanation and no markdown."""

    user_prompt = (
        f"Database schema:\n{schema}\n\n"
        f"User question:\n{question}\n\n"
        f"Standalone version of the question:\n{standalone_question}"
    )

    if error:
        user_prompt += (
            f"\n\nYour previous query:\n{previous_sql}\n\n"
            f"It was rejected or failed with this error:\n{error}\n\n"
            "Write a corrected query."
        )

    try:
        response = sql_llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_prompt),
        ])
        sql = response.content.strip()
        sql = sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
    except Exception as e:
        print(f"SQL generation error: {e}")
        sql = ""

    return {
        "sql_query": sql,
        "sql_error": None,
        "sql_retry_count": sql_retry_count + 1 if error else sql_retry_count,
    }


def sql_guard_node(state: DBLookupState) -> dict:
    sql = (state.get("sql_query") or "").strip().rstrip(";").strip()
    lowered = sql.lower()
    question = state.get("current_query", "")
    schema = state.get("schema", "")

    if not lowered.startswith(("select", "with")):
        return {"sql_error": "The query must be a single SELECT statement."}
    if ";" in sql:
        return {"sql_error": "Only one statement is allowed."}
    if re.search(r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|merge)\b", lowered):
        return {"sql_error": "The query contains a write or schema-changing keyword."}
    if ":user_id" not in sql:
        return {"sql_error": "The query must restrict the data to the current user with the :user_id parameter."}

    prompt_state = f"""User Question:
{question}

Database Schema:
{schema}

Generated SQL:
{sql}"""

    try:
        client = get_jev_client()
        response = client.system_one(
            model="jev-1.13",
            state=prompt_state,
            questions={
                "safety": Choice(
                    instructions="Is the generated SQL safe to run for the current user? It must only read data, only touch data belonging to the current user via :user_id, and must not read system tables or other users' data.",
                    criteria={
                        "safe": "Read-only query that is properly restricted to the current user with :user_id on the right column, directly or through joins.",
                        "unsafe": "Writes or changes data, reads system or schema tables, is not restricted to the current user, or could expose other users' data."
                    }
                )
            }
        )
        verdict = response.answers["safety"].choice
    except Exception as e:
        print(f"JEV SQL guard error: {e}. Treating the query as unsafe")
        verdict = "unsafe"

    if verdict != "safe":
        return {"sql_error": "The query was rejected by the safety check. Make sure it is read-only and restricted to the current user with :user_id on the correct column."}

    return {"sql_query": sql, "sql_error": None}


def sql_execution_node(state: DBLookupState, runtime: Runtime[AgentContext]) -> dict:
    sql = state.get("sql_query", "")
    context = runtime.context

    try:
        with context.engine.connect() as conn:
            result = conn.execute(text(sql), {"user_id": context.user_id})
            rows = result.mappings().fetchmany(MAX_ROWS)

        db_rows = [
            {
                k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                for k, v in row.items()
            }
            for row in rows
        ]
        return {"db_rows": db_rows, "db_status": "ok", "sql_error": None}
    except Exception as e:
        return {"sql_error": str(e)[:500]}


def route_after_sufficiency(state: DBLookupState) -> Literal["query_rewriter", "build_schema"]:
    if state.get("decision") == "reuse":
        return "build_schema"
    return "query_rewriter"


def route_after_retrieval_eval(state: DBLookupState) -> Literal["add_tables", "broaden_query", "fallback"]:
    status = state.get("retrieval_status", "no_context")
    retry_count = state.get("retry_count", 0)

    if status == "relevant":
        return "add_tables"

    if retry_count < 1:
        return "broaden_query"
    return "fallback"


def route_after_add(state: DBLookupState) -> Literal["trim_oldest", "build_schema"]:
    store = state.get("table_store") or []
    new_count = len(state.get("new_tables") or [])
    total_tokens = sum(t["tokens"] for t in store)

    if total_tokens > TABLE_TOKEN_LIMIT and len(store) > new_count:
        return "trim_oldest"
    return "build_schema"


def route_after_guard(state: DBLookupState) -> Literal["sql_execution", "broaden_query", "fallback"]:
    if not state.get("sql_error"):
        return "sql_execution"
    if state.get("sql_retry_count", 0) < 1:
        return "broaden_query"
    return "fallback"


def route_after_execution(state: DBLookupState) -> Literal["broaden_query", "fallback", "__end__"]:
    if not state.get("sql_error"):
        return END
    if state.get("sql_retry_count", 0) < 1:
        return "broaden_query"
    return "fallback"


def create_db_lookup_graph() -> StateGraph:
    workflow = StateGraph(DBLookupState, context_schema=AgentContext)

    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("context_sufficiency_gate", context_sufficiency_gate_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("retrieval_eval", retrieval_evaluation_node)
    workflow.add_node("broaden_query", broaden_query_node)
    workflow.add_node("fallback", fallback_node)
    workflow.add_node("add_tables", add_tables_node)
    workflow.add_node("trim_oldest", trim_oldest_node)
    workflow.add_node("build_schema", build_schema_node)
    workflow.add_node("sql_generator", sql_generator_node)
    workflow.add_node("sql_guard", sql_guard_node)
    workflow.add_node("sql_execution", sql_execution_node)

    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "context_sufficiency_gate")

    workflow.add_conditional_edges(
        "context_sufficiency_gate",
        route_after_sufficiency,
        {
            "query_rewriter": "query_rewriter",
            "build_schema": "build_schema",
        }
    )

    workflow.add_edge("query_rewriter", "retrieval")
    workflow.add_edge("retrieval", "retrieval_eval")

    workflow.add_conditional_edges(
        "retrieval_eval",
        route_after_retrieval_eval,
        {
            "add_tables": "add_tables",
            "broaden_query": "broaden_query",
            "fallback": "fallback",
        }
    )

    workflow.add_edge("broaden_query", "retrieval")
    workflow.add_edge("fallback", END)

    workflow.add_conditional_edges(
        "add_tables",
        route_after_add,
        {
            "trim_oldest": "trim_oldest",
            "build_schema": "build_schema",
        }
    )
    workflow.add_conditional_edges(
        "trim_oldest",
        route_after_add,
        {
            "trim_oldest": "trim_oldest",
            "build_schema": "build_schema",
        }
    )

    workflow.add_edge("build_schema", "sql_generator")
    workflow.add_edge("sql_generator", "sql_guard")

    workflow.add_conditional_edges(
        "sql_guard",
        route_after_guard,
        {
            "sql_execution": "sql_execution",
            "broaden_query": "broaden_query",
            "fallback": "fallback",
        }
    )
    workflow.add_conditional_edges(
        "sql_execution",
        route_after_execution,
        {
            "broaden_query": "broaden_query",
            "fallback": "fallback",
            END: END,
        }
    )

    return workflow


def build_agent():
    workflow = create_db_lookup_graph()
    return workflow.compile(checkpointer=True)


build_workflow = build_agent