from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
from .hitl import Hitl
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
# -----------------------------
# Structured outputs
# -----------------------------
class Threshold(BaseModel):
    source: str  # ac | hitl | auto | default
    value: Any | None = None  # number | [min,max] | [enum,..] | None
    units: Optional[str] = None  # percent | count | etc.

class TestCheck(BaseModel):
    name: str
    kind: str  # generic | business
    target: str = "sql_result"  # dataset | sql_result | column
    metric: str  # row_count | null_pct | accepted_values | range | join_coverage | custom_sql_bool
    columns: List[str] = Field(default_factory=list)  # for column metrics
    where: Optional[str] = None
    comparator: str = ">="
    threshold: Threshold = Field(default_factory=lambda: Threshold(source="default", value=None))
    evidence: str = "counts"
    on_sql_index: int = 0  # which generated SQL result this check applies to

class PlannerOutput(BaseModel):
    tools_needed: List[str] = Field(default_factory=list)
    checks: List[TestCheck] = Field(default_factory=list)
    clarifications: List[str] = Field(default_factory=list)
    ready: bool = True  # false if HITL is required to proceed

class CodegenOutput(BaseModel):
    sql: List[str] = Field(default_factory=list)

@dataclass
class Context:
    story: dict
    schema: dict
    tools: dict
    settings: dict
    hitl: Hitl


def build_llm(settings: dict):
    model = settings["llm"]["model"]
    base_url = settings["llm"]["base_url"]
    return init_chat_model(model, model_provider="ollama", base_url=base_url)

def _structured_call(llm, schema_model, system_text: str, user_payload: dict):
    """
    Force JSON output and parse it into a Pydantic model, without requiring
    .with_structured_output() (not supported by qwen2.5 in some stacks).
    """
    parser = PydanticOutputParser(pydantic_object=schema_model)

    prompt = ChatPromptTemplate.from_messages([
        # Note: use variables {system_text} and {format_instructions} to avoid brace conflicts
        ("system", "{system_text}\n\nYou MUST return only valid JSON matching the schema.\n{format_instructions}"),
        ("human", "{payload_json}")
    ])

    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # Ask Ollama to return pure JSON (supported for many models)
    json_llm = llm.bind(format="json")

    chain = prompt | json_llm | StrOutputParser() | parser
    return chain.invoke({
        "system_text": system_text,
        "payload_json": json.dumps(user_payload)
    })
# -----------------------------
# Planner (plan only; no code)
# -----------------------------

def planner_node(ctx: Context) -> PlannerOutput:
    llm = build_llm(ctx.settings)

    def _invoke_planner(extra_answers: dict | None = None) -> PlannerOutput:
        system_text = (
            "You are a QA Test Planner. Given a user story and dataset schemas, "
            "produce an execution plan (checks + tools_needed). Checks must be structured: "
            "{name, kind, target, metric, columns, where, comparator, "
            "threshold{source,value,units}, evidence, on_sql_index}. "
            "If acceptance criteria lack critical details (thresholds, dimensions), "
            "set ready=false and list clarifications. Do NOT produce code."
        )
        payload = {
            "story": ctx.story,
            "schema": ctx.schema,
            "available_tools": [t["name"] for t in ctx.tools.get("tools", [])],
        }
        if extra_answers:
            payload["hitl_answers"] = extra_answers

        return _structured_call(llm, PlannerOutput, system_text, payload)

    plan = _invoke_planner()

    if (not plan.ready or plan.clarifications) and ctx.settings.get("hitl", {}).get("max_questions_per_node", 0) > 0:
        answers: dict[str, str] = {}
        for i, q in enumerate(plan.clarifications[: ctx.settings["hitl"]["max_questions_per_node"]]):
            answers[f"q{i+1}"] = ctx.hitl.ask(q)
        plan = _invoke_planner(extra_answers=answers)

    return plan


# -----------------------------
# Codegen (uses history + errors to repair)
# -----------------------------

def codegen_node(ctx: Context, plan: PlannerOutput, history: List[Dict[str, Any]] | None = None, prev_error: str | None = None) -> Dict[str, Any]:
    llm = build_llm(ctx.settings)
    system_text = (
        "You generate READ-ONLY SQL for DuckDB against parquet-backed views named after datasets. "
        "Only SELECT/WITH/EXPLAIN. If a prior attempt failed, use prior SQL and error messages to repair."
    )
    payload = {
        "story": ctx.story,
        "schema": ctx.schema,
        "checks": [c.model_dump() for c in plan.checks],
        "tools_available": [t["name"] for t in ctx.tools.get("tools", [])],
        "prev_error": prev_error,
        "history": history or [],
    }
    out = _structured_call(llm, CodegenOutput, system_text, payload)
    return {"sql": out.sql}

# -----------------------------
# Executor
# -----------------------------

def executor_node(ctx: Context, code: Dict[str, Any]) -> Dict[str, Any]:
    from .tools import tool_duckdb_sql
    paths = {ds["name"]: ds["path"] for ds in ctx.schema.get("datasets", [])}

    results, ok = [], True
    last_error_type, last_error = None, None

    for sql in code.get("sql", []):
        try:
            df = tool_duckdb_sql(sql, paths=paths)
            results.append({"sql": sql, "rows": len(df), "sample": df.head(5).to_dict(orient="records")})
        except Exception as e:  # classify
            ok = False
            last_error = str(e)
            msg = last_error.lower()
            if "no such file" in msg or "failed to open" in msg:
                last_error_type = "env"
            elif "catalog" in msg or "column" in msg or "schema" in msg:
                last_error_type = "schema"
            elif "connection" in msg or "network" in msg:
                last_error_type = "connection"
            else:
                last_error_type = "unknown"
            results.append({"sql": sql, "error": last_error, "error_type": last_error_type})
            break

    return {"ok": ok, "results": results, "last_error": last_error, "last_error_type": last_error_type}