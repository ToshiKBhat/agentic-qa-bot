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
import logging, json
logger = logging.getLogger("graph")
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
    return init_chat_model(model, model_provider="ollama", base_url=base_url,reasoning=True)
from langchain_core.runnables import RunnableLambda
from typing import Type

def _clean_and_validate_json(raw: str, schema_model: type) -> dict:
    """
    Cleans and validates raw JSON string against the schema_model.
    Raises ValueError if any required field is missing or any extra field is present.
    """
    import json
    # from pydantic.fields import ModelField
    try:
        data = json.loads(raw)
    except Exception as e:
        raise ValueError(f"Raw output is not valid JSON: {e}\nRaw: {raw}")

    # Get schema fields
    schema_fields = set(schema_model.model_fields.keys())
    required_fields = set(
        k for k, v in schema_model.model_fields.items() if v.is_required
    )
    data_fields = set(data.keys())

    # Check for extra fields
    extra = data_fields - schema_fields
    if extra:
        raise ValueError(f"Extra fields in output: {extra}\nRaw: {raw}")

    # Check for missing required fields
    missing = required_fields - data_fields
    if missing:
        raise ValueError(f"Missing required fields: {missing}\nRaw: {raw}")

    return data

def _structured_call(llm, schema_model: Type[BaseModel], system_text: str, user_payload: dict):
    """
    Force JSON output and parse it into a Pydantic model.
    If the schema is CodegenOutput, also print/log the raw model output BEFORE parsing.
    """
    parser = PydanticOutputParser(pydantic_object=schema_model)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "{system_text}\n\nYou MUST return only valid JSON matching the schema.\n{format_instructions}"),
        ("human", "{payload_json}")
    ])
    prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # Ask Ollama to return pure JSON
    json_llm = llm.bind(format="json")

    # Common pre-parser stage: produce a plain string
    pre_parse = prompt | json_llm | StrOutputParser()

    # Only tap/print for CodegenOutput
    is_codegen = getattr(schema_model, "__name__", "") == "CodegenOutput"

    if is_codegen:
        def _tap(raw: str):
            print("\n=== RAW MODEL OUTPUT (codegen pre-parse) ===\n" + str(raw))
            input("> ").strip()
            logger.info(json.dumps({
                "event": "raw_model_output",
                "stage": "codegen",
                "raw": raw
            }, default=str))
            return raw

        chain = pre_parse | RunnableLambda(_tap)
    else:
        chain = pre_parse

    try:
        raw = chain.invoke({
            "system_text": system_text,
            "payload_json": json.dumps(user_payload, default=str)
        })
        cleaned = _clean_and_validate_json(raw, schema_model)
        return schema_model(**cleaned)
    except Exception as e:
        logger.error(json.dumps({"event": "parse_error", "error": str(e)}, default=str))
        raise

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
    logger.info(json.dumps({"event":"planner_output", "plan": plan}, default=str))
    return plan


# -----------------------------
# Codegen (uses history + errors to repair)
# -----------------------------

def codegen_node(ctx: Context, plan: PlannerOutput, history: List[Dict[str, Any]] | None = None, prev_error: str | None = None) -> Dict[str, Any]:
    llm = build_llm(ctx.settings)
    system_text = (
        "You generate READ-ONLY SQL for DuckDB against parquet-backed views named after datasets. "
        "Only SELECT/WITH . If a prior attempt failed, use prior SQL and error messages to repair."
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
    sql = out.sql
    if isinstance(sql, str):
        sql = [sql]
    elif not isinstance(sql, list):
        sql = [str(sql)]
    return {"sql": sql}

# -----------------------------
# Executor
# -----------------------------
def _normalize_sql(sql_item) -> str:
    """
    Accepts:
      - str: returns stripped string
      - list[str]: joins lines with newlines
      - dict with 'sql' or 'lines' keys: tries to extract and join
    Returns a stripped, single SQL string ready to execute.
    """
    if sql_item is None:
        return ""
    if isinstance(sql_item, str):
        return sql_item.strip()
    if isinstance(sql_item, (list, tuple)):
        # list of lines → single statement
        return "\n".join(map(str, sql_item)).strip()
    if isinstance(sql_item, dict):
        if "sql" in sql_item and isinstance(sql_item["sql"], (list, tuple)):
            return "\n".join(map(str, sql_item["sql"])).strip()
        if "sql" in sql_item and isinstance(sql_item["sql"], str):
            return sql_item["sql"].strip()
        if "lines" in sql_item and isinstance(sql_item["lines"], (list, tuple)):
            return "\n".join(map(str, sql_item["lines"])).strip()
    # last resort
    return str(sql_item).strip()


def _preview_sql(sql: str, max_lines: int = 120) -> str:
    lines = sql.splitlines()
    preview = "\n".join(f"{i+1:>4}: {ln}" for i, ln in enumerate(lines[:max_lines]))
    if len(lines) > max_lines:
        preview += "\n... (truncated)"
    return preview
def executor_node(ctx: Context, code: Dict[str, Any]) -> Dict[str, Any]:
    from .tools import tool_duckdb_sql
    paths = {ds["name"]: ds["path"] for ds in ctx.schema.get("datasets", [])}

    results, ok = [], True
    last_error_type, last_error = None, None

    # Support either a list of full SQL statements or a single one
    raw_sql_items = code.get("sql", [])
    if isinstance(raw_sql_items, (str, dict)):
        raw_sql_items = [raw_sql_items]

    for sql_item in raw_sql_items:
        query = _normalize_sql(sql_item)
        if not query:
            continue

        try:
            df = tool_duckdb_sql(query, paths=paths)
            results.append({
                "sql": query,
                "rows": len(df),
                "sample": df.head(5).to_dict(orient="records")
            })
        except Exception as e:
            ok = False
            last_error = str(e)
            msg = last_error.lower()

            if "syntax error at end of input" in msg or "unterminated" in msg:
                last_error_type = "syntax_incomplete"
            elif "no such file" in msg or "failed to open" in msg:
                last_error_type = "env"
            elif "catalog" in msg or "column" in msg or "schema" in msg:
                last_error_type = "schema"
            elif "connection" in msg or "network" in msg:
                last_error_type = "connection"
            else:
                last_error_type = "unknown"

            # Log a helpful preview to your JSONL / console
            logger.error(json.dumps({
                "event": "sql_error",
                "error": last_error,
                "error_type": last_error_type,
                "sql_preview": _preview_sql(query)
            }, default=str))

            results.append({"sql": query, "error": last_error, "error_type": last_error_type})
            break  # stop on first failing statement

    return {
        "ok": ok,
        "results": results,
    "last_error": last_error,
    "last_error_type": last_error_type
}