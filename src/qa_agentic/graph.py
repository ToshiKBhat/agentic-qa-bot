from __future__ import annotations
import json
from pathlib import Path
from langgraph.graph import StateGraph, END
from .agents import Context, planner_node, codegen_node, executor_node, PlannerOutput
from .hitl import Hitl
from .validator import validate
from .report import write_report
from typing import TypedDict, List, Dict, Any
from typing_extensions import Annotated
import operator
from langgraph.checkpoint.memory import InMemorySaver

class GraphState(TypedDict, total=False):
    plan: Dict[str, Any]                  # or your PlannerOutput as dict
    code: Dict[str, Any]
    exec_out: Dict[str, Any]
    last_error: str
    last_error_type: str
    exec_ok: bool
    attempts: int
    # use reducers to append/extend instead of overwrite
    code_history: Annotated[List[Dict[str, Any]], operator.add]
    validations: Annotated[List[Dict[str, Any]], operator.add]
    overall_pass: bool
    report_path: str


def load_json(p: str | Path):
    return json.loads(Path(p).read_text())


def build_graph(story_path: str, schema_path: str, tools_path: str, settings: dict):
    story = load_json(story_path)
    schema = load_json(schema_path)
    tools = load_json(tools_path)
    hitl = Hitl(
        mode=settings.get("hitl", {}).get("mode", "cli"),
        auto_approve=settings.get("hitl", {}).get("auto_approve", False),
    )
    ctx = Context(story=story, schema=schema, tools=tools, settings=settings, hitl=hitl)

    from .agents import PlannerOutput, planner_node  # ensure imports

    def _plan(state: GraphState):
        plan_obj: PlannerOutput = planner_node(ctx)

        # Store plain dicts in state to avoid serialization/merge issues
        plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else dict(plan_obj)

        # Return *only updates*; initialize lists so reducers can append later
        return {
            "plan": plan_dict,
            "attempts": 0,
            "code_history": [],
            "last_error": None,
            "last_error_type": None,
        }
    from .agents import CodegenOutput, codegen_node, PlannerOutput  # ensure imports

    def _code(state: GraphState):
        # Ensure we have a plan; if not, re-plan once defensively
        plan_dict = state.get("plan")
        if plan_dict is None:
            plan_obj = planner_node(ctx)
            plan_dict = plan_obj.model_dump() if hasattr(plan_obj, "model_dump") else dict(plan_obj)

        # Rehydrate Pydantic (codegen expects a PlannerOutput)
        try:
            plan_obj = PlannerOutput(**plan_dict) if isinstance(plan_dict, dict) else plan_dict
        except Exception as e:
            raise RuntimeError(f"Invalid plan in state; cannot rehydrate PlannerOutput: {e}")

        prev_error = state.get("last_error")
        history = state.get("code_history", [])
        code_out: Dict[str, Any] = codegen_node(ctx, plan_obj, history=history, prev_error=prev_error)

        # Normalize to dict (for consistency if you later switch codegen to Pydantic)
        code_dict = code_out if isinstance(code_out, dict) else {"sql": getattr(code_out, "sql", [])}

        return {
            "plan": plan_dict,   # keep latest (in case we re-planned)
            "code": code_dict,
        }



    def _exec(state: GraphState):
        out = executor_node(ctx, state["code"])
        # capture the exact SQL we executed (or tried to)
        executed_sql = []
        for r in out.get("results", []):
            # each r has either "sql" (the normalized string) or "error"
            if "sql" in r:
                executed_sql.append(r["sql"])

        rec = {
            "sql": executed_sql,
            "error": out.get("last_error"),
            "error_type": out.get("last_error_type"),
        }

        return {
            "exec_out": out,
            "last_error": out.get("last_error"),
            "last_error_type": out.get("last_error_type"),
            "exec_ok": out.get("ok", False),
            "code_history": [rec],  # reducer appends
        }


    def _validate(state: GraphState):
        # dataset paths for DuckDB views
        paths = {ds["name"]: ds["path"] for ds in ctx.schema.get("datasets", [])}

        # plan may be a dict (stored) — rehydrate as needed
        plan_dict = state.get("plan") or {}
        plan_checks = plan_dict.get("checks", [])

        results = state.get("exec_out", {}).get("results", [])
        v = validate(plan_checks, results, paths) if plan_checks else []

        overall = bool(state.get("exec_ok") and (all(x.get("passed") for x in v) if v else True))

        return {
            # reducer will add these records (here we set the full set in one go; if you run validate multiple times, they’ll append)
            "validations": v,
            "overall_pass": overall,
        }


    def _hitl(state: GraphState):
        err_type = state.get("last_error_type")

        if err_type == "schema":
            q = (
                "Schema error encountered. If a column or dataset name is wrong, "
                "please provide the correct mapping as JSON (e.g., {\"trips\": \"path-or-table\"}).\n"
                "Or press Enter to keep current settings."
            )
            ans = ctx.hitl.ask(q, default="")
            try:
                if ans.strip():
                    mapping = json.loads(ans)
                    for ds in ctx.schema.get("datasets", []):
                        if ds["name"] in mapping:
                            ds["path"] = mapping[ds["name"]]
            except Exception:
                # swallow malformed JSON; user can retry
                pass

        elif err_type in {"env", "connection"}:
            q = "Environment/connection issue detected. Provide an updated base path for parquet files, or Enter to skip:"
            base = ctx.hitl.ask(q, default="")
            if base.strip():
                for ds in ctx.schema.get("datasets", []):
                    if not Path(ds["path"]).exists():
                        ds["path"] = str(Path(base) / Path(ds["path"]).name)

        return {
            "attempts": int(state.get("attempts", 0)) + 1
        }


    def _report(state: GraphState):
        lines = ["# QA Run Report", ""]
        lines.append(f"Story: {ctx.story.get('id')} – {ctx.story.get('title')}")
        lines.append("")

        # Planned Checks
        lines.append("## Planned Checks:")
        plan_dict = state.get("plan") or {}
        for c in plan_dict.get("checks", []):
            thr_val = (c.get("threshold") or {}).get("value")
            lines.append(f"- [{c.get('kind')}] **{c.get('name')}** – metric={c.get('metric')}, comparator={c.get('comparator')}, threshold={thr_val}")

        # SQL Results
        lines.append("")
        lines.append("## SQL Results:")
        for r in state.get("exec_out", {}).get("results", []):
            if "error" in r:
                lines.append("\n### Query (errored)\n")
                lines.append("```sql\n" + (r.get("sql") or "") + "\n```")
                lines.append(f"Error Type: {r.get('error_type')}\n\nError: {r.get('error')}")
            else:
                lines.append("\n### Query\n")
                lines.append("```sql\n" + (r.get("sql") or "") + "\n```")
                lines.append(f"Rows: {r.get('rows')}")
                sample = r.get("sample")
                if sample:
                    lines.append("Sample:")
                    import json as _json
                    for row in sample:
                        lines.append("- " + _json.dumps(row, default=str))

        # Validations
        lines.append("")
        lines.append("## Validations:")
        for v in state.get("validations", []):
            status = "✅ PASS" if v.get("passed") else "❌ FAIL"
            lines.append(
                f"- **{v.get('name')}** [{status}] value={v.get('value')} "
                f"threshold={v.get('threshold_str')} source={v.get('threshold_source')} "
                f"evidence={v.get('evidence_summary')}"
            )

        md = "\n".join(lines)
        path = write_report(md, settings["report"]["out_dir"])
        return {"report_path": str(path)}


    g = StateGraph(GraphState)
    g.add_node("plan", _plan)
    g.add_node("code", _code)
    g.add_node("exec", _exec)
    g.add_node("validate", _validate)
    g.add_node("hitl", _hitl)
    g.add_node("report", _report)

    g.set_entry_point("plan")
    g.add_edge("plan", "code")
    g.add_edge("code", "exec")

    from rich import print

    # --- New retry node ---
    def _retry(state: GraphState):
        attempts = int(state.get("attempts", 0)) + 1
        max_retries = int(settings.get("execution", {}).get("retries", 0))
        print(f"[yellow]Retry attempt {attempts}/{max_retries}[/]")
        return {"attempts": attempts}

    # Register the node
    g.add_node("retry", _retry)

    # --- Pure branch function (no state mutation) ---
    def _branch_after_exec(state: GraphState):
        if state.get("exec_ok"):
            return "validate"

        attempts = int(state.get("attempts", 0))
        max_retries = int(settings.get("execution", {}).get("retries", 0))

        if attempts < max_retries:
            return "retry"

        if settings.get("execution", {}).get("hitl_after_retries", True):
            return "hitl"

        return "report"

    # --- Conditional routing from exec ---
    g.add_conditional_edges(
        "exec",
        _branch_after_exec,
        {
            "validate": "validate",
            "retry": "retry",
            "hitl": "hitl",
            "report": "report",
        },
    )

    # --- After retry, always go back to codegen ---
    g.add_edge("retry", "code")
 # After validate → report
    g.add_edge("validate", "report")

    # after hitl, go to code
    g.add_edge("hitl", "code")

    g.add_edge("report", END)

    checkpointer = InMemorySaver()
    compiled = g.compile(checkpointer=checkpointer)
    return compiled, ctx