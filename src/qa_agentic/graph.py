from __future__ import annotations
import json
from pathlib import Path
from langgraph.graph import StateGraph, END
from .agents import Context, planner_node, codegen_node, executor_node, PlannerOutput
from .hitl import Hitl
from .validator import validate
from .report import write_report

class State(dict):
    pass


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

    def _plan(state: State):
        plan: PlannerOutput = planner_node(ctx)
        state["plan"] = plan
        state["attempts"] = 0
        state["code_history"] = []
        state["last_error"] = None
        state["last_error_type"] = None
        return state

    def _code(state: State):
        prev_error = state.get("last_error")
        state["code"] = codegen_node(ctx, state["plan"], history=state.get("code_history"), prev_error=prev_error)
        return state

    def _exec(state: State):
        out = executor_node(ctx, state["code"])
        state["exec_out"] = out
        state["last_error"] = out.get("last_error")
        state["last_error_type"] = out.get("last_error_type")
        state["exec_ok"] = out.get("ok", False)
        # record history
        rec = {"sql": state["code"].get("sql"), "error": state["last_error"], "error_type": state["last_error_type"]}
        state["code_history"].append(rec)
        return state

    def _validate(state: State):
        paths = {ds["name"]: ds["path"] for ds in ctx.schema.get("datasets", [])}
        plan_checks = [c.model_dump() for c in state["plan"].checks]
        v = validate(plan_checks, state["exec_out"].get("results", []), paths)
        state["validations"] = v
        # overall pass if all passed and exec_ok
        state["overall_pass"] = bool(state.get("exec_ok") and all(x.get("passed") for x in v))
        return state

    def _hitl(state: State):
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
                pass
        elif err_type in {"env", "connection"}:
            q = (
                "Environment/connection issue detected. Provide an updated base path for parquet files, or Enter to skip:"
            )
            base = ctx.hitl.ask(q, default="")
            if base.strip():
                for ds in ctx.schema.get("datasets", []):
                    if not Path(ds["path"]).exists():
                        ds["path"] = str(Path(base) / Path(ds["path"]).name)
        state["attempts"] = int(state.get("attempts", 0)) + 1
        return state

    def _report(state: State):
        # Build markdown
        lines = ["# QA Run Report", ""]
        lines.append(f"Story: {ctx.story.get('id')} – {ctx.story.get('title')}")
        lines.append("")
        lines.append("## Planned Checks:")
        for c in state["plan"].checks:
            thr = c.threshold.value if c.threshold else None
            lines.append(f"- [{c.kind}] **{c.name}** – metric={c.metric}, comparator={c.comparator}, threshold={thr}")
        lines.append("")
        lines.append("## SQL Results:")
        for r in state["exec_out"].get("results", []):
            if "error" in r:
                lines.append("\n### Query (errored)\n")
                lines.append("```sql\n" + r.get("sql", "") + "\n```")
                lines.append(f"Error Type: {r['error_type']}\n\nError: {r['error']}")
            else:
                lines.append("\n### Query\n")
                lines.append("```sql\n" + r["sql"] + "\n```")
                lines.append(f"Rows: {r['rows']}")
                sample = r.get("sample")
                if sample:
                    lines.append("Sample:")
                    import json as _json
                    for row in sample:
                        lines.append("- " + _json.dumps(row))
        lines.append("")
        lines.append("## Validations:")
        for v in state.get("validations", []):
            status = "✅ PASS" if v.get("passed") else "❌ FAIL"
            lines.append(f"- **{v['name']}** [{status}] value={v.get('value')} threshold={v.get('threshold_str')} source={v.get('threshold_source')} evidence={v.get('evidence_summary')}")
        md = "\n".join(lines)
        path = write_report(md, settings["report"]["out_dir"])
        state["report_path"] = str(path)
        return state

    g = StateGraph(State)
    g.add_node("plan", _plan)
    g.add_node("code", _code)
    g.add_node("exec", _exec)
    g.add_node("validate", _validate)
    g.add_node("hitl", _hitl)
    g.add_node("report", _report)

    g.set_entry_point("plan")
    g.add_edge("plan", "code")
    g.add_edge("code", "exec")

    # After exec: either ok→validate or retry/hitl/report
    def _branch_after_exec(state: State):
        if state.get("exec_ok"):
            return "validate"
        attempts = int(state.get("attempts", 0))
        max_retries = int(settings.get("execution", {}).get("retries", 0))
        if attempts < max_retries:
            state["attempts"] = attempts + 1
            return "code"
        if settings.get("execution", {}).get("hitl_after_retries", True):
            return "hitl"
        return "report"

    g.add_conditional_edges("exec", _branch_after_exec, {"code": "code", "hitl": "hitl", "validate": "validate", "report": "report"})

    # After validate → report
    g.add_edge("validate", "report")

    # after hitl, go to code
    g.add_edge("hitl", "code")

    g.add_edge("report", END)

    return g.compile(), ctx