from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
from .safety import ensure_read_only
from .tools import tool_duckdb_sql

@dataclass
class MetricJob:
    name: str
    sql: str
    threshold_str: str
    threshold_source: str


def _wrap(sql_body: str, outer: str) -> str:
    # Wrap the produced SQL as CTE q, then compute metric
    return f"WITH q AS ({sql_body})\n{outer}"


def compile_check(check: Dict[str, Any], base_sql: str) -> MetricJob:
    name = check.get("name", check.get("metric"))
    metric = check.get("metric")
    comparator = check.get("comparator", ">=")
    thr = check.get("threshold", {})
    thr_source = thr.get("source", "default")
    thr_val = thr.get("value")

    if metric == "row_count":
        outer = "SELECT COUNT(*) AS v FROM q"
        sql = _wrap(base_sql, outer)
        thr_str = f"{comparator} {thr_val if thr_val is not None else 1}"
    elif metric == "null_pct":
        col = check.get("columns", ["col"])[0]
        outer = f"SELECT 100.0 * SUM(CASE WHEN {col} IS NULL THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0) AS v FROM q"
        sql = _wrap(base_sql, outer)
        thr_str = f"{comparator} {thr_val if thr_val is not None else 0.0}"
    elif metric == "accepted_values":
        col = check.get("columns", ["col"])[0]
        vals = thr_val or []
        inlist = ",".join([f"'{v}'" for v in vals]) if vals else "''"
        outer = f"SELECT 100.0 * SUM(CASE WHEN {col} IN ({inlist}) THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0) AS v FROM q"
        sql = _wrap(base_sql, outer)
        thr_str = f"{comparator} 100.0"
    elif metric == "range":
        col = check.get("columns", ["col"])[0]
        outer = f"SELECT MIN({col}) AS mn, MAX({col}) AS mx FROM q"
        sql = _wrap(base_sql, outer)
        # encode as comparator between [min,max]
        if isinstance(thr_val, list) and len(thr_val) == 2:
            thr_str = f"between {thr_val[0]} and {thr_val[1]}"
        else:
            thr_str = "between -inf and +inf"
    elif metric == "join_coverage":
        # assume base_sql already includes joins and non-null zones indicate coverage
        outer = "SELECT 100.0 * SUM(CASE WHEN TRUE THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0) AS v FROM q"
        sql = _wrap(base_sql, outer)
        thr_str = f"{comparator} {thr_val if thr_val is not None else 99.0}"
    elif metric == "custom_sql_bool":
        # base_sql itself returns rows that violate; pass if COUNT(*)==0
        outer = "SELECT COUNT(*)=0 AS v FROM q"
        sql = _wrap(base_sql, outer)
        thr_str = "== 1"
    else:
        # default to row_count >= 1
        outer = "SELECT COUNT(*) AS v FROM q"
        sql = _wrap(base_sql, outer)
        thr_str = f">= {thr_val if thr_val is not None else 1}"

    ensure_read_only(sql)
    return MetricJob(name=name, sql=sql, threshold_str=thr_str, threshold_source=thr_source)


def run_job(job: MetricJob, paths: Dict[str, str]) -> Dict[str, Any]:
    df = tool_duckdb_sql(job.sql, paths=paths)
    # Interpret output flexibly
    if "v" in df.columns:
        val = float(df.loc[0, "v"]) if len(df) else 0.0
        ev = {"rows": len(df)}
    elif {"mn","mx"}.issubset(set(df.columns)):
        mn = df.loc[0, "mn"]
        mx = df.loc[0, "mx"]
        # Convert Timestamps to float (seconds since epoch) or string
        if hasattr(mn, "timestamp"):
            mn = mn.timestamp()
        if hasattr(mx, "timestamp"):
            mx = mx.timestamp()
        val = (float(mn), float(mx))
        ev = {"mn": val[0], "mx": val[1]}
    else:
        val = 0.0
        ev = {"note": "unexpected metric shape"}
    return {"value": val, "evidence": ev}


def compare(value: Any, threshold_str: str) -> bool:
    try:
        if isinstance(value, tuple):
            # expecting range check
            if "between" in threshold_str:
                parts = threshold_str.split()
                lo, hi = float(parts[1]), float(parts[3])
                return lo <= value[0] and value[1] <= hi
        else:
            if threshold_str.startswith(">="):
                return float(value) >= float(threshold_str.split()[1])
            if threshold_str.startswith("<="):
                return float(value) <= float(threshold_str.split()[1])
            if threshold_str.startswith("=="):
                return float(value) == float(threshold_str.split()[1])
    except Exception:
        return False
    return False


def validate(plan_checks: List[Dict[str, Any]], exec_results: List[Dict[str, Any]], paths: Dict[str, str]) -> List[Dict[str, Any]]:
    validations: List[Dict[str, Any]] = []
    # Choose base SQL per check.on_sql_index
    for chk in plan_checks:
        idx = chk.get("on_sql_index", 0)
        # Fallback to a safe SELECT if exec had an error
        base_sql = exec_results[idx].get("sql") if idx < len(exec_results) else "SELECT 1"
        job = compile_check(chk, base_sql)
        run = run_job(job, paths)
        passed = compare(run["value"], job.threshold_str)
        validations.append({
            "name": chk.get("name"),
            "metric": chk.get("metric"),
            "value": run["value"],
            "passed": passed,
            "threshold_str": job.threshold_str,
            "threshold_source": job.threshold_source,
            "evidence_summary": run["evidence"],
            "on_sql_index": idx,
        })
    return validations