# Agentic QA Test POC

End-to-end agentic QA tester for **data pipelines** (Parquet/DuckDB):
- **Planner** (LLM) → structured checks (no hardcoding)
- **Codegen** (LLM) → read-only SQL (DuckDB, Parquet views)
- **Executor** → runs SQL, classifies errors
- **HITL gates** → clarify ACs, fix env/schema/connectivity
- **Validator** → compiles metrics from plan and computes PASS/FAIL dynamically
- **Reporter** → Markdown summary with evidence

## Prereqs
- Python 3.10+
- **Ollama** running locally with a code-capable model (e.g., `qwen2.5` or `llama3.1:instruct`).
- ~2–3 GB free disk for datasets.

## Setup
```bash
# 1) Install
pip install -e .

# 2) Pull an LLM
ollama pull qwen2.5

# 3) Download sample data (TLC + zones)
python scripts/fetch_data.py

# 4) (optional) set env
cp .env.example .env
# edit if you changed model or base URL

# 5) Run
python -m qa_agentic.run --story mocks/jira_story.json --schema mocks/schema_api.json --tools mocks/tools_registry.json
```

## What you should see
- Planner may ask HITL clarifications if ACs lack thresholds; otherwise proceeds.
- Codegen produces SQL (joins zones + filters by AC), Executor runs it.
- Validator computes metrics (join_coverage, null_pct, etc.) from the actual result and marks PASS/FAIL.
- Reporter writes `data/outputs/qa_report.md`.

## Notes
- Only **data-pipeline** testing is active; API/UI stubs are out of scope for this POC.
- If files move, HITL prompts let you supply new paths at runtime.