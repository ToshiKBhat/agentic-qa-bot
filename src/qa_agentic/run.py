import os
import typer
from dotenv import load_dotenv
import yaml
from rich import print
from .graph import build_graph

app = typer.Typer()

@app.command()
def main(
    story: str = typer.Option(..., help="Path to mocked Jira story JSON"),
    schema: str = typer.Option(..., help="Path to mocked Schema API JSON"),
    tools: str = typer.Option(..., help="Path to tools registry JSON"),
    settings_path: str = typer.Option("config/settings.yaml", help="Settings YAML")
):
    load_dotenv()
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)
    # Expand env
    settings["llm"]["model"] = os.getenv("OLLAMA_MODEL", settings["llm"]["model"])
    settings["llm"]["base_url"] = os.getenv("OLLAMA_BASE_URL", settings["llm"]["base_url"])

    
    graph, ctx = build_graph(story, schema, tools, settings)

    import json, uuid, logging, pathlib, datetime
    from rich import print

    # --- basic file logger ---
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(exist_ok=True, parents=True)
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    log_path = log_dir / f"{run_id}.jsonl"

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # keep raw JSON lines
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()]
    )
    logger = logging.getLogger("graph")

    # Use a thread_id so we can later fetch full state history (requires a checkpointer; see §2)
    config = {"configurable": {"thread_id": run_id}}

    logger.info(json.dumps({
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "event": "run_start",
        "run_id": run_id,
        "settings": settings
    }, default=str))

    # --- stream only node state updates ---
    for chunk in graph.stream({}, stream_mode="updates", config=config):
        # chunk looks like: {"node_name": {"state_key": value, ...}}
        logger.info(json.dumps({
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "event": "node_update",
            "run_id": run_id,
            "chunk": chunk
        }, default=str))

    # final state (also logged)
    final_state = graph.invoke({}, config=config)
    logger.info(json.dumps({
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "event": "run_end",
        "run_id": run_id,
        "final_state_keys": list(final_state.keys()),
        "report_path": final_state.get("report_path")
    }, default=str))

    print(f"[bold green]Report:[/]", final_state.get("report_path"))
    print(f"[bold cyan]Run log:[/]", str(log_path))

    # print(f"[bold green]Report written to[/]: {state.get('report_path')}")
    history = graph.get_state_history(config)  # ordered most-recent first
    from pathlib import Path
    history_path = Path("logs") / f"{run_id}.history.jsonl"
    with open(history_path, "w", encoding="utf-8") as f:
        for snap in reversed(list(history)):  # oldest→newest
            f.write(json.dumps({
                "checkpoint_id": snap.checkpoint_id,
                "parent_checkpoint_id": getattr(snap, "parent_checkpoint_id", None),
                "values": snap.values,  # state snapshot at that point
                "next": snap.next,      # next nodes planned
                "tasks": snap.tasks     # tasks spawned
            }, default=str) + "\n")
    print(f"[bold cyan]State history:[/]", str(history_path))

if __name__ == "__main__":
    app()