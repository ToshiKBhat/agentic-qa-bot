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

    # Run with verbose=True to see step-by-step
    state = graph.invoke({}, verbose=True)


    print(f"[bold green]Report written to[/]: {state.get('report_path')}")

if __name__ == "__main__":
    app()