from langgraph.prebuilt import create_app
from qa_agentic.graph import build_graph
import yaml

def get_app():
    settings = yaml.safe_load(open("config/settings.yaml"))
    graph, ctx = build_graph(
        "mocks/jira_story.json",
        "mocks/schema_api.json",
        "mocks/tools_registry.json",
        settings
    )
    return create_app(graph)

app = get_app()