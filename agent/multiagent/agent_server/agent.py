"""
Multi-agent orchestrator with Genie subagent for Enterprise Sales Analytics.

Configuration is loaded from ``config.yaml`` at the project root.  Edit that
file to set your Genie Space ID, add serving-endpoint subagents, or change the
orchestrator model — no code changes required.
"""

from contextlib import nullcontext
from pathlib import Path
from typing import AsyncGenerator

import mlflow
import yaml
from agents import Agent, Runner, function_tool, set_default_openai_api, set_default_openai_client
from agents.tracing import set_trace_processors
from databricks_openai import AsyncDatabricksOpenAI
from databricks_openai.agents import McpServer
from mlflow.genai.agent_server import invoke, stream
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)

from agent_server.utils import (
    build_mcp_url,
    get_user_workspace_client,
    process_agent_stream_events,
    sanitize_output_items,
)

# ---------------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(_CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

SUBAGENTS: list[dict] = _config.get("subagents", [])
ORCHESTRATOR_MODEL: str = _config.get("orchestrator_model", "databricks-claude-sonnet-4-5")

assert SUBAGENTS, (
    f"Configure at least one subagent in {_CONFIG_PATH}. "
    "Uncomment an entry and replace placeholder values."
)

# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

set_default_openai_client(AsyncDatabricksOpenAI())
set_default_openai_api("chat_completions")
set_trace_processors([])  # only use mlflow for trace processing
mlflow.openai.autolog()

_tool_client = AsyncDatabricksOpenAI()

# ---------------------------------------------------------------------------
# Subagent tools — one tool per non-genie subagent
# ---------------------------------------------------------------------------


def _make_subagent_tool(subagent: dict):
    """Create a function_tool for a single subagent definition."""
    endpoint = subagent["endpoint"]
    model = f"apps/{endpoint}" if subagent["type"] == "app" else endpoint

    async def _call(question: str) -> str:
        response = await _tool_client.responses.create(
            model=model,
            input=[{"role": "user", "content": question}],
        )
        return response.output_text

    _call.__name__ = f"query_{subagent['name']}"
    _call.__doc__ = subagent["description"]
    return function_tool(_call)


subagent_tools = [_make_subagent_tool(sa) for sa in SUBAGENTS if sa["type"] != "genie"]


# ---------------------------------------------------------------------------
# MCP server + orchestrator agent
# ---------------------------------------------------------------------------


async def init_mcp_server():
    """Create a Genie MCP server if a genie subagent is configured."""
    genie = next((sa for sa in SUBAGENTS if sa["type"] == "genie"), None)
    if genie is None:
        return nullcontext()
    return McpServer(
        url=build_mcp_url(f"/api/2.0/mcp/genie/{genie['space_id']}"),
        name=genie["description"],
    )


def _build_instructions() -> str:
    """Dynamically build orchestrator instructions from the configured subagents."""
    lines = [
        "You are an orchestrator agent for Enterprise Sales Analytics.",
        "Route the user's request to the most appropriate tool or data source:\n",
    ]
    for sa in SUBAGENTS:
        if sa["type"] == "genie":
            lines.append(
                f"- Use the Genie MCP tools for: {sa['description'].strip()}"
            )
        else:
            lines.append(
                f"- Use query_{sa['name']} for: {sa['description'].strip()}"
            )
    lines.append(
        "\nIf the user's question doesn't clearly match any tool, ask for "
        "clarification.  Always prefer the most specific tool available."
    )
    return "\n".join(lines)


def create_orchestrator_agent(mcp_server: McpServer) -> Agent:
    """Build the orchestrator agent with all tools and MCP servers."""
    return Agent(
        name="Orchestrator",
        instructions=_build_instructions(),
        model=ORCHESTRATOR_MODEL,
        mcp_servers=[mcp_server] if mcp_server else [],
        tools=subagent_tools,
    )


# ---------------------------------------------------------------------------
# MLflow Responses API handlers
# ---------------------------------------------------------------------------


def _normalize_input(request: ResponsesAgentRequest) -> list[dict]:
    """Convert request input to dicts the agents SDK can consume.

    The agents SDK ``Converter.items_to_messages`` expects assistant message
    ``content`` to be a list of content-part dicts (e.g.
    ``[{"type": "output_text", "text": "…"}]``).  Callers that send plain
    string content on assistant turns (the Responses API "easy" format) would
    otherwise crash the converter.  This helper normalises such messages.
    """
    messages = [i.model_dump() for i in request.input]
    for msg in messages:
        if (
            msg.get("role") == "assistant"
            and isinstance(msg.get("content"), str)
        ):
            msg["content"] = [{"type": "output_text", "text": msg["content"]}]
    return messages


@invoke()
async def invoke_handler(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    async with await init_mcp_server() as mcp_server:
        agent = create_orchestrator_agent(mcp_server)
        messages = _normalize_input(request)
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(output=sanitize_output_items(result.new_items))


@stream()
async def stream_handler(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    async with await init_mcp_server() as mcp_server:
        agent = create_orchestrator_agent(mcp_server)
        messages = _normalize_input(request)
        result = Runner.run_streamed(agent, input=messages)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
