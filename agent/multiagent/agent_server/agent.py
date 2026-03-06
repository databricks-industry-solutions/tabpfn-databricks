"""
Multi-agent orchestrator with Genie subagent for Enterprise Sales Analytics.

Configuration is loaded from ``config.yaml`` at the project root.  Edit that
file to set your Genie Space ID, add serving-endpoint subagents, or change the
orchestrator model — no code changes required.
"""

from contextlib import AsyncExitStack
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
# Patch: Databricks external MCP proxy returns NDJSON (multiple JSON-RPC
# messages on separate lines) but the MCP client expects a single object.
# Split on newlines and deliver each message individually.
# ---------------------------------------------------------------------------
import logging as _logging

import httpx as _httpx
from mcp.client.streamable_http import StreamableHTTPTransport
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage

_patch_logger = _logging.getLogger(__name__)

_original_handle_json = StreamableHTTPTransport._handle_json_response


async def _patched_handle_json_response(
    self,
    response: _httpx.Response,
    read_stream_writer,
    is_initialization: bool = False,
) -> None:
    content = await response.aread()
    lines = [line for line in content.split(b"\n") if line.strip()]
    if len(lines) <= 1:
        return await _original_handle_json(
            self, response, read_stream_writer, is_initialization
        )
    _patch_logger.debug("NDJSON response with %d messages — splitting", len(lines))
    for line in lines:
        try:
            message = JSONRPCMessage.model_validate_json(line)
            if is_initialization:
                self._maybe_extract_protocol_version_from_message(message)
            await read_stream_writer.send(SessionMessage(message))
        except Exception:
            _patch_logger.exception("Error parsing NDJSON line: %s", line[:200])


StreamableHTTPTransport._handle_json_response = _patched_handle_json_response

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


_MCP_TYPES = {"genie", "mcp"}
subagent_tools = [_make_subagent_tool(sa) for sa in SUBAGENTS if sa["type"] not in _MCP_TYPES]


# ---------------------------------------------------------------------------
# MCP server + orchestrator agent
# ---------------------------------------------------------------------------


_DEFAULT_MCP_TIMEOUT = 20
_EXTERNAL_MCP_TIMEOUT = 120


async def init_mcp_servers():
    """Create MCP servers for all genie and external mcp subagents."""
    servers = []
    for sa in SUBAGENTS:
        if sa["type"] == "genie":
            servers.append(McpServer(
                url=build_mcp_url(f"/api/2.0/mcp/genie/{sa['space_id']}"),
                name=sa["description"],
                client_session_timeout_seconds=_DEFAULT_MCP_TIMEOUT,
            ))
        elif sa["type"] == "mcp":
            servers.append(McpServer(
                url=build_mcp_url(f"/api/2.0/mcp/external/{sa['connection_name']}"),
                name=sa.get("server_name", sa["name"]),
                client_session_timeout_seconds=sa.get(
                    "timeout_seconds", _EXTERNAL_MCP_TIMEOUT
                ),
            ))
    return servers


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
        elif sa["type"] == "mcp":
            lines.append(
                f"- Use the {sa['name']} MCP tools for: {sa['description'].strip()}"
            )
        else:
            lines.append(
                f"- Use query_{sa['name']} for: {sa['description'].strip()}"
            )
    lines.append(
        "\nCRITICAL TIME CONSTRAINT: You have a 120-second hard timeout. "
        "Every Genie query+poll cycle takes ~15 seconds. Plan accordingly."
        "\n\nTABPFN FOR PREDICTIVE ANALYSIS (MANDATORY):"
        "\nWhenever the user's request involves ANY of the following intents, "
        "you MUST use the TabPFN MCP server for the predictive/analytical "
        "component. TabPFN is a state-of-the-art foundation model for "
        "tabular prediction — use it for ALL structured-data prediction tasks."
        "\n"
        "\nTrigger keywords and intents (non-exhaustive):"
        "\n  - predict, forecast, estimate, project"
        "\n  - recommend, suggest, advise, propose"
        "\n  - classify, categorize, segment, score, rank"
        "\n  - likelihood, probability, propensity, risk"
        "\n  - which deals will close, what will happen, expected outcome"
        "\n  - next best action, cross-sell, upsell opportunity"
        "\n  - churn, attrition, conversion, win probability"
        "\n  - what-if analysis, scenario modeling"
        "\n"
        "\nWorkflow: (1) Use Genie to fetch the relevant structured data, "
        "then (2) call TabPFN to perform the prediction/classification. "
        "Do NOT attempt to predict outcomes using only LLM reasoning — "
        "always delegate to TabPFN when structured data is available."
        "\n\nEFFICIENCY RULES (MANDATORY):"
        "\n1. NEVER issue more than 2 Genie queries total. Combine ALL data "
        "needs into ONE comprehensive query using JOINs, CTEs, window "
        "functions, and subqueries."
        "\n2. Include ALL computed columns you will need later (LTV, rankings, "
        "next-product labels, etc.) in that single query. Do NOT make a "
        "separate query for aggregations you could compute inline."
        "\n3. If a query returns empty results, check the schema hints in the "
        "tool description before retrying — do NOT run exploratory queries."
        "\n4. After getting Genie data, go DIRECTLY to TabPFN. Do not make "
        "additional Genie queries to 'refine' or 'get more data'."
        "\n5. Keep inline datasets to ~50 training rows to minimize "
        "transcription errors and token usage. Use LIMIT 50 with "
        "ORDER BY RAND() in SQL to sample."
        "\n\nMULTI-TOOL WORKFLOW (Genie → TabPFN):"
        "\n- Query Genie ONCE for all training data, labels, and metadata."
        "\n- Transform data in-context, then call TabPFN ONCE."
        "\n- Generate the final report from TabPFN results in-context."
        "\n\nTABPFN DATA PREPARATION (MANDATORY):"
        "\n1. Ensure every feature that distinguishes prediction targets is "
        "included in X_train and X_test. If you are predicting an outcome "
        "for multiple items (e.g., different products for the same customer, "
        "or different customers), each item must have at least one unique "
        "feature — otherwise the model returns identical predictions and "
        "the results are useless."
        "\n2. Build X_train and y_train in LOCKSTEP: for each row in the "
        "Genie result, append the feature array to X_train AND the label "
        "to y_train at the same time. Never build them separately. Before "
        "calling fit_and_predict_inline, explicitly STATE the count of "
        "X_train rows and y_train labels and confirm they match."
        "\n3. NEVER retry fit_and_predict_inline more than once. If you get "
        "a length mismatch, carefully re-extract from the original Genie "
        "response rather than tweaking the previous attempt."
        "\n\nIf the user's question doesn't clearly match any tool, ask for "
        "clarification. Always prefer the most specific tool available."
    )
    return "\n".join(lines)


def create_orchestrator_agent(mcp_servers: list[McpServer]) -> Agent:
    """Build the orchestrator agent with all tools and MCP servers."""
    return Agent(
        name="Orchestrator",
        instructions=_build_instructions(),
        model=ORCHESTRATOR_MODEL,
        mcp_servers=mcp_servers,
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
    async with AsyncExitStack() as stack:
        mcp_servers = [await stack.enter_async_context(s) for s in await init_mcp_servers()]
        agent = create_orchestrator_agent(mcp_servers)
        messages = _normalize_input(request)
        result = await Runner.run(agent, messages)
        return ResponsesAgentResponse(output=sanitize_output_items(result.new_items))


@stream()
async def stream_handler(request: ResponsesAgentRequest) -> AsyncGenerator[ResponsesAgentStreamEvent, None]:
    async with AsyncExitStack() as stack:
        mcp_servers = [await stack.enter_async_context(s) for s in await init_mcp_servers()]
        agent = create_orchestrator_agent(mcp_servers)
        messages = _normalize_input(request)
        result = Runner.run_streamed(agent, input=messages)

        async for event in process_agent_stream_events(result.stream_events()):
            yield event
