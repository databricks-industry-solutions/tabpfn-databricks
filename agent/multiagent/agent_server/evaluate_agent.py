import asyncio
import logging

import mlflow
from dotenv import load_dotenv
from mlflow.genai.agent_server import get_invoke_function
from mlflow.genai.scorers import (
    Completeness,
    ConversationalSafety,
    ConversationCompleteness,
    Fluency,
    KnowledgeRetention,
    RelevanceToQuery,
    Safety,
    ToolCallCorrectness,
    UserFrustration,
)
from mlflow.genai.simulators import ConversationSimulator
from mlflow.types.responses import ResponsesAgentRequest

load_dotenv(dotenv_path=".env", override=True)
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)

from agent_server import agent  # noqa: E402, F401

test_cases = [
    {
        "goal": "Understand the overall win rate broken down by lead source",
        "persona": "A VP of Sales who wants a quick performance overview.",
        "simulation_guidelines": [
            "Ask about win rates, then drill into which lead sources underperform.",
        ],
    },
    {
        "goal": "Find the top accounts by closed-won ACV and compare to their targets",
        "persona": "A revenue operations analyst preparing a board deck.",
        "simulation_guidelines": [
            "Start broad, then narrow to accounts exceeding their ACV target.",
            "Prefer short messages.",
        ],
    },
]

simulator = ConversationSimulator(
    test_cases=test_cases,
    max_turns=5,
    user_model="databricks:/databricks-claude-sonnet-4-5",
)

invoke_fn = get_invoke_function()
assert invoke_fn is not None, (
    "No function registered with the `@invoke` decorator found. "
    "Ensure agent_server.agent is imported."
)


if asyncio.iscoroutinefunction(invoke_fn):

    def predict_fn(input: list[dict], **kwargs) -> dict:
        req = ResponsesAgentRequest(input=input)
        response = asyncio.run(invoke_fn(req))
        return response.model_dump()
else:

    def predict_fn(input: list[dict], **kwargs) -> dict:
        req = ResponsesAgentRequest(input=input)
        response = invoke_fn(req)
        return response.model_dump()


def evaluate():
    mlflow.genai.evaluate(
        data=simulator,
        predict_fn=predict_fn,
        scorers=[
            Completeness(),
            ConversationCompleteness(),
            ConversationalSafety(),
            KnowledgeRetention(),
            UserFrustration(),
            Fluency(),
            RelevanceToQuery(),
            Safety(),
            ToolCallCorrectness(),
        ],
    )
