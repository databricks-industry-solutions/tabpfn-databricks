"""
End-to-end ground-truth evaluation for the multiagent pipeline.

Pre-computes ground truth by running known-correct SQL against Delta tables
and calling TabPFN directly, then compares against the agent's outputs
extracted from MLflow traces.

Run:  uv run agent-evaluate-e2e
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import mlflow
import yaml
from databricks import sql
from databricks.sdk.core import Config
from dotenv import load_dotenv
from mlflow.entities import Feedback, Trace
from mlflow.genai.agent_server import get_invoke_function
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import scorer
from mlflow.types.responses import ResponsesAgentRequest

load_dotenv(dotenv_path=".env", override=True)
logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

from agent_server import agent  # noqa: E402, F401
from agent_server.utils import build_mcp_url

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

with open(_CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

_CATALOG = "tabpfn_databricks"
_SCHEMA = "agent"
_FQN = f"{_CATALOG}.{_SCHEMA}"

# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

_TEST_CASES = [
    {
        "question": (
            "Which promotion type would most improve win rates "
            "for Fortune 500 accounts with ACV over 100k?"
        ),
        "ground_truth_sql": f"""
            SELECT a.segment, a.annual_revenue_mm, a.industry,
                o.acv, o.lead_source, o.days_in_pipeline,
                p.promotion_type,
                CASE WHEN o.stage = 'Closed/Won' THEN 1 ELSE 0 END AS won
            FROM {_FQN}.opportunities o
            JOIN {_FQN}.accounts a ON o.account_id = a.account_id
            JOIN {_FQN}.promotions p ON o.opportunity_id = p.opportunity_id
            WHERE o.stage IN ('Closed/Won', 'Closed/Lost')
            ORDER BY RAND() LIMIT 50
        """,
        "target_column": "won",
        "feature_columns": [
            "segment", "annual_revenue_mm", "industry",
            "acv", "lead_source", "days_in_pipeline", "promotion_type",
        ],
        "x_test_scenarios": [
            {
                "segment": "Fortune 500", "annual_revenue_mm": 18000,
                "industry": "SaaS", "acv": 150000,
                "lead_source": "Inbound", "days_in_pipeline": 90,
                "promotion_type": "Discount",
            },
            {
                "segment": "Fortune 500", "annual_revenue_mm": 18000,
                "industry": "SaaS", "acv": 150000,
                "lead_source": "Inbound", "days_in_pipeline": 90,
                "promotion_type": "Delivery Support",
            },
            {
                "segment": "Fortune 500", "annual_revenue_mm": 18000,
                "industry": "SaaS", "acv": 150000,
                "lead_source": "Inbound", "days_in_pipeline": 90,
                "promotion_type": "Enablement",
            },
        ],
        "expected_direction": (
            "The response should rank promotion types by predicted win "
            "probability and recommend the best one for Fortune 500 deals."
        ),
        "task_type": "classification",
    },
    {
        "question": (
            "Predict the win probability for a Mid-Market deal "
            "from each lead source."
        ),
        "ground_truth_sql": f"""
            SELECT a.segment, a.annual_revenue_mm, a.industry, a.region,
                o.acv, o.lead_source, o.days_in_pipeline,
                CASE WHEN o.stage = 'Closed/Won' THEN 1 ELSE 0 END AS won
            FROM {_FQN}.opportunities o
            JOIN {_FQN}.accounts a ON o.account_id = a.account_id
            WHERE o.stage IN ('Closed/Won', 'Closed/Lost')
              AND a.segment = 'Mid-Market'
            ORDER BY RAND() LIMIT 50
        """,
        "target_column": "won",
        "feature_columns": [
            "segment", "annual_revenue_mm", "industry", "region",
            "acv", "lead_source", "days_in_pipeline",
        ],
        "x_test_scenarios": [
            {
                "segment": "Mid-Market", "annual_revenue_mm": 345,
                "industry": "SaaS", "region": "North America",
                "acv": 24000, "lead_source": "Inbound",
                "days_in_pipeline": 60,
            },
            {
                "segment": "Mid-Market", "annual_revenue_mm": 345,
                "industry": "SaaS", "region": "North America",
                "acv": 24000, "lead_source": "Outbound",
                "days_in_pipeline": 60,
            },
            {
                "segment": "Mid-Market", "annual_revenue_mm": 345,
                "industry": "SaaS", "region": "North America",
                "acv": 24000, "lead_source": "Partner",
                "days_in_pipeline": 60,
            },
        ],
        "expected_direction": (
            "The response should present win probabilities per lead source "
            "and identify which lead source is strongest for Mid-Market deals."
        ),
        "task_type": "classification",
    },
    {
        "question": (
            "What ACV can we expect for a Fortune 500 account "
            "in the SaaS sector across different regions?"
        ),
        "ground_truth_sql": f"""
            SELECT a.segment, a.industry, a.region, a.annual_revenue_mm,
                o.lead_source, o.days_in_pipeline, o.acv
            FROM {_FQN}.opportunities o
            JOIN {_FQN}.accounts a ON o.account_id = a.account_id
            WHERE o.stage = 'Closed/Won'
            ORDER BY RAND() LIMIT 50
        """,
        "target_column": "acv",
        "feature_columns": [
            "segment", "industry", "region", "annual_revenue_mm",
            "lead_source", "days_in_pipeline",
        ],
        "x_test_scenarios": [
            {
                "segment": "Fortune 500", "industry": "SaaS",
                "region": "North America", "annual_revenue_mm": 18000,
                "lead_source": "Inbound", "days_in_pipeline": 90,
            },
            {
                "segment": "Fortune 500", "industry": "SaaS",
                "region": "Europe", "annual_revenue_mm": 18000,
                "lead_source": "Inbound", "days_in_pipeline": 90,
            },
            {
                "segment": "Fortune 500", "industry": "SaaS",
                "region": "APAC", "annual_revenue_mm": 18000,
                "lead_source": "Inbound", "days_in_pipeline": 90,
            },
        ],
        "expected_direction": (
            "The response should predict ACV values for Fortune 500 / SaaS "
            "accounts by region and recommend the highest-value region."
        ),
        "task_type": "regression",
    },
    {
        "question": (
            "Which region has the highest win probability "
            "for Fortune 500 deals?"
        ),
        "ground_truth_sql": f"""
            SELECT a.segment, a.annual_revenue_mm, a.industry, a.region,
                o.acv, o.lead_source, o.days_in_pipeline,
                CASE WHEN o.stage = 'Closed/Won' THEN 1 ELSE 0 END AS won
            FROM {_FQN}.opportunities o
            JOIN {_FQN}.accounts a ON o.account_id = a.account_id
            WHERE o.stage IN ('Closed/Won', 'Closed/Lost')
              AND a.segment = 'Fortune 500'
            ORDER BY RAND() LIMIT 50
        """,
        "target_column": "won",
        "feature_columns": [
            "segment", "annual_revenue_mm", "industry", "region",
            "acv", "lead_source", "days_in_pipeline",
        ],
        "x_test_scenarios": [
            {
                "segment": "Fortune 500", "annual_revenue_mm": 18000,
                "industry": "SaaS", "region": "North America",
                "acv": 113000, "lead_source": "Inbound",
                "days_in_pipeline": 90,
            },
            {
                "segment": "Fortune 500", "annual_revenue_mm": 18000,
                "industry": "SaaS", "region": "Europe",
                "acv": 113000, "lead_source": "Inbound",
                "days_in_pipeline": 90,
            },
            {
                "segment": "Fortune 500", "annual_revenue_mm": 18000,
                "industry": "SaaS", "region": "APAC",
                "acv": 113000, "lead_source": "Inbound",
                "days_in_pipeline": 90,
            },
            {
                "segment": "Fortune 500", "annual_revenue_mm": 18000,
                "industry": "SaaS", "region": "LATAM",
                "acv": 113000, "lead_source": "Inbound",
                "days_in_pipeline": 90,
            },
        ],
        "expected_direction": (
            "The response should rank regions by predicted win probability "
            "and identify the best region for Fortune 500 deals."
        ),
        "task_type": "classification",
    },
    {
        "question": (
            "Predict days in pipeline for a Mid-Market Outbound deal "
            "versus an Inbound deal."
        ),
        "ground_truth_sql": f"""
            SELECT a.segment, a.industry, a.region,
                o.lead_source, o.acv, o.has_promotion, o.days_in_pipeline
            FROM {_FQN}.opportunities o
            JOIN {_FQN}.accounts a ON o.account_id = a.account_id
            WHERE o.stage IN ('Closed/Won', 'Closed/Lost')
              AND a.segment = 'Mid-Market'
            ORDER BY RAND() LIMIT 50
        """,
        "target_column": "days_in_pipeline",
        "feature_columns": [
            "segment", "industry", "region", "lead_source",
            "acv", "has_promotion",
        ],
        "x_test_scenarios": [
            {
                "segment": "Mid-Market", "industry": "SaaS",
                "region": "North America", "lead_source": "Outbound",
                "acv": 24000, "has_promotion": False,
            },
            {
                "segment": "Mid-Market", "industry": "SaaS",
                "region": "North America", "lead_source": "Inbound",
                "acv": 24000, "has_promotion": False,
            },
            {
                "segment": "Mid-Market", "industry": "SaaS",
                "region": "North America", "lead_source": "Outbound",
                "acv": 24000, "has_promotion": True,
            },
            {
                "segment": "Mid-Market", "industry": "SaaS",
                "region": "North America", "lead_source": "Inbound",
                "acv": 24000, "has_promotion": True,
            },
        ],
        "expected_direction": (
            "The response should predict days in pipeline and compare "
            "Outbound vs Inbound, noting whether promotions accelerate deals."
        ),
        "task_type": "regression",
    },
]


# ---------------------------------------------------------------------------
# Ground truth: SQL execution
# ---------------------------------------------------------------------------


def _run_sql(query: str, retries: int = 3) -> list[dict]:
    """Execute SQL against Delta tables and return rows as dicts."""
    cfg = Config()
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID is required for ground truth SQL. "
            "Set it in your .env file."
        )

    for attempt in range(1, retries + 1):
        try:
            conn = sql.connect(
                server_hostname=cfg.host,
                http_path=f"/sql/1.0/warehouses/{warehouse_id}",
                credentials_provider=lambda: cfg.authenticate,
            )
            try:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    columns = [desc[0] for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
            finally:
                conn.close()
        except Exception as exc:
            logger.warning("SQL attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))

    return []


# ---------------------------------------------------------------------------
# Ground truth: TabPFN direct call
# ---------------------------------------------------------------------------


async def _call_tabpfn(
    x_train: list[list],
    y_train: list,
    x_test: list[list],
    task_type: str,
) -> list | None:
    """Call TabPFN MCP directly and return predictions.

    Returns ``None`` if the call fails (connection issues, auth, etc.).
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    tabpfn_sa = next(
        (
            sa
            for sa in _config.get("subagents", [])
            if sa.get("type") == "mcp" and sa.get("name") == "tabpfn"
        ),
        None,
    )
    if tabpfn_sa is None:
        logger.warning("TabPFN subagent not found in config.yaml — skipping")
        return None

    url = build_mcp_url(
        f"/api/2.0/mcp/external/{tabpfn_sa['connection_name']}"
    )
    cfg = Config()
    auth_headers = cfg.authenticate()

    raw_texts: list[str] = []
    is_error = False

    try:
        async with streamablehttp_client(
            url=url, headers=auth_headers
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                arguments = {
                    "X_train": x_train,
                    "y_train": y_train,
                    "X_test": x_test,
                    "task_type": task_type,
                }
                if task_type == "classification":
                    arguments["output_type"] = "probas"
                result = await session.call_tool(
                    "fit_and_predict_inline",
                    arguments=arguments,
                )
                is_error = getattr(result, "isError", False)
                for part in result.content:
                    text = getattr(part, "text", None)
                    if text is not None:
                        raw_texts.append(text)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        logger.exception("TabPFN direct call failed — predictions unavailable")
        return None

    if is_error:
        logger.warning("TabPFN tool returned an error: %s", raw_texts)
        return None

    for text in raw_texts:
        if not text.strip():
            continue
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "predictions" in parsed:
                return parsed["predictions"]
            return parsed
        except json.JSONDecodeError:
            logger.warning(
                "Could not parse TabPFN response part: %.200s", text
            )

    logger.warning(
        "No parseable predictions in TabPFN response (got %d content parts, "
        "texts: %s)",
        len(raw_texts),
        [t[:100] for t in raw_texts],
    )
    return None


# ---------------------------------------------------------------------------
# Ground truth: setup
# ---------------------------------------------------------------------------


async def _setup_ground_truth() -> list[dict]:
    """Pre-compute ground truth for every test case.

    For each case: run the known-correct SQL, split into features / target,
    call TabPFN directly, and store everything in ``expectations``.
    """
    eval_data: list[dict] = []

    for tc in _TEST_CASES:
        logger.info("Computing ground truth for: %s", tc["question"][:60])
        rows = _run_sql(tc["ground_truth_sql"])
        if not rows:
            logger.warning("No rows returned — skipping test case")
            continue

        feature_cols = tc["feature_columns"]
        target_col = tc["target_column"]
        task_type = tc["task_type"]

        x_train = [
            [row[col] for col in feature_cols if col in row]
            for row in rows
        ]
        y_train = [row[target_col] for row in rows]

        expectations = {
            "expected_tools": [
                "query_space",
                "poll_response",
                "fit_and_predict_inline",
            ],
            "ground_truth_columns": feature_cols,
            "ground_truth_row_count": len(rows),
            "expected_direction": tc["expected_direction"],
            "task_type": task_type,
            "target_column": target_col,
            "ground_truth_x_train": x_train,
            "ground_truth_y_train": y_train,
        }

        eval_data.append(
            {
                "inputs": {
                    "input": [{"role": "user", "content": tc["question"]}],
                },
                "expectations": expectations,
            }
        )

    if not eval_data:
        raise RuntimeError("No test cases produced valid ground truth data")

    return eval_data


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------


def _find_spans(trace: Trace, name_pattern: str) -> list:
    """Find spans whose name contains *name_pattern* (case-insensitive)."""
    pattern = name_pattern.lower()
    return [
        span
        for span in trace.search_spans()
        if pattern in span.name.lower()
    ]


def _parse_span_data(data) -> dict | list | None:
    """Best-effort parse of span inputs/outputs (may be dict, str, or None)."""
    if data is None:
        return None
    if isinstance(data, (dict, list)):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def _unwrap_mcp_text(data) -> dict | list | str | None:
    """Parse span data, unwrapping MCP content-part format if present.

    MCP tool spans store outputs as ``{"type": "text", "text": "<JSON>"}``
    where the actual payload is a JSON string inside the ``text`` field.
    """
    parsed = _parse_span_data(data)
    if isinstance(parsed, dict) and parsed.get("type") == "text" and "text" in parsed:
        inner = parsed["text"]
        if isinstance(inner, str):
            try:
                return json.loads(inner)
            except (json.JSONDecodeError, TypeError):
                return inner
        return inner
    return parsed


def _extract_genie_columns(trace: Trace) -> list[str]:
    """Extract column names from the Genie query result manifest.

    Walks the trace in reverse to find the last ``query_space`` or
    ``poll_response`` span whose output contains a successful
    ``statement_response`` with a column manifest.
    """
    for span in reversed(trace.search_spans()):
        if not any(
            p in span.name.lower()
            for p in ("query_space", "poll_response")
        ):
            continue
        outputs = _unwrap_mcp_text(span.outputs)
        if not isinstance(outputs, dict):
            continue
        attachments = outputs.get("content", {}).get("queryAttachments", [])
        for att in attachments:
            columns = (
                att.get("statement_response", {})
                .get("manifest", {})
                .get("schema", {})
                .get("columns", [])
            )
            if columns:
                return [c["name"] for c in columns]
    return []


def _genie_string_matches(genie_str: str, agent_val) -> bool:
    """Check if a Genie string value matches a typed agent value."""
    if isinstance(agent_val, str):
        return genie_str == agent_val
    if isinstance(agent_val, bool):
        return genie_str.lower() in ("true", "1") if agent_val else genie_str.lower() in ("false", "0")
    if isinstance(agent_val, (int, float)):
        try:
            return float(genie_str) == float(agent_val)
        except (ValueError, TypeError):
            return False
    return str(agent_val) == genie_str


def _resolve_agent_columns(
    trace: Trace,
    agent_x_train: list[list],
    genie_columns: list[str],
) -> list[str]:
    """Determine which Genie columns the agent actually kept.

    When the agent drops columns (e.g. a constant ``segment``), the Genie
    manifest has more column names than X_train/X_test features.  This
    function compares the first Genie result row against the first agent
    X_train row using a two-pointer walk to identify the kept columns.
    """
    for span in reversed(trace.search_spans()):
        if not any(
            p in span.name.lower()
            for p in ("query_space", "poll_response")
        ):
            continue
        outputs = _unwrap_mcp_text(span.outputs)
        if not isinstance(outputs, dict):
            continue
        attachments = outputs.get("content", {}).get("queryAttachments", [])
        for att in attachments:
            data_array = (
                att.get("statement_response", {})
                .get("result", {})
                .get("data_array", [])
            )
            if not data_array:
                continue
            genie_row = [
                v.get("string_value", "")
                for v in data_array[0].get("values", [])
            ]
            if not genie_row or not agent_x_train:
                continue

            agent_row = agent_x_train[0]
            kept: list[str] = []
            ai = 0
            for gi, gval in enumerate(genie_row):
                if ai >= len(agent_row):
                    break
                if gi < len(genie_columns) and _genie_string_matches(gval, agent_row[ai]):
                    kept.append(genie_columns[gi])
                    ai += 1

            if len(kept) == len(agent_row):
                return kept
    return []


def _align_features(
    gt_x_train: list[list],
    gt_columns: list[str],
    agent_x_test: list[list],
    agent_columns: list[str],
) -> tuple[list[list], list[list], list[str]] | None:
    """Subset GT X_train and agent X_test to their shared columns.

    Returns ``(subsetted_x_train, subsetted_x_test, shared_cols)`` or
    ``None`` if alignment is not possible (no overlap or column-count
    mismatch).
    """
    if not gt_x_train or not agent_x_test:
        return None

    gt_width = len(gt_x_train[0])
    agent_width = len(agent_x_test[0])

    if len(gt_columns) != gt_width:
        return None
    if len(agent_columns) != agent_width:
        return None  # caller should use _resolve_agent_columns first

    shared = [c for c in gt_columns if c in agent_columns]
    if not shared:
        return None

    gt_indices = [gt_columns.index(c) for c in shared]
    agent_indices = [agent_columns.index(c) for c in shared]
    sub_x_train = [[row[i] for i in gt_indices] for row in gt_x_train]
    sub_x_test = [[row[i] for i in agent_indices] for row in agent_x_test]
    return sub_x_train, sub_x_test, shared


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------


@scorer
def tool_workflow(trace: Trace) -> Feedback:
    """Verify the agent followed the predictive workflow.

    Required: ``query_space`` and ``fit_and_predict_inline``.
    Optional: ``poll_response`` (only needed when Genie doesn't return
    results immediately from ``query_space``).
    """
    required = ["query_space", "fit_and_predict_inline"]
    all_spans = trace.search_spans()
    tool_names = [span.name for span in all_spans]
    tool_names_lower = [n.lower() for n in tool_names]

    missing = [
        t for t in required
        if not any(t in n for n in tool_names_lower)
    ]

    if not missing:
        return Feedback(
            name="correct_workflow",
            value=True,
            rationale=f"All required tools called: {required}",
        )
    return Feedback(
        name="correct_workflow",
        value=False,
        rationale=f"Missing tool calls: {missing}. Observed: {tool_names}",
    )


@scorer
def training_data_quality(
    expectations: dict, trace: Trace
) -> list[Feedback]:
    """Compare the agent's training data against ground truth."""
    feedbacks: list[Feedback] = []
    gt_columns = set(expectations.get("ground_truth_columns", []))
    gt_row_count = expectations.get("ground_truth_row_count", 0)

    tabpfn_spans = _find_spans(trace, "fit_and_predict_inline")
    if not tabpfn_spans:
        feedbacks.append(
            Feedback(
                name="training_data_found",
                value=False,
                rationale="No fit_and_predict_inline span found in trace",
            )
        )
        return feedbacks

    feedbacks.append(
        Feedback(
            name="training_data_found",
            value=True,
            rationale="fit_and_predict_inline span found in trace",
        )
    )

    span_inputs = _parse_span_data(tabpfn_spans[0].inputs)
    if not isinstance(span_inputs, dict):
        feedbacks.append(
            Feedback(
                name="column_overlap_pct",
                value=0.0,
                rationale="Could not parse TabPFN span inputs",
            )
        )
        return feedbacks

    x_train = span_inputs.get("X_train") or span_inputs.get(
        "arguments", {}
    ).get("X_train")

    if not x_train or not isinstance(x_train, list):
        feedbacks.append(
            Feedback(
                name="column_overlap_pct",
                value=0.0,
                rationale="X_train not found or empty in span inputs",
            )
        )
        return feedbacks

    # Column overlap
    agent_columns: set[str] = set()
    first_row = x_train[0] if x_train else {}
    if isinstance(first_row, dict):
        agent_columns = set(first_row.keys())
    else:
        target_col = expectations.get("target_column", "")
        genie_cols = _extract_genie_columns(trace)
        agent_columns = {c for c in genie_cols if c != target_col}

    if gt_columns and agent_columns:
        overlap = len(gt_columns & agent_columns) / len(gt_columns)
    elif gt_columns:
        overlap = 0.0
    else:
        overlap = 1.0

    feedbacks.append(
        Feedback(
            name="column_overlap_pct",
            value=round(overlap, 2),
            rationale=(
                f"Agent columns: {sorted(agent_columns)}, "
                f"expected: {sorted(gt_columns)}, "
                f"overlap: {overlap:.0%}"
            ),
        )
    )

    # Row count ratio
    agent_row_count = len(x_train)
    if gt_row_count > 0:
        ratio = agent_row_count / gt_row_count
    else:
        ratio = 1.0

    feedbacks.append(
        Feedback(
            name="row_count_ratio",
            value=round(ratio, 2),
            rationale=(
                f"Agent rows: {agent_row_count}, "
                f"ground truth rows: {gt_row_count}, "
                f"ratio: {ratio:.2f}"
            ),
        )
    )

    # y_train alignment check
    y_train = span_inputs.get("y_train") or span_inputs.get(
        "arguments", {}
    ).get("y_train")
    y_len = len(y_train) if isinstance(y_train, list) else 0
    aligned = y_len == agent_row_count

    feedbacks.append(
        Feedback(
            name="xy_train_aligned",
            value=aligned,
            rationale=(
                f"X_train rows: {agent_row_count}, "
                f"y_train length: {y_len}"
            ),
        )
    )

    return feedbacks


@scorer
def prediction_accuracy(
    expectations: dict, trace: Trace
) -> list[Feedback]:
    """Compare agent predictions against ground truth TabPFN predictions.

    Extracts the agent's X_test from the trace, then calls TabPFN directly
    with the ground truth training data and the agent's X_test so that
    prediction counts always align.
    """
    feedbacks: list[Feedback] = []
    task_type = expectations.get("task_type", "classification")

    tabpfn_spans = _find_spans(trace, "fit_and_predict_inline")
    if not tabpfn_spans:
        feedbacks.append(
            Feedback(
                name="has_prediction",
                value=False,
                rationale="No fit_and_predict_inline span in trace",
            )
        )
        return feedbacks

    # Extract agent predictions from span outputs
    span_outputs = _unwrap_mcp_text(tabpfn_spans[0].outputs)
    agent_preds = None

    if isinstance(span_outputs, list):
        agent_preds = span_outputs
    elif isinstance(span_outputs, dict):
        agent_preds = (
            span_outputs.get("predictions")
            or span_outputs.get("output")
            or span_outputs.get("result")
        )
        if agent_preds is None:
            for v in span_outputs.values():
                if isinstance(v, list):
                    agent_preds = v
                    break
    elif isinstance(span_outputs, str):
        try:
            parsed = json.loads(span_outputs)
            if isinstance(parsed, list):
                agent_preds = parsed
        except (json.JSONDecodeError, TypeError):
            pass

    feedbacks.append(
        Feedback(
            name="has_prediction",
            value=agent_preds is not None and len(agent_preds) > 0,
            rationale=(
                f"Agent produced {len(agent_preds)} predictions"
                if agent_preds
                else "No predictions extracted from trace"
            ),
        )
    )

    if agent_preds is None:
        return feedbacks

    # Recompute GT predictions using the agent's X_test and GT training data,
    # subsetting both to shared columns when feature counts differ.
    span_inputs = tabpfn_spans[0].inputs
    agent_x_test = (
        span_inputs.get("X_test") if isinstance(span_inputs, dict) else None
    )
    x_train = expectations.get("ground_truth_x_train")
    y_train = expectations.get("ground_truth_y_train")
    gt_columns = expectations.get("ground_truth_columns", [])
    target_col = expectations.get("target_column", "")
    genie_cols = _extract_genie_columns(trace)
    agent_feature_cols = [c for c in genie_cols if c != target_col]

    if agent_x_test and len(agent_feature_cols) != len(agent_x_test[0]):
        agent_x_train = (
            span_inputs.get("X_train") if isinstance(span_inputs, dict) else None
        )
        if agent_x_train:
            agent_feature_cols = _resolve_agent_columns(
                trace, agent_x_train, genie_cols
            )

    gt_preds = None
    if x_train and y_train and agent_x_test:
        aligned = _align_features(
            x_train, gt_columns, agent_x_test, agent_feature_cols
        )
        if aligned:
            sub_x_train, sub_x_test, shared_cols = aligned
            gt_preds = asyncio.run(
                _call_tabpfn(sub_x_train, y_train, sub_x_test, task_type)
            )

    if gt_preds is None:
        feedbacks.append(
            Feedback(
                name="gt_recompute",
                value=False,
                rationale=(
                    "Could not recompute GT predictions "
                    f"(x_train={bool(x_train)}, y_train={bool(y_train)}, "
                    f"agent_x_test={bool(agent_x_test)})"
                ),
            )
        )
        return feedbacks

    # Ranking / direction comparison
    gt_vals = _extract_numeric_predictions(gt_preds)
    agent_vals = _extract_numeric_predictions(agent_preds)

    if not gt_vals or not agent_vals or len(gt_vals) != len(agent_vals):
        feedbacks.append(
            Feedback(
                name="gt_recompute",
                value=False,
                rationale=(
                    f"Numeric extraction mismatch: "
                    f"gt_vals={len(gt_vals)}, agent_vals={len(agent_vals)}"
                ),
            )
        )
        return feedbacks

    if task_type == "classification":
        gt_best = gt_vals.index(max(gt_vals))
        agent_best = agent_vals.index(max(agent_vals))
        rank_match = gt_best == agent_best

        feedbacks.append(
            Feedback(
                name="prediction_rank_match",
                value=rank_match,
                rationale=(
                    f"Ground truth best index: {gt_best}, "
                    f"agent best index: {agent_best}. "
                    f"GT values: {gt_vals}, agent values: {agent_vals}"
                ),
            )
        )

    elif task_type == "regression":
        within_tolerance = all(
            abs(g - a) / max(abs(g), 1) < 0.5
            for g, a in zip(gt_vals, agent_vals)
        )
        feedbacks.append(
            Feedback(
                name="prediction_within_tolerance",
                value=within_tolerance,
                rationale=(
                    f"GT: {gt_vals}, agent: {agent_vals}. "
                    f"Within 50% tolerance: {within_tolerance}"
                ),
            )
        )

    return feedbacks


def _extract_numeric_predictions(preds: list) -> list[float]:
    """Pull numeric values from a heterogeneous prediction list."""
    values: list[float] = []
    for p in preds:
        if isinstance(p, (int, float)):
            values.append(float(p))
        elif isinstance(p, list) and p:
            nums = [x for x in p if isinstance(x, (int, float))]
            if len(nums) == 2:
                values.append(float(nums[1]))
            elif nums:
                values.append(float(max(nums)))
        elif isinstance(p, dict):
            for key in ("probability", "prediction", "value", "score", "prob"):
                if key in p and isinstance(p[key], (int, float)):
                    values.append(float(p[key]))
                    break
            else:
                nums = [v for v in p.values() if isinstance(v, (int, float))]
                if nums:
                    values.append(float(nums[0]))
        elif isinstance(p, str):
            try:
                values.append(float(p))
            except ValueError:
                pass
    return values


answer_quality = make_judge(
    name="answer_quality",
    instructions="""
    Evaluate whether the sales analytics agent's response correctly presents
    ML-based predictions and a data-driven recommendation.

    User question: {{ inputs }}
    Agent response: {{ outputs }}
    Expected prediction behaviour: {{ expectations }}
    Execution trace: {{ trace }}

    Criteria:
    1. The response contains specific numerical predictions (probabilities or
       predicted values), not just qualitative opinions.
    2. The recommendation is logically consistent with the numbers presented.
    3. The response is concise (3-5 key findings) and does not hallucinate data
       that was not retrieved by the tools shown in the trace.

    Respond with "yes" if all three criteria are met, "no" otherwise.
    """,
    model="databricks:/databricks-claude-sonnet-4-5",
)


# ---------------------------------------------------------------------------
# Predict function
# ---------------------------------------------------------------------------

invoke_fn = get_invoke_function()
assert invoke_fn is not None, (
    "No function registered with the @invoke decorator found. "
    "Ensure agent_server.agent is imported."
)

if asyncio.iscoroutinefunction(invoke_fn):

    def predict_fn(input: list[dict], **kwargs) -> dict:  # noqa: A002
        req = ResponsesAgentRequest(input=input)
        response = asyncio.run(invoke_fn(req))
        return response.model_dump()

else:

    def predict_fn(input: list[dict], **kwargs) -> dict:  # noqa: A002
        req = ResponsesAgentRequest(input=input)
        response = invoke_fn(req)
        return response.model_dump()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def evaluate():
    """Run end-to-end ground-truth evaluation."""
    logger.info("Setting up ground truth …")
    eval_data = asyncio.run(_setup_ground_truth())
    logger.info(
        "Ground truth ready — %d test cases. Starting evaluation …",
        len(eval_data),
    )

    mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=predict_fn,
        scorers=[
            tool_workflow,
            training_data_quality,
            prediction_accuracy,
            answer_quality,
        ],
    )
