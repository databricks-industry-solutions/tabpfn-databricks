import os
import logging
import time
import pandas as pd
from databricks.sdk.core import Config
from databricks import sql

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CATALOG = "tabpfn_databricks"
SCHEMA = "agent"

_cache: dict[str, pd.DataFrame] = {}


def _get_connection():
    cfg = Config()
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError("DATABRICKS_WAREHOUSE_ID environment variable is not set")
    logger.info("Connecting to warehouse %s on %s", warehouse_id, cfg.host)
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: cfg.authenticate,
    )


def _run_query(query: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(1, retries + 1):
        try:
            conn = _get_connection()
            try:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    return pd.DataFrame(rows, columns=columns)
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Query attempt %d/%d failed: %s", attempt, retries, e)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    return pd.DataFrame()


def _load_and_cache(key: str, query_fn) -> pd.DataFrame:
    if key not in _cache:
        logger.info("Loading dataset: %s", key)
        _cache[key] = query_fn()
        logger.info("Loaded %s: %d rows", key, len(_cache[key]))
    return _cache[key]


def get_opportunities() -> pd.DataFrame:
    return _load_and_cache("opportunities", lambda: _run_query(f"""
        SELECT
            o.*,
            CASE WHEN o.stage = 'Closed/Won' THEN 1 ELSE 0 END AS is_won,
            DATE_TRUNC('MONTH', TO_DATE(o.created_date)) AS created_month
        FROM {CATALOG}.{SCHEMA}.opportunities o
    """))


def get_account_opportunities() -> pd.DataFrame:
    return _load_and_cache("account_opportunities", lambda: _run_query(f"""
        SELECT
            o.opportunity_id, o.account_id, o.rep_id, o.lead_source,
            o.acv, o.stage, o.created_date, o.close_date, o.days_in_pipeline,
            a.account_name, a.segment, a.industry, a.region,
            a.employee_count, a.annual_revenue_mm,
            DATE_TRUNC('MONTH', TO_DATE(o.created_date)) AS created_month
        FROM {CATALOG}.{SCHEMA}.opportunities o
        JOIN {CATALOG}.{SCHEMA}.accounts a ON o.account_id = a.account_id
    """))


def get_product_revenue() -> pd.DataFrame:
    return _load_and_cache("product_revenue", lambda: _run_query(f"""
        SELECT
            op.opp_product_id, op.opportunity_id, op.product_id,
            op.line_acv, op.discount_pct,
            p.product_name, p.tier, p.category, p.list_acv
        FROM {CATALOG}.{SCHEMA}.opportunity_products op
        JOIN {CATALOG}.{SCHEMA}.products p ON op.product_id = p.product_id
    """))


def get_account_rep_summary() -> pd.DataFrame:
    return _load_and_cache("account_rep_summary", lambda: _run_query(f"""
        SELECT
            o.account_id, a.account_name, a.segment, a.region, a.industry,
            a.annual_acv_target, o.rep_id, r.rep_name, r.team,
            o.acv, o.stage, o.created_date,
            DATE_TRUNC('MONTH', TO_DATE(o.created_date)) AS created_month
        FROM {CATALOG}.{SCHEMA}.opportunities o
        JOIN {CATALOG}.{SCHEMA}.accounts a ON o.account_id = a.account_id
        JOIN {CATALOG}.{SCHEMA}.sales_reps r ON o.rep_id = r.rep_id
    """))


def get_account_target_summary() -> pd.DataFrame:
    return _load_and_cache("account_target_summary", lambda: _run_query(f"""
        WITH account_won_acv AS (
            SELECT
                a.account_id, a.account_name, a.annual_acv_target,
                COALESCE(SUM(CASE WHEN o.stage = 'Closed/Won' THEN o.acv ELSE 0 END), 0) AS won_acv
            FROM {CATALOG}.{SCHEMA}.accounts a
            LEFT JOIN {CATALOG}.{SCHEMA}.opportunities o ON a.account_id = o.account_id
            GROUP BY a.account_id, a.account_name, a.annual_acv_target
        )
        SELECT
            *,
            CASE WHEN annual_acv_target > 0
                 THEN (won_acv / annual_acv_target) * 100
                 ELSE 0 END AS attainment_pct
        FROM account_won_acv
    """))


def get_promotion_analysis() -> pd.DataFrame:
    return _load_and_cache("promotion_analysis", lambda: _run_query(f"""
        SELECT
            o.opportunity_id, o.acv, o.stage, o.days_in_pipeline,
            o.has_promotion, o.lead_source,
            p.promotion_id, p.promotion_type, p.had_effect, p.applied_date,
            CASE WHEN o.stage = 'Closed/Won' THEN 1 ELSE 0 END AS is_won,
            a.segment, a.region
        FROM {CATALOG}.{SCHEMA}.opportunities o
        LEFT JOIN {CATALOG}.{SCHEMA}.promotions p ON o.opportunity_id = p.opportunity_id
        LEFT JOIN {CATALOG}.{SCHEMA}.accounts a ON o.account_id = a.account_id
    """))


def clear_cache():
    _cache.clear()
    logger.info("Cache cleared")


def chat_with_agent(messages: list[dict]) -> str:
    """Send conversation history to the multiagent and return the response.

    When MULTIAGENT_ENDPOINT is set (local dev), POSTs directly to that URL.
    Otherwise, calls the deployed Databricks App via DatabricksOpenAI.
    """
    endpoint = os.getenv("MULTIAGENT_ENDPOINT")
    if endpoint:
        import requests

        resp = requests.post(endpoint, json={"input": messages}, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        parts = []
        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        parts.append(part.get("text", ""))
        return "\n".join(parts) or "No response"

    from databricks_openai import DatabricksOpenAI

    app_name = os.getenv("MULTIAGENT_APP_NAME")
    if not app_name:
        raise RuntimeError(
            "MULTIAGENT_APP_NAME environment variable is not set. "
            "Set it to the name of the deployed multiagent Databricks App."
        )

    client = DatabricksOpenAI()
    response = client.responses.create(
        model=f"apps/{app_name}",
        input=messages,
        timeout=600,
    )
    return response.output_text
