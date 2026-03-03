import logging

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import backend

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COLORS = {
    "primary": "#1B3A5C",
    "accent": "#4A90D9",
    "bg": "#F7F8FA",
    "card": "#FFFFFF",
    "text": "#2C3E50",
    "muted": "#8895A7",
    "success": "#27AE60",
    "warning": "#F39C12",
    "danger": "#E74C3C",
}

STAGE_ORDER = ["Discovery", "Demo", "Negotiation", "Closed/Won", "Closed/Lost"]
STAGE_COLORS = {
    "Discovery": "#4A90D9",
    "Demo": "#F5A623",
    "Negotiation": "#7B68EE",
    "Closed/Won": "#27AE60",
    "Closed/Lost": "#E74C3C",
}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    title="Enterprise Sales Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server


def fmt_currency(val, compact=True):
    if pd.isna(val):
        return "$0"
    if compact and abs(val) >= 1_000_000:
        return f"${val / 1_000_000:,.1f}M"
    if compact and abs(val) >= 1_000:
        return f"${val / 1_000:,.1f}K"
    return f"${val:,.0f}"


def fmt_pct(val):
    if pd.isna(val):
        return "0.0%"
    return f"{val * 100:.1f}%"


def fmt_number(val):
    if pd.isna(val):
        return "0"
    return f"{val:,.0f}"


def chart_layout(fig, height=380):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color=COLORS["text"]),
        margin=dict(l=40, r=20, t=10, b=40),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="#EEF0F4", linecolor="#DEE2E8")
    fig.update_yaxes(gridcolor="#EEF0F4", linecolor="#DEE2E8")
    return fig


def empty_fig(height=380):
    fig = go.Figure()
    chart_layout(fig, height)
    return fig


CHAT_PANEL_STYLE = {
    "position": "fixed",
    "bottom": "100px",
    "right": "30px",
    "width": "420px",
    "height": "600px",
    "display": "none",
    "flexDirection": "column",
    "zIndex": 1000,
    "borderRadius": "16px",
    "boxShadow": "0 8px 32px rgba(0,0,0,0.18)",
    "overflow": "hidden",
    "backgroundColor": COLORS["card"],
    "border": "1px solid #E0E4EA",
}


def _typing_indicator():
    return html.Div(
        html.Div(
            "Thinking…",
            style={
                "backgroundColor": "#F0F2F5",
                "padding": "10px 14px",
                "borderRadius": "16px 16px 16px 4px",
                "color": COLORS["muted"],
                "fontSize": "0.85rem",
                "fontStyle": "italic",
            },
        ),
        style={"display": "flex", "justifyContent": "flex-start", "marginBottom": "12px"},
    )


def render_chat_messages(messages, typing=False):
    if not messages:
        return [html.Div(
            [
                html.I(className="fas fa-comments", style={
                    "fontSize": "2rem", "color": COLORS["muted"], "marginBottom": "12px",
                }),
                html.P(
                    "Ask me anything about your sales data!",
                    style={"color": COLORS["muted"], "fontSize": "0.9rem", "margin": "0"},
                ),
            ],
            style={"textAlign": "center", "paddingTop": "40%"},
        )]

    elements = []
    for msg in messages:
        if msg["role"] == "user":
            elements.append(html.Div(
                html.Div(msg["content"], style={
                    "backgroundColor": COLORS["primary"], "color": "white",
                    "padding": "10px 14px", "borderRadius": "16px 16px 4px 16px",
                    "maxWidth": "80%", "fontSize": "0.9rem",
                    "lineHeight": "1.5", "wordBreak": "break-word",
                }),
                style={"display": "flex", "justifyContent": "flex-end", "marginBottom": "12px"},
            ))
        else:
            elements.append(html.Div(
                html.Div(
                    dcc.Markdown(
                        msg["content"],
                        style={"margin": "0", "fontSize": "0.9rem", "lineHeight": "1.5"},
                    ),
                    style={
                        "backgroundColor": "#F0F2F5", "padding": "10px 14px",
                        "borderRadius": "16px 16px 16px 4px", "maxWidth": "85%",
                        "wordBreak": "break-word", "overflow": "auto",
                    },
                ),
                style={"display": "flex", "justifyContent": "flex-start", "marginBottom": "12px"},
            ))

    if typing:
        elements.append(_typing_indicator())
    return elements


def make_card(title, children, height=None):
    style = {}
    if height:
        style["minHeight"] = height
    return dbc.Card(
        [
            dbc.CardHeader(
                html.H6(title, className="mb-0 fw-semibold", style={"color": COLORS["primary"]}),
                className="bg-white border-bottom",
            ),
            dbc.CardBody(children, style=style),
        ],
        className="shadow-sm border-0 h-100",
    )


def _prep_df(df, date_col="created_month", numeric_cols=None):
    """Cast types so charts and aggregations work correctly."""
    df = df.copy()
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    for col in numeric_cols or []:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Layout — built immediately so the server responds to health checks.
# All data‑dependent content is rendered via callbacks.
# ---------------------------------------------------------------------------

app.layout = dbc.Container(
    [
        dcc.Store(id="data-loaded", data=False),
        dcc.Interval(id="startup-trigger", interval=500, max_intervals=1),

        # Header
        dbc.Row(
            dbc.Col(html.Div([
                html.H2(
                    [html.I(className="fas fa-chart-line me-2"), "Enterprise Sales Dashboard"],
                    className="fw-bold mb-1", style={"color": COLORS["primary"]},
                ),
                html.P("Real-time pipeline analytics across segments, regions, and products",
                       className="text-muted mb-0"),
            ], className="py-3")),
            className="mb-3",
        ),

        # KPI row
        dbc.Row(id="kpi-row", className="g-3 mb-4"),

        # Charts row 1: Pipeline by Stage + Lead Source
        dbc.Row([
            dbc.Col(make_card("Pipeline by Stage", dcc.Graph(id="fig-pipeline-stage", figure=empty_fig(), config={"displayModeBar": False})), md=6),
            dbc.Col(make_card("Opportunities by Lead Source", dcc.Graph(id="fig-lead-source", figure=empty_fig(), config={"displayModeBar": False})), md=6),
        ], className="g-3 mb-4"),

        # Charts row 2: Product Revenue + Segment Pie + Region Pie
        dbc.Row([
            dbc.Col(make_card("Revenue by Product", dcc.Graph(id="fig-prod-rev", figure=empty_fig(400), config={"displayModeBar": False})), md=6),
            dbc.Col(make_card("Revenue by Account Segment", dcc.Graph(id="fig-seg-pie", figure=empty_fig(360), config={"displayModeBar": False})), md=3),
            dbc.Col(make_card("Revenue by Region", dcc.Graph(id="fig-reg-pie", figure=empty_fig(360), config={"displayModeBar": False})), md=3),
        ], className="g-3 mb-4"),

        # Charts row 3: Opp trend + Avg deal size trend
        dbc.Row([
            dbc.Col(make_card("Opportunity Creation Trend", dcc.Graph(id="fig-opp-trend", figure=empty_fig(), config={"displayModeBar": False})), md=6),
            dbc.Col(make_card("Average Deal Size Trend", dcc.Graph(id="fig-avg-trend", figure=empty_fig(), config={"displayModeBar": False})), md=6),
        ], className="g-3 mb-4"),

        # --- Region Analysis ---
        html.Hr(className="my-4"),
        html.H4([html.I(className="fas fa-globe me-2"), "Region Analysis"],
                className="fw-bold mb-3", style={"color": COLORS["primary"]}),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Label("Region Filter", className="fw-semibold mb-2", style={"color": COLORS["primary"]}),
                dcc.Dropdown(id="region-filter", multi=True, placeholder="All Regions", className="mb-0"),
            ]), className="shadow-sm border-0 h-100"), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Region Total ACV", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H4(id="region-total-acv", className="fw-bold mb-0", style={"color": COLORS["primary"]}),
            ]), className="shadow-sm border-0 h-100"), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Region Won ACV", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H4(id="region-won-acv", className="fw-bold mb-0", style={"color": COLORS["success"]}),
            ]), className="shadow-sm border-0 h-100"), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.P("Accounts in Region", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H4(id="region-account-count", className="fw-bold mb-0", style={"color": COLORS["accent"]}),
            ]), className="shadow-sm border-0 h-100"), md=3),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(make_card("ACV by Segment", dcc.Graph(id="region-seg-bar", figure=empty_fig(), config={"displayModeBar": False})), md=6),
            dbc.Col(make_card("Pipeline by Stage", dcc.Graph(id="region-stage-bar", figure=empty_fig(), config={"displayModeBar": False})), md=6),
        ], className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(make_card("ACV Trend by Region", dcc.Graph(id="fig-reg-trend", figure=empty_fig(), config={"displayModeBar": False})), md=6),
            dbc.Col(make_card("ACV Trend by Industry", dcc.Graph(id="fig-seg-trend", figure=empty_fig(), config={"displayModeBar": False})), md=6),
        ], className="g-3 mb-4"),

        # --- Account Analysis ---
        html.Hr(className="my-4"),
        html.H4([html.I(className="fas fa-building me-2"), "Account Analysis"],
                className="fw-bold mb-3", style={"color": COLORS["primary"]}),

        dbc.Row(dbc.Col(make_card("Top Accounts by ACV", html.Div(id="top-accounts-table"))), className="g-3 mb-4"),

        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Label("Select Account", className="fw-semibold mb-2", style={"color": COLORS["primary"]}),
                dcc.Dropdown(id="account-filter", placeholder="All Accounts", className="mb-3"),
                html.P("Account Target Attainment", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
                html.H4(id="account-attainment", className="fw-bold mb-0", style={"color": COLORS["accent"]}),
            ]), className="shadow-sm border-0 h-100"), md=3),
            dbc.Col(make_card("Account ACV Over Time", dcc.Graph(id="account-acv-trend", figure=empty_fig(), config={"displayModeBar": False})), md=9),
        ], className="g-3 mb-4"),

        html.Hr(),
        html.P("Enterprise Sales Dashboard | Powered by Databricks",
               className="text-center text-muted py-3", style={"fontSize": "0.85rem"}),

        # --- Chat Interface ---
        dcc.Store(id="chat-messages-store", data=[]),
        dcc.Store(id="chat-typing", data=False),
        html.Div(id="chat-scroll-trigger", style={"display": "none"}),

        html.Div(
            id="chat-panel",
            style=CHAT_PANEL_STYLE,
            children=[
                html.Div(
                    [
                        html.Div([
                            html.I(className="fas fa-robot me-2"),
                            html.Span("Sales Analytics Assistant", className="fw-semibold"),
                        ], style={"display": "flex", "alignItems": "center"}),
                        html.Button(
                            html.I(className="fas fa-times"),
                            id="chat-close-btn",
                            className="btn btn-link text-white p-0",
                            style={"fontSize": "1.1rem", "lineHeight": "1", "textDecoration": "none"},
                        ),
                    ],
                    style={
                        "display": "flex", "justifyContent": "space-between",
                        "alignItems": "center", "padding": "14px 18px",
                        "backgroundColor": COLORS["primary"], "color": "white",
                        "fontSize": "0.95rem",
                    },
                ),
                html.Div(
                    id="chat-messages",
                    children=render_chat_messages([]),
                    style={"flex": "1", "overflowY": "auto", "padding": "16px"},
                ),
                html.Div(
                    [
                        dcc.Input(
                            id="chat-input", type="text",
                            placeholder="Ask about sales data…",
                            debounce=False, disabled=False,
                            style={
                                "flex": "1", "border": "1px solid #DEE2E8",
                                "borderRadius": "8px", "padding": "10px 14px",
                                "fontSize": "0.9rem", "outline": "none",
                            },
                            n_submit=0,
                        ),
                        html.Button(
                            html.I(className="fas fa-paper-plane"),
                            id="chat-send-btn", className="btn", disabled=False,
                            style={
                                "backgroundColor": COLORS["accent"], "color": "white",
                                "borderRadius": "8px", "padding": "8px 14px",
                                "marginLeft": "8px", "border": "none", "fontSize": "0.9rem",
                            },
                            n_clicks=0,
                        ),
                    ],
                    style={
                        "display": "flex", "alignItems": "center",
                        "padding": "12px 16px", "borderTop": "1px solid #E0E4EA",
                        "backgroundColor": "#FAFBFC",
                    },
                ),
            ],
        ),

        html.Button(
            html.I(className="fas fa-comments", style={"fontSize": "1.4rem"}),
            id="chat-toggle-btn", className="btn",
            style={
                "position": "fixed", "bottom": "30px", "right": "30px",
                "width": "56px", "height": "56px", "borderRadius": "50%",
                "backgroundColor": COLORS["accent"], "color": "white",
                "border": "none",
                "boxShadow": "0 4px 16px rgba(74, 144, 217, 0.4)",
                "zIndex": 1001, "display": "flex",
                "alignItems": "center", "justifyContent": "center",
                "cursor": "pointer",
            },
            n_clicks=0,
        ),
    ],
    fluid=True,
    style={"backgroundColor": COLORS["bg"], "minHeight": "100vh", "padding": "20px 30px"},
)


# ---------------------------------------------------------------------------
# Startup callback — loads data on first page visit and populates all static
# charts + filter options.  Triggered by dcc.Interval (fires once).
# ---------------------------------------------------------------------------

@app.callback(
    [
        Output("data-loaded", "data"),
        Output("kpi-row", "children"),
        Output("fig-pipeline-stage", "figure"),
        Output("fig-lead-source", "figure"),
        Output("fig-prod-rev", "figure"),
        Output("fig-seg-pie", "figure"),
        Output("fig-reg-pie", "figure"),
        Output("fig-opp-trend", "figure"),
        Output("fig-avg-trend", "figure"),
        Output("fig-reg-trend", "figure"),
        Output("fig-seg-trend", "figure"),
        Output("region-filter", "options"),
        Output("account-filter", "options"),
    ],
    Input("startup-trigger", "n_intervals"),
    prevent_initial_call=False,
)
def load_data_and_build_static(_n):
    logger.info("Startup callback triggered — loading data from warehouse")

    try:
        opps = _prep_df(backend.get_opportunities(), numeric_cols=["acv", "is_won"])
        acct_opps = _prep_df(backend.get_account_opportunities(), numeric_cols=["acv"])
        prod_rev = _prep_df(backend.get_product_revenue(), date_col=None, numeric_cols=["line_acv"])
        acct_rep = _prep_df(backend.get_account_rep_summary(), numeric_cols=["acv", "annual_acv_target"])
    except Exception as exc:
        logger.error("Failed to load data: %s", exc, exc_info=True)
        error_card = dbc.Col(dbc.Alert(
            f"Unable to load data — the SQL warehouse may still be starting. Refresh in 30 s.  Error: {exc}",
            color="warning",
        ), width=12)
        return (
            False, [error_card],
            empty_fig(), empty_fig(), empty_fig(400), empty_fig(360), empty_fig(360),
            empty_fig(), empty_fig(), empty_fig(), empty_fig(),
            [], [],
        )

    logger.info("All datasets loaded successfully")

    # --- KPI cards ---
    total_pipeline = opps["acv"].sum()
    won_acv_val = opps.loc[opps["stage"] == "Closed/Won", "acv"].sum()
    closed = opps[opps["stage"].isin(["Closed/Won", "Closed/Lost"])]
    win_rate = closed["is_won"].mean() if len(closed) > 0 else 0
    avg_deal = opps["acv"].mean()

    kpi_children = [
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Total Pipeline ACV", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
            html.H3(fmt_currency(total_pipeline), className="fw-bold mb-0", style={"color": COLORS["primary"]}),
        ]), className="shadow-sm border-0 h-100"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Won Deals ACV", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
            html.H3(fmt_currency(won_acv_val), className="fw-bold mb-0", style={"color": COLORS["success"]}),
        ]), className="shadow-sm border-0 h-100"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Win Rate", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
            html.H3(fmt_pct(win_rate), className="fw-bold mb-0", style={"color": COLORS["accent"]}),
        ]), className="shadow-sm border-0 h-100"), md=3),
        dbc.Col(dbc.Card(dbc.CardBody([
            html.P("Average Deal Size", className="text-muted mb-1", style={"fontSize": "0.85rem"}),
            html.H3(fmt_currency(avg_deal), className="fw-bold mb-0", style={"color": COLORS["warning"]}),
        ]), className="shadow-sm border-0 h-100"), md=3),
    ]

    # --- Pipeline by Stage ---
    stage_acv = opps.groupby("stage", as_index=False)["acv"].sum()
    stage_acv["stage"] = pd.Categorical(stage_acv["stage"], categories=STAGE_ORDER, ordered=True)
    stage_acv = stage_acv.sort_values("stage")
    fig_ps = px.bar(stage_acv, x="stage", y="acv", color="stage", color_discrete_map=STAGE_COLORS, text_auto=".2s")
    fig_ps.update_traces(textposition="outside")
    chart_layout(fig_ps)
    fig_ps.update_layout(showlegend=False)

    # --- Opps by Lead Source ---
    lead_counts = opps.groupby("lead_source", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False)
    fig_ls = px.bar(lead_counts, x="lead_source", y="count", color="lead_source", text_auto=True, color_discrete_sequence=px.colors.qualitative.Set2)
    fig_ls.update_traces(textposition="outside")
    chart_layout(fig_ls)
    fig_ls.update_layout(showlegend=False)

    # --- Revenue by Product ---
    prod_acv = prod_rev.groupby("product_name", as_index=False)["line_acv"].sum().sort_values("line_acv", ascending=True)
    fig_pr = px.bar(prod_acv, y="product_name", x="line_acv", orientation="h", color="product_name", text_auto=".2s", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_pr.update_traces(textposition="outside")
    chart_layout(fig_pr, height=400)
    fig_pr.update_layout(showlegend=False, yaxis_title=None)

    # --- Segment pie ---
    seg_acv = acct_opps.groupby("segment", as_index=False)["acv"].sum()
    fig_sp = px.pie(seg_acv, values="acv", names="segment", color_discrete_sequence=px.colors.qualitative.Set2, hole=0.4)
    fig_sp.update_traces(textinfo="label+percent", textposition="outside")
    chart_layout(fig_sp, height=360)

    # --- Region pie ---
    reg_acv = acct_opps.groupby("region", as_index=False)["acv"].sum()
    fig_rp = px.pie(reg_acv, values="acv", names="region", color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.4)
    fig_rp.update_traces(textinfo="label+percent", textposition="outside")
    chart_layout(fig_rp, height=360)

    # --- Opp creation trend ---
    opp_trend = opps.groupby("created_month", as_index=False).size().rename(columns={"size": "count"}).sort_values("created_month")
    fig_ot = px.line(opp_trend, x="created_month", y="count", markers=True, color_discrete_sequence=[COLORS["accent"]])
    chart_layout(fig_ot)

    # --- Avg deal size trend ---
    avg_trend = opps.groupby("created_month", as_index=False)["acv"].mean().sort_values("created_month")
    fig_at = px.line(avg_trend, x="created_month", y="acv", markers=True, color_discrete_sequence=[COLORS["warning"]])
    chart_layout(fig_at)

    # --- ACV trend by region ---
    reg_trend = acct_opps.groupby(["created_month", "region"], as_index=False)["acv"].sum().sort_values("created_month")
    fig_rt = px.line(reg_trend, x="created_month", y="acv", color="region", markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
    chart_layout(fig_rt)

    # --- ACV trend by segment ---
    ind_trend = acct_opps.groupby(["created_month", "industry"], as_index=False)["acv"].sum().sort_values("created_month")
    fig_st = px.line(ind_trend, x="created_month", y="acv", color="industry", markers=True, color_discrete_sequence=px.colors.qualitative.Pastel)
    chart_layout(fig_st)

    # --- Filter options ---
    region_opts = [{"label": r, "value": r} for r in sorted(acct_opps["region"].dropna().unique())]
    account_opts = [{"label": a, "value": a} for a in sorted(acct_rep["account_name"].dropna().unique())]

    return (
        True, kpi_children,
        fig_ps, fig_ls, fig_pr, fig_sp, fig_rp,
        fig_ot, fig_at, fig_rt, fig_st,
        region_opts, account_opts,
    )


# ---------------------------------------------------------------------------
# Region analysis callback
# ---------------------------------------------------------------------------

@app.callback(
    [
        Output("region-total-acv", "children"),
        Output("region-won-acv", "children"),
        Output("region-account-count", "children"),
        Output("region-seg-bar", "figure"),
        Output("region-stage-bar", "figure"),
    ],
    [Input("region-filter", "value"), Input("data-loaded", "data")],
)
def update_region_section(selected_regions, loaded):
    if not loaded:
        return ("—", "—", "—", empty_fig(), empty_fig())

    df = _prep_df(backend.get_account_opportunities(), numeric_cols=["acv"])
    if selected_regions:
        df = df[df["region"].isin(selected_regions)]

    total = fmt_currency(df["acv"].sum())
    won = fmt_currency(df.loc[df["stage"] == "Closed/Won", "acv"].sum())
    n_accounts = fmt_number(df["account_id"].nunique())

    seg_data = df.groupby("segment", as_index=False)["acv"].sum().sort_values("acv", ascending=False)
    fig_seg = px.bar(seg_data, x="segment", y="acv", color="segment", text_auto=".2s", color_discrete_sequence=px.colors.qualitative.Set2)
    fig_seg.update_traces(textposition="outside")
    chart_layout(fig_seg)
    fig_seg.update_layout(showlegend=False)

    stage_data = df.groupby("stage", as_index=False)["acv"].sum()
    stage_data["stage"] = pd.Categorical(stage_data["stage"], categories=STAGE_ORDER, ordered=True)
    stage_data = stage_data.sort_values("stage")
    fig_stage = px.bar(stage_data, x="stage", y="acv", color="stage", text_auto=".2s", color_discrete_map=STAGE_COLORS)
    fig_stage.update_traces(textposition="outside")
    chart_layout(fig_stage)
    fig_stage.update_layout(showlegend=False)

    return total, won, n_accounts, fig_seg, fig_stage


# ---------------------------------------------------------------------------
# Account analysis callback
# ---------------------------------------------------------------------------

@app.callback(
    [Output("account-attainment", "children"), Output("account-acv-trend", "figure")],
    [Input("account-filter", "value"), Input("data-loaded", "data")],
)
def update_account_section(selected_account, loaded):
    if not loaded:
        return ("—", empty_fig())

    tgt_df = _prep_df(
        backend.get_account_target_summary(),
        date_col=None,
        numeric_cols=["attainment_pct", "won_acv", "annual_acv_target"],
    )
    trend_df = _prep_df(backend.get_account_opportunities(), numeric_cols=["acv"])

    if selected_account:
        tgt_df = tgt_df[tgt_df["account_name"] == selected_account]
        trend_df = trend_df[trend_df["account_name"] == selected_account]

    attainment = tgt_df["attainment_pct"].mean() if len(tgt_df) > 0 else 0
    monthly = trend_df.groupby("created_month", as_index=False)["acv"].sum().sort_values("created_month")
    fig = px.line(monthly, x="created_month", y="acv", markers=True, color_discrete_sequence=[COLORS["accent"]])
    chart_layout(fig)

    return f"{attainment:.1f}%", fig


# ---------------------------------------------------------------------------
# Top accounts table callback
# ---------------------------------------------------------------------------

@app.callback(
    Output("top-accounts-table", "children"),
    [Input("region-filter", "value"), Input("data-loaded", "data")],
)
def update_top_accounts(selected_regions, loaded):
    if not loaded:
        return html.P("Loading...", className="text-muted")

    df = _prep_df(backend.get_account_rep_summary(), numeric_cols=["acv", "annual_acv_target"])
    if selected_regions:
        df = df[df["region"].isin(selected_regions)]

    won = df[df["stage"] == "Closed/Won"]
    table_data = (
        won.groupby(["account_name", "segment", "region", "rep_name", "team"], as_index=False)
        .agg(total_acv=("acv", "sum"), target_acv=("annual_acv_target", "max"), opportunities=("acv", "count"))
        .sort_values("total_acv", ascending=False)
        .head(25)
    )
    table_data["total_acv"] = table_data["total_acv"].apply(lambda v: fmt_currency(v, compact=False))
    table_data["target_acv"] = table_data["target_acv"].apply(lambda v: fmt_currency(v, compact=False))

    return dash_table.DataTable(
        data=table_data.to_dict("records"),
        columns=[
            {"name": "Account", "id": "account_name"},
            {"name": "Segment", "id": "segment"},
            {"name": "Region", "id": "region"},
            {"name": "Sales Rep", "id": "rep_name"},
            {"name": "Team", "id": "team"},
            {"name": "Total ACV", "id": "total_acv"},
            {"name": "Target ACV", "id": "target_acv"},
            {"name": "Opportunities", "id": "opportunities"},
        ],
        page_size=25,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": COLORS["primary"],
            "color": "white",
            "fontWeight": "600",
            "fontSize": "0.85rem",
            "textAlign": "left",
            "padding": "10px 12px",
        },
        style_cell={
            "fontSize": "0.85rem",
            "padding": "8px 12px",
            "textAlign": "left",
            "border": "none",
            "borderBottom": "1px solid #EEF0F4",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#F9FAFB"},
        ],
    )


# ---------------------------------------------------------------------------
# Chat callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("chat-panel", "style"),
    [Input("chat-toggle-btn", "n_clicks"), Input("chat-close-btn", "n_clicks")],
    State("chat-panel", "style"),
    prevent_initial_call=True,
)
def toggle_chat(toggle_clicks, close_clicks, current_style):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    style = dict(current_style) if current_style else dict(CHAT_PANEL_STYLE)
    if triggered_id == "chat-close-btn":
        style["display"] = "none"
    else:
        style["display"] = "none" if style.get("display") == "flex" else "flex"
    return style


app.clientside_callback(
    """
    function(nClicks, nSubmit, inputValue, messages) {
        if (!inputValue || !inputValue.trim()) {
            return [
                window.dash_clientside.no_update,
                window.dash_clientside.no_update,
                window.dash_clientside.no_update
            ];
        }
        var updated = (messages || []).slice();
        updated.push({role: 'user', content: inputValue.trim()});
        return [updated, true, ''];
    }
    """,
    [Output("chat-messages-store", "data"),
     Output("chat-typing", "data"),
     Output("chat-input", "value")],
    [Input("chat-send-btn", "n_clicks"), Input("chat-input", "n_submit")],
    [State("chat-input", "value"), State("chat-messages-store", "data")],
    prevent_initial_call=True,
)


@app.callback(
    [Output("chat-messages-store", "data", allow_duplicate=True),
     Output("chat-typing", "data", allow_duplicate=True)],
    Input("chat-typing", "data"),
    State("chat-messages-store", "data"),
    prevent_initial_call=True,
)
def call_agent(typing, messages):
    if not typing:
        raise dash.exceptions.PreventUpdate
    messages = list(messages or [])
    try:
        response_text = backend.chat_with_agent(messages)
        messages.append({"role": "assistant", "content": response_text})
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        messages.append({"role": "assistant", "content": f"Sorry, I encountered an error: {exc}"})
    return messages, False


@app.callback(
    Output("chat-messages", "children"),
    [Input("chat-messages-store", "data"), Input("chat-typing", "data")],
)
def render_messages_cb(messages, typing):
    return render_chat_messages(messages or [], typing=typing)


@app.callback(
    [Output("chat-send-btn", "disabled"), Output("chat-input", "disabled")],
    Input("chat-typing", "data"),
)
def toggle_chat_input(typing):
    return typing, typing


app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var el = document.getElementById('chat-messages');
            if (el) el.scrollTop = el.scrollHeight;
        }, 100);
        return '';
    }
    """,
    Output("chat-scroll-trigger", "children"),
    Input("chat-messages", "children"),
    prevent_initial_call=True,
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
