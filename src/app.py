"""
app.py
------
Customer Churn & Revenue Risk Analytics Dashboard.

Reads the scored dataset (output of model.py) and the SHAP feature
importance table, and serves an interactive Dash app with:
    - KPI summary row (customers, churn rate, revenue at risk, avg CLV)
    - Segment distribution (pie) + revenue at risk by segment (bar)
    - Churn driver analysis (SHAP bar) + tenure/charges box plots
    - Deep dive: scatter, histogram, churn rate by contract/internet
    - Written, data-driven recommendations

Run locally:
    python src/app.py
Then open http://127.0.0.1:8050

For Render deployment, gunicorn points at app:server (see Dockerfile).
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "scored_customers.csv")
IMPORTANCE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_importance.csv")

df = pd.read_csv(DATA_PATH)
importance_df = pd.read_csv(IMPORTANCE_PATH)

SEGMENT_ORDER = ["Champions", "Loyal High-Value", "At Risk", "Price-Sensitive / New", "Low-Value / Disengaged"]
SEGMENT_COLORS = {
    "Champions": "#0e9594",
    "Loyal High-Value": "#3d7ea6",
    "At Risk": "#f2765a",
    "Price-Sensitive / New": "#f4a259",
    "Low-Value / Disengaged": "#b6bbc4",
}
CHURN_COLORS = {"Retained": "#0e9594", "Churned": "#f2765a"}

PLOT_FONT = dict(family="Inter, sans-serif", size=12, color="#12213a")
PLOT_LAYOUT = dict(
    font=PLOT_FONT,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=20, t=10, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

app = Dash(__name__, title="Churn & Revenue Risk Analytics")
server = app.server  # exposed for gunicorn


def kpi_card(label, value, sub="", risk=False):
    return html.Div(
        className=f"kpi-card{' risk' if risk else ''}",
        children=[
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value"),
            html.Div(sub, className="kpi-sub") if sub else None,
        ],
    )


def chart_card(title, subtitle, graph):
    return html.Div(
        className="chart-card",
        children=[
            html.Div(title, className="chart-title"),
            html.Div(subtitle, className="chart-subtitle"),
            graph,
        ],
    )


def filter_data(segment, contract):
    d = df.copy()
    if segment and segment != "All":
        d = d[d["Segment"] == segment]
    if contract and contract != "All":
        d = d[d["Contract"] == contract]
    return d


# ---------------------------------------------------------------- layout --
app.layout = html.Div([
    html.Div(className="app-header", children=[
        html.H1("Customer Churn & Revenue Risk Analytics"),
        html.P("$ telco_churn_pipeline --dataset=7043_customers --model=xgboost --status=live"),
    ]),

    html.Div(className="filter-bar", children=[
        html.Div([
            html.Div("Segment", className="filter-label"),
            dcc.Dropdown(
                id="segment-filter",
                options=[{"label": "All Segments", "value": "All"}] +
                        [{"label": s, "value": s} for s in SEGMENT_ORDER],
                value="All", clearable=False, style={"width": "220px"},
            ),
        ]),
        html.Div([
            html.Div("Contract Type", className="filter-label"),
            dcc.Dropdown(
                id="contract-filter",
                options=[{"label": "All Contracts", "value": "All"}] +
                        [{"label": c, "value": c} for c in sorted(df["Contract"].unique())],
                value="All", clearable=False, style={"width": "220px"},
            ),
        ]),
    ]),

    html.Div(id="kpi-row", className="kpi-row"),

    dcc.Tabs(id="tabs", value="overview", className="custom-tabs", children=[
        dcc.Tab(label="Overview", value="overview"),
        dcc.Tab(label="Churn Drivers", value="drivers"),
        dcc.Tab(label="Deep Dive", value="deepdive"),
        dcc.Tab(label="Recommendations", value="recs"),
    ]),

    html.Div(id="tab-content", className="content-area"),
])


# --------------------------------------------------------------- callbacks --
@app.callback(Output("kpi-row", "children"),
              Input("segment-filter", "value"), Input("contract-filter", "value"))
def update_kpis(segment, contract):
    d = filter_data(segment, contract)
    total_customers = len(d)
    churn_rate = d["ChurnFlag"].mean() if total_customers else 0
    revenue_at_risk = d["RevenueAtRisk"].sum()
    avg_clv = d["ProjectedCLV"].mean() if total_customers else 0

    return [
        kpi_card("Customers", f"{total_customers:,}"),
        kpi_card("Churn Rate", f"{churn_rate:.1%}", sub=f"{int(d['ChurnFlag'].sum()):,} churned historically"),
        kpi_card("Annual Revenue at Risk", f"${revenue_at_risk:,.0f}", risk=True,
                 sub="churn probability × 12mo charges"),
        kpi_card("Avg Projected CLV", f"${avg_clv:,.0f}", sub="24-month horizon"),
    ]


@app.callback(Output("tab-content", "children"),
              Input("tabs", "value"), Input("segment-filter", "value"), Input("contract-filter", "value"))
def render_tab(tab, segment, contract):
    d = filter_data(segment, contract)

    if tab == "overview":
        return render_overview(d)
    elif tab == "drivers":
        return render_drivers(d)
    elif tab == "deepdive":
        return render_deepdive(d)
    elif tab == "recs":
        return render_recommendations(d)


def render_overview(d):
    seg_counts = d["Segment"].value_counts().reindex(SEGMENT_ORDER).dropna()
    pie = go.Figure(data=[go.Pie(
        labels=seg_counts.index, values=seg_counts.values, hole=0.45,
        marker=dict(colors=[SEGMENT_COLORS[s] for s in seg_counts.index]),
        textinfo="label+percent", textfont=dict(size=11),
    )])
    pie.update_layout(**PLOT_LAYOUT, showlegend=False, height=340)

    rev_by_seg = d.groupby("Segment")["RevenueAtRisk"].sum().reindex(SEGMENT_ORDER).dropna().sort_values()
    bar = go.Figure(data=[go.Bar(
        x=rev_by_seg.values, y=rev_by_seg.index, orientation="h",
        marker_color=[SEGMENT_COLORS[s] for s in rev_by_seg.index],
        text=[f"${v:,.0f}" for v in rev_by_seg.values], textposition="outside",
    )])
    bar.update_layout(**PLOT_LAYOUT, height=340, xaxis_title="Annual Revenue at Risk ($)")

    churn_by_seg = d.groupby("Segment")["ChurnFlag"].mean().reindex(SEGMENT_ORDER).dropna()
    churn_bar = go.Figure(data=[go.Bar(
        x=churn_by_seg.index, y=churn_by_seg.values,
        marker_color=[SEGMENT_COLORS[s] for s in churn_by_seg.index],
        text=[f"{v:.1%}" for v in churn_by_seg.values], textposition="outside",
    )])
    churn_bar.update_layout(**PLOT_LAYOUT, height=320, yaxis_title="Historical Churn Rate", yaxis_tickformat=".0%")

    return html.Div([
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
            chart_card("Customer Segments", "CLV-adapted RFM segmentation (Recency proxy: tenure)",
                       dcc.Graph(figure=pie, config={"displayModeBar": False})),
            chart_card("Revenue at Risk by Segment", "Sum of (churn probability × annual charges)",
                       dcc.Graph(figure=bar, config={"displayModeBar": False})),
        ]),
        chart_card("Historical Churn Rate by Segment", "Champions churn least; new/price-sensitive customers churn most",
                   dcc.Graph(figure=churn_bar, config={"displayModeBar": False})),
    ])


def render_drivers(d):
    top_features = importance_df.head(10).sort_values("mean_abs_shap")
    shap_bar = go.Figure(data=[go.Bar(
        x=top_features["mean_abs_shap"], y=top_features["feature"], orientation="h",
        marker_color="#0e9594",
    )])
    shap_bar.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="Mean |SHAP value| (impact on churn probability)")

    d2 = d.copy()
    d2["ChurnLabel"] = d2["ChurnFlag"].map({0: "Retained", 1: "Churned"})

    box_tenure = px.box(d2, x="ChurnLabel", y="tenure", color="ChurnLabel",
                         color_discrete_map=CHURN_COLORS, points=False)
    box_tenure.update_layout(**PLOT_LAYOUT, height=340, showlegend=False, yaxis_title="Tenure (months)", xaxis_title="")

    box_charges = px.box(d2, x="ChurnLabel", y="MonthlyCharges", color="ChurnLabel",
                          color_discrete_map=CHURN_COLORS, points=False)
    box_charges.update_layout(**PLOT_LAYOUT, height=340, showlegend=False, yaxis_title="Monthly Charges ($)", xaxis_title="")

    return html.Div([
        chart_card("Top Churn Drivers (SHAP Feature Importance)",
                   "XGBoost model, TreeExplainer — larger bar = stronger influence on churn prediction",
                   dcc.Graph(figure=shap_bar, config={"displayModeBar": False})),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
            chart_card("Tenure: Retained vs. Churned", "Churned customers are heavily concentrated at low tenure",
                       dcc.Graph(figure=box_tenure, config={"displayModeBar": False})),
            chart_card("Monthly Charges: Retained vs. Churned", "Churned customers pay more per month on average",
                       dcc.Graph(figure=box_charges, config={"displayModeBar": False})),
        ]),
    ])


def render_deepdive(d):
    d2 = d.copy()
    d2["ChurnLabel"] = d2["ChurnFlag"].map({0: "Retained", 1: "Churned"})

    scatter = px.scatter(
        d2, x="tenure", y="MonthlyCharges", color="ChurnProbability",
        color_continuous_scale=["#0e9594", "#f4a259", "#f2765a"],
        opacity=0.6, hover_data=["Segment", "Contract"],
    )
    scatter.update_layout(**PLOT_LAYOUT, height=380, xaxis_title="Tenure (months)", yaxis_title="Monthly Charges ($)",
                           coloraxis_colorbar=dict(title="Churn<br>Prob."))

    hist = px.histogram(d2, x="tenure", color="ChurnLabel", barmode="overlay", nbins=30,
                         color_discrete_map=CHURN_COLORS, opacity=0.75)
    hist.update_layout(**PLOT_LAYOUT, height=340, xaxis_title="Tenure (months)", yaxis_title="Number of Customers")

    churn_by_contract = d2.groupby("Contract")["ChurnFlag"].mean().sort_values()
    contract_bar = go.Figure(data=[go.Bar(
        x=churn_by_contract.values, y=churn_by_contract.index, orientation="h",
        marker_color="#3d7ea6", text=[f"{v:.1%}" for v in churn_by_contract.values], textposition="outside",
    )])
    contract_bar.update_layout(**PLOT_LAYOUT, height=280, xaxis_title="Churn Rate", xaxis_tickformat=".0%")

    churn_by_internet = d2.groupby("InternetService")["ChurnFlag"].mean().sort_values()
    internet_bar = go.Figure(data=[go.Bar(
        x=churn_by_internet.index, y=churn_by_internet.values,
        marker_color="#f4a259", text=[f"{v:.1%}" for v in churn_by_internet.values], textposition="outside",
    )])
    internet_bar.update_layout(**PLOT_LAYOUT, height=280, yaxis_title="Churn Rate", yaxis_tickformat=".0%")

    return html.Div([
        chart_card("Tenure vs. Monthly Charges", "Colored by predicted churn probability — risk clusters top-left",
                   dcc.Graph(figure=scatter, config={"displayModeBar": False})),
        chart_card("Tenure Distribution: Retained vs. Churned", "Sharp churn concentration in the first ~10 months",
                   dcc.Graph(figure=hist, config={"displayModeBar": False})),
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}, children=[
            chart_card("Churn Rate by Contract Type", "Month-to-month contracts are the dominant risk factor",
                       dcc.Graph(figure=contract_bar, config={"displayModeBar": False})),
            chart_card("Churn Rate by Internet Service", "Fiber optic customers churn at a notably higher rate",
                       dcc.Graph(figure=internet_bar, config={"displayModeBar": False})),
        ]),
    ])


def render_recommendations(d):
    total_risk = d["RevenueAtRisk"].sum()
    at_risk_seg = d[d["Segment"] == "At Risk"]
    price_sens_seg = d[d["Segment"] == "Price-Sensitive / New"]
    mtm = d[d["Contract"] == "Month-to-month"]
    mtm_share_of_risk = (mtm["RevenueAtRisk"].sum() / total_risk * 100) if total_risk > 0 else 0

    return html.Div([
        html.Div(className="recommendation-box", children=[
            html.H4("1. Prioritize month-to-month, low-tenure customers for retention outreach"),
            html.P(f"Month-to-month contracts account for roughly {mtm_share_of_risk:.0f}% of total annualized "
                   f"revenue at risk (${mtm['RevenueAtRisk'].sum():,.0f} of ${total_risk:,.0f}). Customers in "
                   f"their first 10 months of tenure show the sharpest concentration of churn. A targeted "
                   f"retention offer (e.g., a discounted 1-year contract upgrade) aimed specifically at "
                   f"month-to-month customers under 12 months tenure would address the single largest "
                   f"identifiable risk pool."),
        ]),
        html.Div(className="recommendation-box", children=[
            html.H4("2. Bundle security & support add-ons as a retention lever"),
            html.P("SHAP analysis shows the absence of Online Security and Tech Support are among the "
                   "strongest churn drivers, independent of contract type. Offering these as a free "
                   "3-month trial to At Risk and Price-Sensitive segments is a low-cost intervention "
                   "that directly targets a top model-identified driver rather than a generic discount."),
        ]),
        html.Div(className="recommendation-box", children=[
            html.H4("3. Investigate fiber optic service experience"),
            html.P("Fiber optic customers churn at a meaningfully higher rate than DSL or no-internet-service "
                   "customers despite typically paying more, which suggests a service quality or pricing "
                   "perception issue rather than a pure affordability issue. This warrants a qualitative "
                   "follow-up (support ticket review or a short satisfaction survey) rather than a discount-only fix."),
        ]),
        html.Div(className="recommendation-box", children=[
            html.H4(f"4. Segment-level focus: 'At Risk' and 'Price-Sensitive / New' cohorts"),
            html.P(f"These two segments combine for {len(at_risk_seg) + len(price_sens_seg):,} customers and "
                   f"show the highest churn rates of any segment. Because they are also lower in projected CLV "
                   f"than Champions or Loyal High-Value customers, retention spend per customer here should stay "
                   f"modest and automated (email/in-app offers) rather than high-touch (account manager calls), "
                   f"which should be reserved for high-CLV segments."),
        ]),
    ])


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
