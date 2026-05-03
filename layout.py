"""
App layout — builds and returns the full Dash component tree.
Call build_layout() after the app is created.
"""

from dash import dcc, html, dash_table

from config import (
    DATA_PATH, DEFAULT_KPI, DEFAULT_KPI_2, DEFAULT_KPI_3,
    DEFAULT_X, DEFAULT_Y, DEFAULT_COLOR,
    P0_COLOR, P1_COLOR,
    SECTION_STYLE, APP_STYLE, TABLE_PAGE_SIZE,
)
from data_utils import categorical_options
from precompute import (
    DF, LOAD_ERROR,
    NUMERIC_COLS, KPI_OPTIONS, DATE_MIN, DATE_MAX,
)


def build_layout():
    return html.Div([
        html.H1("RAN KPI Congestion Dashboard",
                style={"margin": "0 0 4px 0", "color": "#0f172a"}),
        html.P("Explore load, throughput, accessibility and congestion patterns "
               "by region, site and cell.",
               style={"margin": "0 0 18px 0", "color": "#475569"}),
        html.Div(
            LOAD_ERROR if LOAD_ERROR else f"Loaded {len(DF):,} rows from {DATA_PATH}",
            style={
                "marginBottom": "16px",
                "padding": "12px 14px",
                "borderRadius": "12px",
                "background": "#fee2e2" if LOAD_ERROR else "#e0f2fe",
                "color": "#111827",
                "border": "1px solid #fecaca" if LOAD_ERROR else "1px solid #bae6fd",
            },
        ),

        # ---- Top filter row ----
        html.Div([
            html.Div([
                html.Label("Region"),
                dcc.Dropdown(
                    id="region-filter",
                    options=(categorical_options(DF["Region"])
                             if not DF.empty and "Region" in DF.columns else []),
                    multi=True,
                ),
            ], style={"flex": 1}),
            html.Div([html.Label("Radio Site"), dcc.Dropdown(id="site-filter", multi=True)],
                     style={"flex": 1}),
            html.Div([html.Label("Cell Name"),  dcc.Dropdown(id="cell-filter", multi=True)],
                     style={"flex": 1.3}),
            html.Div([
                html.Label("FDD / TDD"),
                dcc.Dropdown(
                    id="mode-filter",
                    options=(categorical_options(DF["Cell FDD TDD Indication"])
                             if not DF.empty and "Cell FDD TDD Indication" in DF.columns else []),
                    multi=True,
                ),
            ], style={"flex": 0.7}),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px", "flexWrap": "wrap"}),

        html.Div([
            html.Div([
                html.Label("Date Range"),
                dcc.DatePickerRange(
                    id="date-filter",
                    start_date=DATE_MIN, end_date=DATE_MAX,
                    min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX,
                    display_format="YYYY-MM-DD",
                ),
            ], style={"flex": 1}),
            html.Div([
                html.Label("Main KPI 1"),
                dcc.Dropdown(id="main-kpi", options=KPI_OPTIONS,
                             value=(DEFAULT_KPI if DEFAULT_KPI in NUMERIC_COLS
                                    else (NUMERIC_COLS[0] if NUMERIC_COLS else None))),
            ], style={"flex": 1}),
            html.Div([
                html.Label("Main KPI 2"),
                dcc.Dropdown(id="main-kpi-2", options=KPI_OPTIONS,
                             value=(DEFAULT_KPI_2 if DEFAULT_KPI_2 in NUMERIC_COLS
                                    else (NUMERIC_COLS[1] if len(NUMERIC_COLS) > 1 else None))),
            ], style={"flex": 1}),
            html.Div([
                html.Label("Main KPI 3"),
                dcc.Dropdown(id="main-kpi-3", options=KPI_OPTIONS,
                             value=(DEFAULT_KPI_3 if DEFAULT_KPI_3 in NUMERIC_COLS
                                    else (NUMERIC_COLS[2] if len(NUMERIC_COLS) > 2 else None))),
            ], style={"flex": 1}),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "18px", "flexWrap": "wrap"}),

        html.Div(id="summary-cards", style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
            "gap": "12px", "marginBottom": "16px",
        }),

        dcc.Tabs([
            _tab_kpi_analytics(),
            _tab_congestion_analytics(),
            _tab_geometry_map(),
        ]),

    ], style=APP_STYLE)


# ---- Private tab builders ----

def _tab_kpi_analytics():
    return dcc.Tab(label="General KPI Analytics", children=[
        html.Div([
            html.Div([dcc.Graph(id="timeseries-chart")],   style=SECTION_STYLE),
            html.Div([dcc.Graph(id="timeseries-chart-2")], style=SECTION_STYLE),
            html.Div([dcc.Graph(id="timeseries-chart-3")], style=SECTION_STYLE),
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Histogram KPI"),
                        dcc.Dropdown(id="hist-kpi", options=KPI_OPTIONS,
                                     value=(DEFAULT_KPI if DEFAULT_KPI in NUMERIC_COLS
                                            else (NUMERIC_COLS[0] if NUMERIC_COLS else None))),
                    ], style={"marginBottom": "10px"}),
                    dcc.Graph(id="histogram-chart"),
                ], style={"flex": 1}),
                html.Div([
                    html.Div([
                        html.Label("Top Cells KPI"),
                        dcc.Dropdown(id="top-kpi", options=KPI_OPTIONS,
                                     value=(DEFAULT_KPI if DEFAULT_KPI in NUMERIC_COLS
                                            else (NUMERIC_COLS[0] if NUMERIC_COLS else None))),
                    ], style={"marginBottom": "10px"}),
                    dcc.Graph(id="top-cells-chart"),
                ], style={"flex": 1}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Scatter X"),
                        dcc.Dropdown(id="scatter-x", options=KPI_OPTIONS,
                                     value=(DEFAULT_X if DEFAULT_X in NUMERIC_COLS
                                            else (NUMERIC_COLS[0] if NUMERIC_COLS else None))),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Label("Scatter Y"),
                        dcc.Dropdown(id="scatter-y", options=KPI_OPTIONS,
                                     value=(DEFAULT_Y if DEFAULT_Y in NUMERIC_COLS
                                            else (NUMERIC_COLS[1] if len(NUMERIC_COLS) > 1 else None))),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Label("Color"),
                        dcc.Dropdown(id="scatter-color",
                                     options=[{"label": "None", "value": "__none__"}] + KPI_OPTIONS,
                                     value=(DEFAULT_COLOR if DEFAULT_COLOR in NUMERIC_COLS
                                            else "__none__")),
                    ], style={"marginBottom": "8px"}),
                    dcc.Graph(id="scatter-chart"),
                ], style={"flex": 1}),
                html.Div([dcc.Graph(id="correlation-chart")],
                         style={**SECTION_STYLE, "flex": 1}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
            html.Div([
                html.H3("Filtered Data Table", style={"marginTop": 0}),
                dash_table.DataTable(
                    id="detail-table", page_size=TABLE_PAGE_SIZE,
                    sort_action="native", filter_action="native", page_action="native",
                    style_table={"overflowX": "auto", "maxHeight": "650px", "overflowY": "auto"},
                    style_cell={"textAlign": "left", "minWidth": "120px",
                                "width": "120px", "maxWidth": "220px", "padding": "8px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#f1f5f9"},
                ),
            ], style=SECTION_STYLE),
        ])
    ])


def _tab_congestion_analytics():
    return dcc.Tab(label="Congestion Analytics", children=[
        html.Div([
            html.Div([
                html.Span("Congestion tiers: ",
                          style={"fontSize": "12px", "color": "#64748b", "fontWeight": 600}),
                html.Span("● P0", style={"fontSize": "13px", "fontWeight": 700,
                                          "color": P0_COLOR, "marginLeft": "10px"}),
                html.Span(" — most severe", style={"fontSize": "12px", "color": "#475569"}),
                html.Span("    ● P1", style={"fontSize": "13px", "fontWeight": 700,
                                              "color": P1_COLOR, "marginLeft": "16px"}),
                html.Span(" — moderate", style={"fontSize": "12px", "color": "#475569"}),
            ], style={
                "padding": "10px 16px", "background": "#fffbeb",
                "border": "1px solid #fde68a", "borderRadius": "10px",
                "marginBottom": "14px",
            }),
            html.Div(id="congestion-cards", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                "gap": "12px", "marginBottom": "16px",
            }),
            html.Div([dcc.Graph(id="congestion-timeseries-chart")], style=SECTION_STYLE),
            html.Div([dcc.Graph(id="congestion-level-chart")],       style=SECTION_STYLE),
            html.Div([
                html.Div([dcc.Graph(id="top-congested-cells-chart")], style={"flex": 1}),
                html.Div([dcc.Graph(id="busy-hour-chart")],           style={"flex": 1}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                      "marginBottom": "16px"}),
            html.Div([dcc.Graph(id="congested-scatter-chart")], style=SECTION_STYLE),
            html.Div([
                html.H3("Congested Rows Table (P0 or P1)", style={"marginTop": 0}),
                dash_table.DataTable(
                    id="congestion-table", page_size=TABLE_PAGE_SIZE,
                    sort_action="native", filter_action="native", page_action="native",
                    style_table={"overflowX": "auto", "maxHeight": "650px", "overflowY": "auto"},
                    style_cell={"textAlign": "left", "minWidth": "120px",
                                "width": "120px", "maxWidth": "220px", "padding": "8px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#f1f5f9"},
                    style_data_conditional=[
                        {"if": {"filter_query": "{P0_Congestion} = true"},
                         "backgroundColor": "#fee2e2", "color": "#7f1d1d"},
                        {"if": {"filter_query": "{P1_Congestion} = true"},
                         "backgroundColor": "#fff7ed", "color": "#7c2d12"},
                    ],
                ),
            ], style=SECTION_STYLE),
        ])
    ])


def _tab_geometry_map():
    return dcc.Tab(label="🗺️ Geometry Map", children=[
        html.Div(
            "Map — to be built.",
            style={
                "padding": "60px",
                "textAlign": "center",
                "color": "#94a3b8",
                "fontSize": "16px",
            },
        )
    ])
