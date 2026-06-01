"""
Forecast tab layout — AI-powered forecasting panel.

Changes (v2):
  - Cell selector removed — cells are driven by the global filter strip.
  - "Save Week Forecast" button forecasts 7 days for all filtered cells in
    a background thread and persists the result to datasets/forecast_cache.csv.
  - Live status bar (fc-forecast-status) shows real-time progress.
  - Changing global filter when a saved cache exists auto-reloads the store.
"""

from dash import dcc, html, dash_table
from config import SECTION_STYLE, _PILL_BTN, _PRIMARY_BTN, TABLE_PAGE_SIZE, _TAB_SELECTED, _TAB_STYLE

_ACCENT      = "#6366f1"
_ACCENT_SOFT = "#eef2ff"
_WARN        = "#f97316"
_DANGER      = "#ef4444"
_SUCCESS     = "#22c55e"
_SAVE_COLOR  = "#0891b2"

_SECTION = {**SECTION_STYLE, "marginBottom": "16px"}

_INP = {"padding": "4px 8px", "border": "1px solid #d1d5db",
        "borderRadius": "8px", "fontSize": "12px"}


def _fc_thr_inp(id_: str, label: str, default, mn, mx):
    return html.Div([
        html.Label(label, style={"fontSize": "10px", "color": "#64748b",
                                  "display": "block", "marginBottom": "2px"}),
        dcc.Input(id=id_, type="number", value=default, min=mn, max=mx,
                  debounce=True, style={**_INP, "width": "80px"}),
    ])


_RUN_BTN = {
    **_PRIMARY_BTN,
    "background": _ACCENT,
    "border": f"1px solid {_ACCENT}",
    "fontSize": "13px",
    "padding": "8px 22px",
    "borderRadius": "10px",
    "fontWeight": 700,
    "letterSpacing": "0.01em",
}

_SAVE_BTN = {
    **_PRIMARY_BTN,
    "background": _SAVE_COLOR,
    "border": f"1px solid {_SAVE_COLOR}",
    "fontSize": "13px",
    "padding": "8px 18px",
    "borderRadius": "10px",
    "fontWeight": 700,
    "letterSpacing": "0.01em",
    "cursor": "pointer",
}

_EXPORT_BTN = {
    **_PILL_BTN,
    "fontSize": "12px",
    "padding": "6px 16px",
}


def _loading(child_id, child, spinner_type="dot", color=_ACCENT, height=None):
    style = {}
    if height:
        style["minHeight"] = height
    return dcc.Loading(
        type=spinner_type, color=color, style=style,
        overlay_style={"visibility": "visible", "opacity": 0.4},
        children=child,
    )


def build_forecast_tab():
    return dcc.Tab(label="🔮 AI Forecasting", style=_TAB_STYLE, selected_style=_TAB_SELECTED, children=[
        html.Div([

            # ── Header ──────────────────────────────────────────────────────
            html.Div([
                html.Div([
                    html.H2("AI-Powered KPI Forecasting",
                            style={"margin": "0 0 4px 0", "color": "#0f172a", "fontSize": "18px"}),
                    html.P(
                        "Forecast DL PRB Load, Active Users and Throughput. "
                        "Cells are selected via the global filter above. "
                        "Save a full week forecast for the current filter, then re-filter to explore.",
                        style={"margin": 0, "color": "#64748b", "fontSize": "13px"},
                    ),
                ], style={"flex": 1}),
                _loading(
                    "fc-model-status-bar",
                    html.Div(id="fc-model-status-bar",
                             style={"display": "flex", "gap": "8px",
                                    "alignItems": "center", "flexWrap": "wrap"}),
                    spinner_type="dot",
                ),
            ], style={
                "display": "flex", "alignItems": "flex-start", "gap": "16px",
                "flexWrap": "wrap", "marginBottom": "16px",
                "padding": "16px 20px",
                "background": "white", "border": f"1px solid {_ACCENT}33",
                "borderRadius": "16px", "borderLeft": f"4px solid {_ACCENT}",
            }),

            # ── Control panel ─────────────────────────────────────────────────
            html.Div([

                # Global-filter summary pill
                html.Div([
                    html.Div([
                        html.Span("📡", style={"marginRight": "6px", "fontSize": "14px"}),
                        html.Span("Using global filter", style={
                            "fontSize": "12px", "color": "#374151", "fontWeight": 600}),
                    ], style={"marginBottom": "6px", "display": "flex", "alignItems": "center"}),
                    html.Div(id="fc-filter-summary",
                             style={"fontSize": "13px", "color": _ACCENT, "fontWeight": 700,
                                    "marginBottom": "4px"}),
                    html.Span(
                        "Adjust Region / Site / Cell / Date filters above to change selection.",
                        style={"fontSize": "10px", "color": "#94a3b8"},
                    ),
                ], style={
                    "flex": "0 0 auto", "minWidth": "210px",
                    "padding": "10px 14px",
                    "background": "#f0f4ff",
                    "border": "1px solid #c7d2fe",
                    "borderRadius": "12px",
                }),

                # Horizon
                html.Div([
                    html.Label("Forecast Horizon",
                               style={"fontSize": "12px", "fontWeight": 600,
                                      "color": "#374151", "marginBottom": "6px",
                                      "display": "block"}),
                    dcc.RadioItems(
                        id="fc-horizon",
                        options=[
                            {"label": "6 h",    "value": 6},
                            {"label": "12 h",   "value": 12},
                            {"label": "24 h",   "value": 24},
                            {"label": "48 h",   "value": 48},
                            {"label": "72 h",   "value": 72},
                            {"label": "1 week", "value": 168},
                        ],
                        value=24, inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "14px", "fontSize": "13px", "cursor": "pointer"},
                    ),
                ], style={"flex": 1, "minWidth": "260px"}),

                # P0 Thresholds
                html.Div([
                    html.Label("🔴 P0 Thresholds",
                               style={"fontSize": "11px", "fontWeight": 700,
                                      "color": _DANGER, "marginBottom": "6px", "display": "block"}),
                    html.Div([
                        _fc_thr_inp("fc-p0-prb",   "PRB > (%)",      90, 50, 100),
                        _fc_thr_inp("fc-p0-thp",   "Tput < (kbps)", 2000, 100, 20000),
                        _fc_thr_inp("fc-p0-users", "Users >",          8, 1, 200),
                    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
                ], style={"flex": "0 0 auto"}),

                # P1 Thresholds
                html.Div([
                    html.Label("🟠 P1 Thresholds",
                               style={"fontSize": "11px", "fontWeight": 700,
                                      "color": _WARN, "marginBottom": "6px", "display": "block"}),
                    html.Div([
                        _fc_thr_inp("fc-p1-prb",   "PRB > (%)",      80, 40, 99),
                        _fc_thr_inp("fc-p1-thp",   "Tput < (kbps)", 4000, 100, 20000),
                        _fc_thr_inp("fc-p1-users", "Users >",          4, 1, 100),
                    ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
                ], style={"flex": "0 0 auto"}),

                # Legacy hidden IDs kept for backward compat
                html.Div([
                    dcc.Input(id="fc-prb-threshold",   type="number", value=90, style={"display": "none"}),
                    dcc.Input(id="fc-users-threshold", type="number", value=4,  style={"display": "none"}),
                ]),

                # Action buttons
                html.Div([
                    dcc.Loading(
                        id="fc-run-loading",
                        type="circle",
                        color=_ACCENT,
                        children=html.Div([
                            html.Button("▶  Run Forecast", id="fc-run-btn",
                                        n_clicks=0, style=_RUN_BTN),
                            html.Div(id="fc-run-spinner-target", style={"display": "none"}),
                        ]),
                    ),
                    html.Div(style={"height": "8px"}),
                    html.Button("📅  Save Week Forecast",
                                id="fc-save-week-btn", n_clicks=0, style=_SAVE_BTN),
                    html.Div(
                        "Runs 7-day forecast for all filtered cells in background.",
                        style={"fontSize": "10px", "color": "#64748b",
                               "marginTop": "4px", "maxWidth": "220px"},
                    ),
                ], style={
                    "flex": "0 0 auto", "display": "flex",
                    "flexDirection": "column", "alignItems": "flex-start",
                }),

            ], style={
                "display": "flex", "gap": "20px", "flexWrap": "wrap",
                "alignItems": "flex-end",
                "padding": "16px 20px", "background": "white",
                "border": "1px solid #e5e7eb", "borderRadius": "16px",
                "marginBottom": "8px",
            }),

            # ── Live status / progress bar ────────────────────────────────────
            html.Div(
                id="fc-forecast-status",
                style={
                    "marginBottom": "14px", "padding": "10px 18px",
                    "background": "#f8fafc", "border": "1px solid #e2e8f0",
                    "borderRadius": "10px", "fontSize": "13px",
                    "minHeight": "42px", "display": "flex",
                    "alignItems": "center", "gap": "10px",
                },
            ),

            # Polling interval (disabled until a background job starts)
            dcc.Interval(id="fc-forecast-poll", interval=1200, disabled=True),

            # ── Summary cards ─────────────────────────────────────────────────
            dcc.Loading(
                type="dot", color=_ACCENT,
                children=html.Div(
                    id="fc-summary-cards",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
                        "gap": "12px", "marginBottom": "16px", "minHeight": "90px",
                    },
                ),
            ),

            # ── Alert banner ──────────────────────────────────────────────────
            dcc.Loading(
                type="dot", color=_WARN,
                children=html.Div(id="fc-alert-banner",
                                  style={"marginBottom": "14px", "minHeight": "40px"}),
            ),

            # ── Main forecast chart ───────────────────────────────────────────
            html.Div([
                html.Div([
                    html.Label("KPI to display",
                               style={"fontSize": "12px", "fontWeight": 600,
                                      "color": "#374151", "marginRight": "10px"}),
                    dcc.RadioItems(
                        id="fc-kpi-selector",
                        options=[
                            {"label": "DL PRB Load",           "value": "DL PRB Load"},
                            {"label": "Active Users",           "value": "Active Users"},
                            {"label": "LTE DL User Throughput", "value": "LTE DL User Throughput"},
                        ],
                        value="DL PRB Load", inline=True,
                        inputStyle={"marginRight": "4px"},
                        labelStyle={"marginRight": "18px", "fontSize": "13px"},
                    ),
                    html.Div(style={"flex": 1}),
                    html.Button("⬇  Export CSV", id="fc-export-btn",
                                n_clicks=0, style=_EXPORT_BTN),
                    dcc.Download(id="fc-download"),
                ], style={"display": "flex", "alignItems": "center",
                          "flexWrap": "wrap", "gap": "8px", "marginBottom": "10px"}),

                dcc.Loading(
                    type="graph", color=_ACCENT,
                    children=dcc.Graph(
                        id="fc-main-chart",
                        config={"displayModeBar": True,
                                "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
                    ),
                ),
            ], style=_SECTION),

            # ── Heatmap + PRB box ─────────────────────────────────────────────
            html.Div([
                html.Div([
                    html.H4("Congestion Risk Heatmap  (Hour of Day × Cell)",
                            style={"margin": "0 0 8px 0", "fontSize": "14px", "color": "#0f172a"}),
                    html.P("Red = P0 forecast · Orange = P1 · Green = Clear",
                           style={"fontSize": "11px", "color": "#64748b", "margin": "0 0 8px 0"}),
                    dcc.Loading(type="cube", color=_ACCENT,
                                children=dcc.Graph(id="fc-heatmap", config={"displayModeBar": False})),
                ], style={**_SECTION, "flex": 3, "marginBottom": 0}),

                html.Div([
                    html.H4("Forecasted PRB Distribution",
                            style={"margin": "0 0 8px 0", "fontSize": "14px", "color": "#0f172a"}),
                    html.P("Box plot of forecasted DL PRB Load per cell.",
                           style={"fontSize": "11px", "color": "#64748b", "margin": "0 0 8px 0"}),
                    dcc.Loading(type="cube", color=_ACCENT,
                                children=dcc.Graph(id="fc-prb-box", config={"displayModeBar": False})),
                ], style={**_SECTION, "flex": 2, "marginBottom": 0}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"}),

            # ── Time-to-breach + busy hours ───────────────────────────────────
            html.Div([
                html.Div([
                    html.H4("Estimated Time-to-Breach per Cell",
                            style={"margin": "0 0 8px 0", "fontSize": "14px", "color": "#0f172a"}),
                    html.P("Hours until each cell first hits the P0 threshold.",
                           style={"fontSize": "11px", "color": "#64748b", "margin": "0 0 8px 0"}),
                    dcc.Loading(type="dot", color=_DANGER,
                                children=dcc.Graph(id="fc-time-to-breach", config={"displayModeBar": False})),
                ], style={**_SECTION, "flex": 1, "marginBottom": 0}),

                html.Div([
                    html.H4("Forecasted Busy Hours",
                            style={"margin": "0 0 8px 0", "fontSize": "14px", "color": "#0f172a"}),
                    html.P("Hours of day with highest avg forecasted PRB load.",
                           style={"fontSize": "11px", "color": "#64748b", "margin": "0 0 8px 0"}),
                    dcc.Loading(type="dot", color=_WARN,
                                children=dcc.Graph(id="fc-busy-hours", config={"displayModeBar": False})),
                ], style={**_SECTION, "flex": 1, "marginBottom": 0}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"}),

            # ── Weekly risk calendar ──────────────────────────────────────────
            html.Div([
                html.H4("📅  Weekly Risk Calendar  (Hour of Day × Day of Week)",
                        style={"margin": "0 0 4px 0", "fontSize": "14px", "color": "#0f172a"}),
                html.P("Predicted DL PRB Load averaged over all selected cells. "
                       "Dark red = P0 risk · Orange = P1 · Green = clear. "
                       "Requires a ≥ 168 h (1-week) horizon.",
                       style={"fontSize": "11px", "color": "#64748b", "margin": "0 0 8px 0"}),
                dcc.Loading(type="cube", color=_ACCENT,
                            children=dcc.Graph(id="fc-weekly-calendar", config={"displayModeBar": False})),
            ], style=_SECTION),

            # ── What-if ───────────────────────────────────────────────────────
            html.Div([
                html.H4("⚡  What-If: Traffic Growth Simulator",
                        style={"margin": "0 0 4px 0", "fontSize": "14px", "color": "#0f172a"}),
                html.P("Scale the forecasted DL PRB Load by a growth multiplier "
                       "and see how many cells would breach the P0 threshold.",
                       style={"fontSize": "11px", "color": "#64748b", "margin": "0 0 12px 0"}),
                html.Div([
                    html.Div([
                        html.Label("Traffic Growth Multiplier",
                                   style={"fontSize": "12px", "fontWeight": 600,
                                          "color": "#374151", "marginBottom": "6px", "display": "block"}),
                        dcc.Slider(
                            id="fc-growth-slider", min=1.0, max=3.0, step=0.1, value=1.0,
                            marks={1: "×1 (current)", 1.5: "×1.5", 2: "×2", 2.5: "×2.5", 3: "×3"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                    ], style={"flex": 2}),
                    html.Div(id="fc-whatif-output",
                             style={"flex": 1, "padding": "10px 16px",
                                    "background": "#f8fafc", "borderRadius": "12px",
                                    "border": "1px solid #e2e8f0",
                                    "fontSize": "13px", "minHeight": "60px"}),
                ], style={"display": "flex", "gap": "20px", "alignItems": "center", "flexWrap": "wrap"}),
                dcc.Loading(type="dot", color=_WARN,
                            children=dcc.Graph(id="fc-whatif-chart", config={"displayModeBar": False})),
            ], style=_SECTION),

            # ── Congestion alert table ────────────────────────────────────────
            html.Div([
                html.Div([
                    html.H3("🚨  Forecasted Congestion Alert",
                            style={"margin": "0 0 4px 0", "fontSize": "15px", "color": "#0f172a"}),
                    html.P("Cells predicted to be congested, ranked by severity.",
                           style={"fontSize": "12px", "color": "#64748b", "margin": "0 0 12px 0"}),
                ]),
                dcc.Loading(
                    type="dot", color=_DANGER,
                    children=dash_table.DataTable(
                        id="fc-alert-table",
                        page_size=TABLE_PAGE_SIZE,
                        sort_action="native", filter_action="native", page_action="native",
                        style_table={"overflowX": "auto", "maxHeight": "480px", "overflowY": "auto"},
                        style_cell={"textAlign": "left", "minWidth": "120px",
                                    "padding": "9px 12px",
                                    "fontFamily": "Inter, Arial, sans-serif", "fontSize": "12px"},
                        style_header={"fontWeight": 700, "backgroundColor": "#f1f5f9",
                                      "borderBottom": "2px solid #e2e8f0"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{Risk Level} = "P0"'},
                             "backgroundColor": "#fee2e2", "color": "#7f1d1d"},
                            {"if": {"filter_query": '{Risk Level} = "P1"'},
                             "backgroundColor": "#fff7ed", "color": "#7c2d12"},
                        ],
                    ),
                ),
            ], style=_SECTION),

            # ── Hidden stores ─────────────────────────────────────────────────
            dcc.Store(id="fc-forecast-store", data={}),
            # Holds metadata about the last saved week-forecast cache
            dcc.Store(id="fc-week-store", storage_type="session", data={}),

        ], style={"padding": "4px 0"})
    ])
