"""
App layout — builds and returns the full Dash component tree.
Call build_layout() after the app is created.
"""

import dash_leaflet as dl
from dash import dcc, html, dash_table

from pathlib import Path

from config import (
    DATA_PATH, RAW_DATA_PATH, DEFAULT_KPI, DEFAULT_KPI_2, DEFAULT_KPI_3,
    DEFAULT_X, DEFAULT_Y, DEFAULT_COLOR,
    P0_COLOR, P1_COLOR,
    SECTOR_ZOOM_THRESHOLD,
    SECTION_STYLE, APP_STYLE, TABLE_PAGE_SIZE,
    ACCENT, TEXT_MUTED, TEXT_PRIMARY, BORDER_COLOR, SURFACE,
    _PILL_BTN, _PRIMARY_BTN, _TAB_STYLE, _TAB_SELECTED,
)
from congestion_engine import DEFAULT_THRESHOLDS as _DT
from data_utils import categorical_options
from precompute import (
    DF, LOAD_ERROR,
    NUMERIC_COLS, KPI_OPTIONS, DATE_MIN, DATE_MAX, DATE_SECOND,
    SITE_SUMMARY,
)
from map_builder import SITE_POINT_TO_LAYER, SECTOR_STYLE
from forecast_tab_layout import build_forecast_tab
from optimizer_tab_layout import build_optimizer_tab


# ── Shared label style ─────────────────────────────────────────────────────
_LBL = {"fontSize": "11px", "fontWeight": 600, "color": TEXT_MUTED,
        "textTransform": "uppercase", "letterSpacing": "0.4px",
        "marginBottom": "4px", "display": "block"}

_DD_STYLE = {"fontSize": "13px"}


def _thr_input(key: str, label: str, default, mn, mx):
    """Labelled number input for a single congestion threshold."""
    return html.Div([
        html.Label(label, style={"fontSize": "11px", "color": TEXT_MUTED,
                                  "display": "block", "marginBottom": "3px"}),
        dcc.Input(
            id=f"thr-{key}", type="number",
            value=default, min=mn, max=mx, step=1,
            debounce=True,
            style={"width": "90px", "padding": "4px 8px", "fontSize": "12px",
                   "border": f"1px solid {BORDER_COLOR}", "borderRadius": "8px"},
        ),
    ])


def build_layout():
    # ── Header status badge ────────────────────────────────────────────────
    if LOAD_ERROR:
        status_badge = html.Div(
            ["⚠️  ", LOAD_ERROR],
            style={"fontSize": "12px", "color": "#fca5a5",
                   "background": "rgba(239,68,68,0.15)",
                   "padding": "6px 14px", "borderRadius": "20px",
                   "border": "1px solid rgba(239,68,68,0.3)"},
        )
    else:
        status_badge = html.Div(
            [html.Span("●", style={"color": "#4ade80", "marginRight": "6px"}),
             f"{len(DF):,} rows · {DATA_PATH.split('/')[-1]}"],
            style={"fontSize": "12px", "color": "rgba(255,255,255,0.8)",
                   "background": "rgba(255,255,255,0.08)",
                   "padding": "6px 14px", "borderRadius": "20px",
                   "border": "1px solid rgba(255,255,255,0.18)"},
        )

    return html.Div([

        # ══ Dark gradient header ══════════════════════════════════════════
        html.Div([
            html.Div([
                html.Span("📡", style={"fontSize": "30px", "lineHeight": "1"}),
                html.Div([
                    html.H1("RAN KPI Dashboard", style={
                        "margin": "0 0 2px 0", "color": "white",
                        "fontSize": "20px", "fontWeight": 800,
                        "letterSpacing": "-0.4px",
                    }),
                    html.Span(
                        "Network congestion & performance analytics",
                        style={"fontSize": "12px",
                               "color": "rgba(255,255,255,0.55)"},
                    ),
                ]),
            ], style={"display": "flex", "alignItems": "center", "gap": "14px"}),
            status_badge,
        ], style={
            "background": "linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%)",
            "padding": "18px 28px",
            "display": "flex", "alignItems": "center",
            "justifyContent": "space-between",
        }),

        # ══ Filter strip ══════════════════════════════════════════════════
        html.Div([
            # Row 1 — dimensions
            html.Div([
                html.Div([
                    html.Label("Region", style=_LBL),
                    dcc.Dropdown(
                        id="region-filter",
                        options=(categorical_options(DF["Region"])
                                 if not DF.empty and "Region" in DF.columns else []),
                        multi=True, placeholder="All regions",
                        style=_DD_STYLE,
                    ),
                ], style={"flex": 1}),
                html.Div([
                    html.Label("Radio Site", style=_LBL),
                    dcc.Dropdown(id="site-filter", multi=True,
                                 placeholder="All sites", style=_DD_STYLE),
                ], style={"flex": 1}),
                html.Div([
                    html.Label("Cell Name", style=_LBL),
                    dcc.Dropdown(id="cell-filter", multi=True,
                                 placeholder="All cells", style=_DD_STYLE),
                ], style={"flex": 1.4}),
                html.Div([
                    html.Label("FDD / TDD", style=_LBL),
                    dcc.Dropdown(
                        id="mode-filter",
                        options=(categorical_options(DF["Cell FDD TDD Indication"])
                                 if not DF.empty and "Cell FDD TDD Indication" in DF.columns else []),
                        multi=True, placeholder="All modes", style=_DD_STYLE,
                    ),
                ], style={"flex": 0.7}),
                html.Div([
                    html.Label("Date Range", style=_LBL),
                    dcc.DatePickerRange(
                        id="date-filter",
                        start_date=DATE_MIN, end_date=DATE_SECOND,
                        min_date_allowed=DATE_MIN, max_date_allowed=DATE_MAX,
                        display_format="YYYY-MM-DD",
                        style={"fontSize": "12px"},
                    ),
                ], style={"flex": 1.2}),
            ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
                      "marginBottom": "10px"}),

            # Row 2 — KPI selectors
            html.Div([
                html.Div([
                    html.Label("KPI Panel 1", style=_LBL),
                    dcc.Dropdown(id="main-kpi", options=KPI_OPTIONS,
                                 value=(DEFAULT_KPI if DEFAULT_KPI in NUMERIC_COLS
                                        else (NUMERIC_COLS[0] if NUMERIC_COLS else None)),
                                 style=_DD_STYLE),
                ], style={"flex": 1}),
                html.Div([
                    html.Label("KPI Panel 2", style=_LBL),
                    dcc.Dropdown(id="main-kpi-2", options=KPI_OPTIONS,
                                 value=(DEFAULT_KPI_2 if DEFAULT_KPI_2 in NUMERIC_COLS
                                        else (NUMERIC_COLS[1] if len(NUMERIC_COLS) > 1 else None)),
                                 style=_DD_STYLE),
                ], style={"flex": 1}),
                html.Div([
                    html.Label("KPI Panel 3", style=_LBL),
                    dcc.Dropdown(id="main-kpi-3", options=KPI_OPTIONS,
                                 value=(DEFAULT_KPI_3 if DEFAULT_KPI_3 in NUMERIC_COLS
                                        else (NUMERIC_COLS[2] if len(NUMERIC_COLS) > 2 else None)),
                                 style=_DD_STYLE),
                ], style={"flex": 1}),
                html.Div(style={"flex": 2.2}),   # spacer
            ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap"}),
        ], style={
            "background": SURFACE,
            "padding": "14px 28px 10px",
            "borderBottom": f"1px solid {BORDER_COLOR}",
            "boxShadow": "0 2px 8px rgba(0,0,0,0.04)",
        }),

        # ══ Main content ══════════════════════════════════════════════════
        html.Div([

            # KPI summary cards row
            html.Div(id="summary-cards", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))",
                "gap": "14px", "marginBottom": "20px",
            }),

            # Tabs
            dcc.Tabs([
                _tab_kpi_analytics(),
                _tab_congestion_analytics(),
                _tab_geometry_map(),
                build_forecast_tab(),
                build_optimizer_tab(), 
                _tab_settings(),
            ], style={
                "borderBottom": f"1px solid {BORDER_COLOR}",
                "marginBottom": "0",
            }),

        ], style={"padding": "20px 28px", "backgroundColor": "#f1f5f9",
                  "minHeight": "calc(100vh - 130px)"}),

        # ══ Stores & timers ═══════════════════════════════════════════════
        dcc.Store(id="selected-sites-store",    data=[]),
        dcc.Store(id="selected-cells-store",    data=[]),
        dcc.Store(id="animation-hour-store",    data=0),
        dcc.Store(id="animation-playing-store", data=False),
        dcc.Store(id="filter-update-source",    data="filters"),
        dcc.Store(id="map-reset-token",         data=0),
        dcc.Store(id="map-zoom-store",          data=8),
        dcc.Store(id="_map-export-dummy",       data=None),
        dcc.Store(id="sector-approach-store",   data="ta"),
        dcc.Store(id="congestion-thresholds-store", data={
            "p0_prb":   _DT["p0_prb"],   "p0_thp":   _DT["p0_thp"],
            "p0_users": _DT["p0_users"],
            "p1_prb":   _DT["p1_prb"],   "p1_thp":   _DT["p1_thp"],
            "p1_users": _DT["p1_users"],
        }),
        dcc.Interval(id="anim-interval", interval=900, n_intervals=0, disabled=True),

    ], style=APP_STYLE)


# ── Private tab builders ───────────────────────────────────────────────────

def _tab_kpi_analytics():
    return dcc.Tab(
        label="📊 KPI Analytics",
        style=_TAB_STYLE, selected_style=_TAB_SELECTED,
        children=[html.Div([

            html.Div([dcc.Graph(id="timeseries-chart")],   style=SECTION_STYLE),
            html.Div([
                html.Div([dcc.Graph(id="timeseries-chart-2")], style={"flex": 1}),
                html.Div([dcc.Graph(id="timeseries-chart-3")], style={"flex": 1}),
            ], style={"display": "flex", "gap": "16px",
                      "flexWrap": "wrap", "marginBottom": "20px"}),

            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Distribution KPI", style=_LBL),
                        dcc.Dropdown(id="hist-kpi", options=KPI_OPTIONS,
                                     value=(DEFAULT_KPI if DEFAULT_KPI in NUMERIC_COLS
                                            else (NUMERIC_COLS[0] if NUMERIC_COLS else None)),
                                     style=_DD_STYLE),
                    ], style={"marginBottom": "10px"}),
                    dcc.Graph(id="histogram-chart"),
                ], style={**SECTION_STYLE, "flex": 1}),
                html.Div([
                    html.Div([
                        html.Label("Top-Cells KPI", style=_LBL),
                        dcc.Dropdown(id="top-kpi", options=KPI_OPTIONS,
                                     value=(DEFAULT_KPI if DEFAULT_KPI in NUMERIC_COLS
                                            else (NUMERIC_COLS[0] if NUMERIC_COLS else None)),
                                     style=_DD_STYLE),
                    ], style={"marginBottom": "10px"}),
                    dcc.Graph(id="top-cells-chart"),
                ], style={**SECTION_STYLE, "flex": 1}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Scatter X", style=_LBL),
                        dcc.Dropdown(id="scatter-x", options=KPI_OPTIONS,
                                     value=(DEFAULT_X if DEFAULT_X in NUMERIC_COLS
                                            else (NUMERIC_COLS[0] if NUMERIC_COLS else None)),
                                     style=_DD_STYLE),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Label("Scatter Y", style=_LBL),
                        dcc.Dropdown(id="scatter-y", options=KPI_OPTIONS,
                                     value=(DEFAULT_Y if DEFAULT_Y in NUMERIC_COLS
                                            else (NUMERIC_COLS[1] if len(NUMERIC_COLS) > 1 else None)),
                                     style=_DD_STYLE),
                    ], style={"marginBottom": "8px"}),
                    html.Div([
                        html.Label("Color by", style=_LBL),
                        dcc.Dropdown(id="scatter-color",
                                     options=[{"label": "None", "value": "__none__"}] + KPI_OPTIONS,
                                     value=(DEFAULT_COLOR if DEFAULT_COLOR in NUMERIC_COLS
                                            else "__none__"),
                                     style=_DD_STYLE),
                    ], style={"marginBottom": "8px"}),
                    dcc.Graph(id="scatter-chart"),
                ], style={**SECTION_STYLE, "flex": 1}),
                html.Div([dcc.Graph(id="correlation-chart")],
                         style={**SECTION_STYLE, "flex": 1}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

            html.Div([
                html.Div([
                    html.Span("📋", style={"fontSize": "15px", "marginRight": "8px"}),
                    html.Span("Filtered Data Table",
                              style={"fontWeight": 700, "fontSize": "14px",
                                     "color": TEXT_PRIMARY}),
                ], style={"display": "flex", "alignItems": "center",
                          "marginBottom": "12px"}),
                dash_table.DataTable(
                    id="detail-table", page_size=TABLE_PAGE_SIZE,
                    sort_action="native", filter_action="native", page_action="native",
                    export_format="csv", export_headers="display",
                    style_table={"overflowX": "auto", "maxHeight": "580px",
                                 "overflowY": "auto", "borderRadius": "12px"},
                    style_cell={"textAlign": "left", "minWidth": "110px",
                                "padding": "10px 12px", "fontSize": "12px",
                                "fontFamily": "'Inter', system-ui, sans-serif"},
                    style_header={"fontWeight": 700, "backgroundColor": "#f8fafc",
                                  "color": TEXT_MUTED, "fontSize": "11px",
                                  "textTransform": "uppercase", "letterSpacing": "0.4px",
                                  "borderBottom": f"2px solid {BORDER_COLOR}"},
                    style_data={"borderBottom": f"1px solid #f1f5f9"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#fafbff"},
                    ],
                ),
            ], style=SECTION_STYLE),

        ], style={"paddingTop": "20px"})]
    )


def _tab_congestion_analytics():
    return dcc.Tab(
        label="🔥 Congestion",
        style=_TAB_STYLE, selected_style=_TAB_SELECTED,
        children=[html.Div([

            # Tier legend strip
            html.Div([
                html.Span("Congestion tiers — ",
                          style={"fontSize": "12px", "color": TEXT_MUTED,
                                 "fontWeight": 500}),
                html.Span("● P0", style={"fontSize": "13px", "fontWeight": 700,
                                          "color": P0_COLOR, "marginLeft": "8px"}),
                html.Span(" severe", style={"fontSize": "12px", "color": TEXT_MUTED}),
                html.Span("    ● P1", style={"fontSize": "13px", "fontWeight": 700,
                                              "color": P1_COLOR, "marginLeft": "14px"}),
                html.Span(" moderate", style={"fontSize": "12px", "color": TEXT_MUTED}),
            ], style={
                "padding": "10px 16px", "background": "#fffbeb",
                "border": "1px solid #fde68a", "borderRadius": "12px",
                "marginBottom": "16px", "display": "flex", "alignItems": "center",
            }),

            html.Div(id="congestion-cards", style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
                "gap": "14px", "marginBottom": "20px",
            }),

            html.Div([dcc.Graph(id="congestion-timeseries-chart")], style=SECTION_STYLE),

            html.Div([
                html.Div([dcc.Graph(id="congestion-by-region-chart")],
                         style={**SECTION_STYLE, "flex": 1}),
                html.Div([dcc.Graph(id="congestion-by-freq-chart")],
                         style={**SECTION_STYLE, "flex": 1}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

            html.Div([
                html.Div([dcc.Graph(id="top-congested-cells-chart")],
                         style={**SECTION_STYLE, "flex": 1}),
                html.Div([dcc.Graph(id="busy-hour-chart")],
                         style={**SECTION_STYLE, "flex": 1}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

            html.Div([dcc.Graph(id="congested-scatter-chart")], style=SECTION_STYLE),

            html.Div([
                html.Div([
                    html.Span("🔥", style={"fontSize": "15px", "marginRight": "8px"}),
                    html.Span("Congested Rows (P0 or P1)",
                              style={"fontWeight": 700, "fontSize": "14px",
                                     "color": TEXT_PRIMARY}),
                ], style={"display": "flex", "alignItems": "center",
                          "marginBottom": "12px"}),
                dash_table.DataTable(
                    id="congestion-table", page_size=TABLE_PAGE_SIZE,
                    sort_action="native", filter_action="native", page_action="native",
                    export_format="csv", export_headers="display",
                    style_table={"overflowX": "auto", "maxHeight": "580px",
                                 "overflowY": "auto", "borderRadius": "12px"},
                    style_cell={"textAlign": "left", "minWidth": "110px",
                                "padding": "10px 12px", "fontSize": "12px",
                                "fontFamily": "'Inter', system-ui, sans-serif"},
                    style_header={"fontWeight": 700, "backgroundColor": "#f8fafc",
                                  "color": TEXT_MUTED, "fontSize": "11px",
                                  "textTransform": "uppercase", "letterSpacing": "0.4px",
                                  "borderBottom": f"2px solid {BORDER_COLOR}"},
                    style_data={"borderBottom": f"1px solid #f1f5f9"},
                    style_data_conditional=[
                        {"if": {"filter_query": "{P0_Congestion} = true"},
                         "backgroundColor": "#fff1f2", "color": "#9f1239"},
                        {"if": {"filter_query": "{P1_Congestion} = true"},
                         "backgroundColor": "#fff7ed", "color": "#9a3412"},
                        {"if": {"row_index": "odd"}, "backgroundColor": "#fafbff"},
                    ],
                ),
            ], style=SECTION_STYLE),

        ], style={"paddingTop": "20px"})]
    )


def _tab_geometry_map():
    if not SITE_SUMMARY.empty:
        center_lat = float(SITE_SUMMARY["lat"].median())
        center_lon = float(SITE_SUMMARY["lon"].median())
    else:
        center_lat, center_lon = 28.0, 2.0

    return dcc.Tab(
        label="🗺️ Network Map",
        style=_TAB_STYLE, selected_style=_TAB_SELECTED,
        children=[html.Div([

            # ── Map toolbar ────────────────────────────────────────────────
            html.Div([
                # Hour-of-day animation
                html.Div([
                    html.Span("Hour", style={"fontSize": "11px", "color": TEXT_MUTED,
                                             "fontWeight": 600, "whiteSpace": "nowrap"}),
                    html.Button("▶", id="anim-play-btn", n_clicks=0,
                                style={**_PILL_BTN, "padding": "4px 10px",
                                       "fontSize": "14px"}),
                    html.Div(dcc.Slider(
                        id="hour-slider", min=0, max=23, step=1, value=0,
                        marks={h: str(h) for h in [0, 6, 12, 18, 23]},
                        tooltip={"placement": "bottom", "always_visible": False},
                    ), style={"flex": 1, "minWidth": "160px"}),
                    html.Span(id="hour-display", children="00:00",
                              style={"fontSize": "12px", "fontWeight": 700,
                                     "color": TEXT_PRIMARY, "minWidth": "44px"}),
                    html.Button("↺", id="anim-reset-btn", n_clicks=0,
                                title="Reset to hour 0",
                                style={**_PILL_BTN, "padding": "4px 10px"}),
                ], style={"display": "flex", "alignItems": "center",
                          "gap": "8px", "flex": 1, "minWidth": "260px"}),

                # Divider
                html.Div(style={"width": "1px", "height": "28px",
                                "background": BORDER_COLOR, "margin": "0 6px"}),

                # Action buttons
                html.Button("↺ Reset", id="map-reset-btn", n_clicks=0,
                            style={**_PRIMARY_BTN}),
                html.Button("📡 Freq Shape", id="sector-approach-btn", n_clicks=0,
                            title="Toggle TA vs Frequency sector shapes",
                            style=_PILL_BTN),
                html.Button("📷 PNG", id="map-export-btn", n_clicks=0,
                            style=_PILL_BTN),

                # Divider
                html.Div(style={"width": "1px", "height": "28px",
                                "background": BORDER_COLOR, "margin": "0 6px"}),

                # Congestion filter
                html.Div([
                    html.Span("Tier:",
                              style={"fontSize": "11px", "color": TEXT_MUTED,
                                     "fontWeight": 600, "whiteSpace": "nowrap"}),
                    dcc.RadioItems(
                        id="map-congestion-filter",
                        options=[
                            {"label": " All",    "value": "all"},
                            {"label": " 🔴 P0",  "value": "p0"},
                            {"label": " 🟠 P1",  "value": "p1"},
                        ],
                        value="all", inline=True,
                        inputStyle={"marginRight": "3px"},
                        labelStyle={"marginRight": "8px", "fontSize": "11px",
                                    "cursor": "pointer"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),

                # Frequency band filter
                html.Div([
                    html.Span("Band:",
                              style={"fontSize": "11px", "color": TEXT_MUTED,
                                     "fontWeight": 600, "whiteSpace": "nowrap"}),
                    dcc.RadioItems(
                        id="map-freq-filter",
                        options=[
                            {"label": " All",  "value": "all"},
                            {"label": " 800",  "value": "800"},
                            {"label": " 1800", "value": "1800"},
                            {"label": " 2100", "value": "2100"},
                            {"label": " TDD",  "value": "tdd"},
                        ],
                        value="all", inline=True,
                        inputStyle={"marginRight": "3px"},
                        labelStyle={"marginRight": "8px", "fontSize": "11px",
                                    "cursor": "pointer"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),

                # KPI traffic overlay
                html.Div([
                    html.Span("Color by:",
                              style={"fontSize": "11px", "color": TEXT_MUTED,
                                     "fontWeight": 600, "whiteSpace": "nowrap"}),
                    dcc.Dropdown(
                        id="map-kpi-overlay",
                        options=[
                            {"label": "Congestion tier",      "value": "__congestion__"},
                            {"label": "DL PRB Load (%)",      "value": "DL PRB Load"},
                            {"label": "Active Users",         "value": "Active Users"},
                            {"label": "DL Throughput (kbps)", "value": "LTE DL User Throughput"},
                        ],
                        value="__congestion__",
                        clearable=False,
                        style={"fontSize": "11px", "minWidth": "170px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),

                # Zoom hint
                html.Span(
                    f"Sectors @ zoom ≥ {SECTOR_ZOOM_THRESHOLD}",
                    style={"fontSize": "10px", "color": TEXT_MUTED,
                           "whiteSpace": "nowrap"},
                ),
            ], style={
                "display": "flex", "alignItems": "center",
                "flexWrap": "wrap", "gap": "8px",
                "padding": "10px 16px",
                "background": SURFACE,
                "border": f"1px solid {BORDER_COLOR}",
                "borderRadius": "14px", "marginBottom": "14px",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.04)",
            }),

            # ── Map + panel ────────────────────────────────────────────────
            html.Div(id="map-export-area", children=[

                dl.Map(
                    id="main-map",
                    center=[center_lat, center_lon],
                    zoom=8,
                    children=[
                        dl.TileLayer(
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                            attribution="© OpenStreetMap contributors",
                            maxZoom=19,
                        ),
                        dl.GeoJSON(
                            id="sectors-geojson",
                            data={"type": "FeatureCollection", "features": []},
                            style=SECTOR_STYLE,
                            onEachFeature={"variable": "mapFunctions.sectorOnEachFeature"},
                            zoomToBoundsOnClick=False,
                            disableDefaultEventHandlers=True,
                        ),
                        dl.GeoJSON(
                            id="sites-geojson",
                            data={"type": "FeatureCollection", "features": []},
                            cluster=True,
                            zoomToBoundsOnClick=True,
                            superClusterOptions={
                                "radius": 80,
                                "maxZoom": SECTOR_ZOOM_THRESHOLD - 1,
                            },
                            pointToLayer=SITE_POINT_TO_LAYER,
                            onEachFeature={"variable": "mapFunctions.siteOnEachFeature"},
                            disableDefaultEventHandlers=True,
                        ),
                        dl.FeatureGroup([
                            dl.EditControl(
                                id="lasso-control",
                                draw={
                                    "polygon":      {"shapeOptions": {"color": ACCENT}},
                                    "rectangle":    {"shapeOptions": {"color": ACCENT}},
                                    "polyline":     False, "circle":  False,
                                    "marker":       False, "circlemarker": False,
                                },
                                edit={"edit": False, "remove": True},
                            )
                        ]),
                    ],
                    style={
                        "height": "680px", "flex": "1 1 0", "minWidth": 0,
                        "borderRadius": "14px", "overflow": "hidden",
                        "border": f"1px solid {BORDER_COLOR}",
                        "boxShadow": "0 2px 12px rgba(0,0,0,0.08)",
                    },
                ),

                html.Div([
                    html.Div(
                        id="map-info-panel",
                        style={
                            "padding": "16px", "background": SURFACE,
                            "border": f"1px solid {BORDER_COLOR}",
                            "borderRadius": "14px",
                            "height": "680px", "overflowY": "auto",
                            "boxShadow": "0 2px 12px rgba(0,0,0,0.06)",
                        },
                    )
                ], style={"width": "320px", "flexShrink": 0}),

            ], style={"display": "flex", "gap": "14px"}),

        ], style={"paddingTop": "16px"})]
    )


def _tab_settings():
    raw_exists = Path(RAW_DATA_PATH).exists()
    raw_label  = RAW_DATA_PATH.split("/")[-1]

    def _row(label, value):
        ok = value and not str(value).startswith("❌")
        return html.Div([
            html.Span(label, style={"fontSize": "12px", "color": TEXT_MUTED,
                                    "minWidth": "160px", "display": "inline-block"}),
            html.Span(str(value), style={"fontSize": "12px",
                                         "color": "#15803d" if ok else "#b91c1c",
                                         "fontWeight": 500}),
        ], style={"marginBottom": "6px"})

    return dcc.Tab(
        label="⚙️ Settings",
        style=_TAB_STYLE, selected_style=_TAB_SELECTED,
        children=[html.Div([

            # ── Section 1: Dataset status ──────────────────────────────────
            html.Div([
                html.Div([
                    html.Span("📂", style={"fontSize": "18px", "marginRight": "10px"}),
                    html.Span("Dataset Status", style={"fontWeight": 700,
                                                        "fontSize": "15px",
                                                        "color": TEXT_PRIMARY}),
                ], style={"display": "flex", "alignItems": "center",
                          "marginBottom": "14px"}),
                _row("Congestion dataset",
                     f"{len(DF):,} rows loaded — {DATA_PATH}"
                     if not LOAD_ERROR else f"❌ {LOAD_ERROR}"),
                _row("Raw KPI file",
                     f"{raw_label} found — {RAW_DATA_PATH}"
                     if raw_exists else f"❌ {raw_label} not found"),
            ], style=SECTION_STYLE),

            # ── Section 2: Congestion engine ──────────────────────────────
            html.Div([
                html.Div([
                    html.Span("🔄", style={"fontSize": "18px", "marginRight": "10px"}),
                    html.Span("Recompute Congestion Dataset",
                              style={"fontWeight": 700, "fontSize": "15px",
                                     "color": TEXT_PRIMARY}),
                ], style={"display": "flex", "alignItems": "center",
                          "marginBottom": "6px"}),
                html.P(
                    "Set the thresholds below, point to the raw KPI CSV, then click "
                    "Compute. The dashboard uses the saved CSV — changes only take "
                    "effect after restarting the server.",
                    style={"fontSize": "12px", "color": TEXT_MUTED,
                           "marginBottom": "18px"},
                ),

                # Thresholds — P0
                html.Div([
                    html.Div("🔴 P0 — severe (all three must hold simultaneously)",
                             style={"fontSize": "12px", "fontWeight": 700,
                                    "color": P0_COLOR, "marginBottom": "10px"}),
                    html.Div([
                        _thr_input("p0_prb",   "DL PRB Load > (%)",      _DT["p0_prb"],   50, 100),
                        _thr_input("p0_thp",   "DL Throughput < (kbps)", _DT["p0_thp"],   100, 50000),
                        _thr_input("p0_users", "Active Users >",         _DT["p0_users"], 1, 200),
                    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                              "marginBottom": "20px"}),
                ]),

                # Thresholds — P1
                html.Div([
                    html.Div("🟠 P1 — moderate (all three must hold simultaneously)",
                             style={"fontSize": "12px", "fontWeight": 700,
                                    "color": P1_COLOR, "marginBottom": "10px"}),
                    html.Div([
                        _thr_input("p1_prb",   "DL PRB Load > (%)",      _DT["p1_prb"],   40, 99),
                        _thr_input("p1_thp",   "DL Throughput < (kbps)", _DT["p1_thp"],   100, 50000),
                        _thr_input("p1_users", "Active Users >",         _DT["p1_users"], 1, 100),
                    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
                              "marginBottom": "20px"}),
                ]),

                # Mode selector
                html.Div([
                    html.Label("Mode", style=_LBL),
                    dcc.RadioItems(
                        id="recompute-mode",
                        options=[
                            {"label": " Replace all (full recompute)", "value": "replace"},
                            {"label": " Append new data only",         "value": "append"},
                        ],
                        value="replace",
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"marginRight": "20px", "fontSize": "13px",
                                    "color": TEXT_PRIMARY},
                    ),
                ], style={"marginBottom": "16px"}),

                # Raw file path + button
                html.Div([
                    html.Div([
                        html.Label("Raw KPI file path", style=_LBL),
                        dcc.Input(
                            id="raw-data-path-input",
                            value=RAW_DATA_PATH, type="text",
                            placeholder="datasets/final_kpi_data.csv",
                            style={"fontSize": "12px", "padding": "6px 10px",
                                   "border": f"1px solid {BORDER_COLOR}",
                                   "borderRadius": "10px", "width": "320px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("​", style=_LBL),  # spacer label
                        html.Button(
                            "⚙️  Compute & Save",
                            id="recompute-congestion-btn", n_clicks=0,
                            style={**_PILL_BTN,
                                   "background": "#f0fdf4",
                                   "border": "1px solid #86efac",
                                   "color": "#15803d",
                                   "fontWeight": 600,
                                   "padding": "7px 18px"},
                        ),
                    ]),
                ], style={"display": "flex", "gap": "16px",
                          "alignItems": "flex-end", "flexWrap": "wrap",
                          "marginBottom": "12px"}),

                html.Div(id="recompute-status",
                         style={"fontSize": "13px", "color": TEXT_MUTED,
                                "fontStyle": "italic"}),

            ], style=SECTION_STYLE),

            # ── Section 3: Retrain XGBoost models ─────────────────────────
            html.Div([
                html.Div([
                    html.Span("🤖", style={"fontSize": "18px", "marginRight": "10px"}),
                    html.Span("Retrain Forecast Models",
                              style={"fontWeight": 700, "fontSize": "15px",
                                     "color": TEXT_PRIMARY}),
                ], style={"display": "flex", "alignItems": "center",
                          "marginBottom": "6px"}),
                html.P([
                    "Trains new XGBoost models for all three KPIs (PRB, Active Users, Throughput) "
                    "from the raw KPI file. Training runs in the background — the dashboard stays "
                    "live using lag-blend fallback while the new models are being built. "
                    "Models reload automatically when done; ",
                    html.Strong("no server restart needed."),
                    html.Br(),
                    html.Span("⚠ Training takes several minutes on large datasets. "
                              "New cells/sites in the data will be learned; "
                              "cells absent from the training file fall back to lag-blend at inference.",
                              style={"color": TEXT_MUTED}),
                ], style={"fontSize": "12px", "color": TEXT_MUTED,
                          "marginBottom": "18px", "lineHeight": "1.6"}),

                html.Div([
                    html.Div([
                        html.Label("Raw KPI file path", style=_LBL),
                        dcc.Input(
                            id="train-data-path",
                            value=RAW_DATA_PATH, type="text",
                            placeholder="datasets/final_kpi_data.csv",
                            style={"fontSize": "12px", "padding": "6px 10px",
                                   "border": f"1px solid {BORDER_COLOR}",
                                   "borderRadius": "10px", "width": "320px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("​", style=_LBL),
                        html.Button(
                            "🚀  Retrain XGBoost Models",
                            id="retrain-models-btn", n_clicks=0,
                            style={**_PILL_BTN,
                                   "background": "#eff6ff",
                                   "border": "1px solid #93c5fd",
                                   "color": "#1d4ed8",
                                   "fontWeight": 600,
                                   "padding": "7px 18px"},
                        ),
                    ]),
                ], style={"display": "flex", "gap": "16px",
                          "alignItems": "flex-end", "flexWrap": "wrap",
                          "marginBottom": "4px"}),

                html.Div(id="retrain-status"),
                dcc.Interval(id="retrain-poll", interval=2000, disabled=True,
                             max_intervals=-1),

            ], style=SECTION_STYLE),

        ], style={"paddingTop": "20px", "maxWidth": "860px"})]
    )
