"""
Optimizer tab layout — congestion optimization control panel.

Engineer workflow:
  1. Pick a Region (or "All regions" for the whole network).
  2. Choose mode: Current congestion (now) or Forecasted (future).
  3. Pick an algorithm: BCDSA / Greedy / GA.
  4. Press Run. Map + table populate with recommendations.
"""

from dash import dcc, html, dash_table
from config import (
    SECTION_STYLE, _PILL_BTN, _PRIMARY_BTN,
    ACCENT, TEXT_PRIMARY, TEXT_MUTED, BORDER_COLOR, SURFACE,
    P0_COLOR, P1_COLOR,
)
from precompute import DF


# ── Region options reused from DF ───────────────────────────────────────────
_region_options = (
    [{"label": "All regions", "value": "__ALL__"}]
    + [{"label": r, "value": r} for r in sorted(DF["Region"].dropna().unique())]
    if not DF.empty and "Region" in DF.columns
    else [{"label": "All regions", "value": "__ALL__"}]
)

_SECTION = {**SECTION_STYLE, "marginBottom": "16px"}

_RUN_BTN = {
    **_PRIMARY_BTN,
    "background": ACCENT,
    "border": f"1px solid {ACCENT}",
    "fontSize": "13px",
    "padding": "10px 26px",
    "borderRadius": "10px",
    "fontWeight": 700,
}


def _loading(child_id, child, height=None):
    style = {"minHeight": height} if height else {}
    return dcc.Loading(
        type="dot", color=ACCENT, style=style,
        overlay_style={"visibility": "visible", "opacity": 0.4},
        children=child,
    )


def _summary_card(title: str, value_id: str, default: str = "—",
                  subtitle_id: str | None = None, color: str = TEXT_PRIMARY) -> html.Div:
    """One of the headline numbers shown after a run."""
    children = [
        html.Div(title, style={
            "fontSize": "10px", "fontWeight": 600, "color": TEXT_MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.5px",
            "marginBottom": "6px",
        }),
        html.Div(default, id=value_id, style={
            "fontSize": "22px", "fontWeight": 700, "color": color,
        }),
    ]
    if subtitle_id:
        children.append(html.Div("—", id=subtitle_id, style={
            "fontSize": "11px", "color": TEXT_MUTED, "marginTop": "4px",
        }))
    return html.Div(children, style={
        "padding": "14px 18px",
        "background": SURFACE,
        "borderRadius": "12px",
        "border": f"1px solid {BORDER_COLOR}",
        "flex": 1,
        "minWidth": "150px",
    })


def build_optimizer_tab() -> dcc.Tab:
    return dcc.Tab(label="🎯 Optimization", children=[
        html.Div([

            # ══ Header ═══════════════════════════════════════════════════════
            html.Div([
                html.Div([
                    html.H2("Congestion Optimization",
                            style={"margin": "0 0 4px 0", "color": TEXT_PRIMARY, "fontSize": "18px"}),
                    html.P(
                        "Recommend per-cell power and handover-margin adjustments "
                        "to relieve congested cells by redistributing users to "
                        "healthy neighbors.",
                        style={"margin": 0, "color": TEXT_MUTED, "fontSize": "13px"},
                    ),
                ], style={"flex": 1}),

                _loading("opt-status-bar",
                         html.Div(id="opt-status-bar", style={
                             "display": "flex", "gap": "8px",
                             "alignItems": "center", "flexWrap": "wrap",
                         })),
            ], style={
                "display": "flex", "alignItems": "flex-start", "gap": "16px",
                "flexWrap": "wrap", "marginBottom": "16px",
                "padding": "16px 20px",
                "background": "white", "border": f"1px solid {ACCENT}33",
                "borderRadius": "16px", "borderLeft": f"4px solid {ACCENT}",
            }),

            # ══ Control panel ════════════════════════════════════════════════
            html.Div([
                # Region picker
                html.Div([
                    html.Label("Region",
                               style={"fontSize": "12px", "fontWeight": 600,
                                      "color": "#374151", "marginBottom": "6px",
                                      "display": "block"}),
                    dcc.Dropdown(
                        id="opt-region",
                        options=_region_options,
                        value="__ALL__",
                        clearable=False,
                        style={"fontSize": "13px"},
                    ),
                ], style={"flex": 2, "minWidth": "200px"}),

                # Mode toggle
                html.Div([
                    html.Label("Optimize based on",
                               style={"fontSize": "12px", "fontWeight": 600,
                                      "color": "#374151", "marginBottom": "6px",
                                      "display": "block"}),
                    dcc.RadioItems(
                        id="opt-mode",
                        options=[
                            {"label": " Current state", "value": "current"},
                            {"label": " Forecast",      "value": "forecasted"},
                        ],
                        value="current", inline=True,
                        labelStyle={"marginRight": "16px", "fontSize": "13px",
                                    "cursor": "pointer"},
                    ),
                    html.Div(
                        "Forecast mode uses the AI Forecasting tab's predictions. "
                        "Run that first if you haven't.",
                        style={"fontSize": "10px", "color": TEXT_MUTED,
                               "marginTop": "4px"},
                    ),
                ], style={"flex": 2, "minWidth": "240px"}),

                # Algorithm picker
                html.Div([
                    html.Label("Algorithm",
                               style={"fontSize": "12px", "fontWeight": 600,
                                      "color": "#374151", "marginBottom": "6px",
                                      "display": "block"}),
                    dcc.RadioItems(
                        id="opt-algorithm",
                        options=[
                            {"label": " BCDSA",  "value": "bcdsa"},
                            {"label": " Greedy", "value": "greedy"},
                            {"label": " GA",     "value": "ga"},
                        ],
                        value="bcdsa", inline=True,
                        labelStyle={"marginRight": "14px", "fontSize": "13px",
                                    "cursor": "pointer"},
                    ),
                ], style={"flex": 1, "minWidth": "240px"}),

                # Run button
                html.Div([
                    html.Label("\u00a0", style={"display": "block", "marginBottom": "6px"}),
                    _loading("opt-run-loading",
                             html.Button("Run Optimizer",
                                         id="opt-run-btn", n_clicks=0,
                                         style=_RUN_BTN)),
                ], style={"flex": 0, "flexBasis": "180px"}),

            ], style={
                **_SECTION,
                "display": "flex", "gap": "20px", "flexWrap": "wrap",
                "alignItems": "flex-end",
            }),

            # ══ Summary cards ═══════════════════════════════════════════════
            html.Div([
                _summary_card("Excess Users — Before",  "opt-card-before"),
                _summary_card("Excess Users — After",   "opt-card-after",
                              subtitle_id="opt-card-reduction", color=ACCENT),
                _summary_card("Cells Changed",          "opt-card-changed",
                              subtitle_id="opt-card-fixed"),
                _summary_card("Runtime",                "opt-card-runtime",
                              subtitle_id="opt-card-algo"),
            ], style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))",
                "gap": "12px", "marginBottom": "16px",
            }),

            # ══ Map + table side-by-side ═════════════════════════════════════
            html.Div([
                # Map panel
                html.Div([
                    html.Div([
                        html.Span("Geographic view",
                                  style={"fontWeight": 600, "fontSize": "14px"}),
                        html.Div([
                            html.Button("Before", id="opt-map-before-btn", n_clicks=0,
                                        style={**_PILL_BTN, "fontSize": "11px",
                                               "padding": "4px 12px"}),
                            html.Button("After",  id="opt-map-after-btn",  n_clicks=0,
                                        style={**_PILL_BTN, "fontSize": "11px",
                                               "padding": "4px 12px",
                                               "marginLeft": "4px"}),
                        ], style={"display": "flex", "gap": "0", "marginLeft": "auto"}),
                    ], style={
                        "display": "flex", "alignItems": "center",
                        "padding": "12px 16px",
                        "borderBottom": f"1px solid {BORDER_COLOR}",
                    }),
                    dcc.Store(id="opt-map-mode", data="before"),
                    _loading("opt-map-container",
                             html.Div(id="opt-map-container",
                                      style={"height": "520px"}),
                             height="520px"),
                ], style={
                    "flex": 3, "minWidth": "400px",
                    "background": SURFACE,
                    "borderRadius": "12px",
                    "border": f"1px solid {BORDER_COLOR}",
                    "overflow": "hidden",
                }),

                # Table panel
                html.Div([
                    html.Div([
                        html.Span("Recommendations",
                                  style={"fontWeight": 600, "fontSize": "14px"}),
                        html.Span(id="opt-table-count", style={
                            "fontSize": "11px", "color": TEXT_MUTED,
                            "marginLeft": "auto",
                        }),
                    ], style={
                        "display": "flex", "alignItems": "center",
                        "padding": "12px 16px",
                        "borderBottom": f"1px solid {BORDER_COLOR}",
                    }),
                    _loading("opt-table-wrap",
                             html.Div(id="opt-table-wrap"),
                             height="520px"),
                ], style={
                    "flex": 2, "minWidth": "380px",
                    "background": SURFACE,
                    "borderRadius": "12px",
                    "border": f"1px solid {BORDER_COLOR}",
                    "overflow": "hidden",
                }),
            ], style={
                "display": "flex", "gap": "16px", "marginBottom": "16px",
                "flexWrap": "wrap",
            }),

            # ══ Bottom chart — top relieved cells ════════════════════════════
            html.Div([
                html.Div("Top relieved cells",
                         style={"fontWeight": 600, "fontSize": "14px",
                                "padding": "12px 16px",
                                "borderBottom": f"1px solid {BORDER_COLOR}"}),
                _loading("opt-top-chart",
                         dcc.Graph(id="opt-top-chart", config={"displayModeBar": False})),
            ], style={
                "background": SURFACE,
                "borderRadius": "12px",
                "border": f"1px solid {BORDER_COLOR}",
            }),

            # ══ Stores ═══════════════════════════════════════════════════════
            dcc.Store(id="opt-results-store", data=None),

        ], style={"padding": "10px 0"}),
    ])
