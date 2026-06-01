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
    P0_COLOR, P1_COLOR, _TAB_SELECTED, _TAB_STYLE
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
    return dcc.Tab(label="🎯 Optimization", style=_TAB_STYLE, selected_style=_TAB_SELECTED, children=[
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

            # ══════════════════════════════════════════════════════════════════
            # RF Cluster Analysis  (reads global filter, not the region picker)
            # ══════════════════════════════════════════════════════════════════

            # ── Divider ───────────────────────────────────────────────────────
            html.Div([
                html.Div(style={
                    "flex": 1,
                    "height": "1px",
                    "background": "linear-gradient(to right, transparent, #c7d2fe, transparent)",
                }),
                html.Span("RF Cluster Analysis", style={
                    "fontSize": "11px", "fontWeight": 700,
                    "color": "#6366f1", "padding": "0 16px",
                    "textTransform": "uppercase", "letterSpacing": "1px",
                    "whiteSpace": "nowrap",
                }),
                html.Div(style={
                    "flex": 1,
                    "height": "1px",
                    "background": "linear-gradient(to right, #c7d2fe, transparent)",
                }),
            ], style={
                "display": "flex", "alignItems": "center",
                "margin": "28px 0 20px 0",
            }),

            # ── Header card ───────────────────────────────────────────────────
            html.Div([
                html.Div([
                    html.H2("🔭  RF Cluster Analysis",
                            style={"margin": "0 0 4px 0", "color": TEXT_PRIMARY,
                                   "fontSize": "18px"}),
                    html.P(
                        "Analyses the cells selected by the global filter (Region / Site / Cell) "
                        "as a whole RF cluster. Detects imbalances within sectors, TA-driven "
                        "coverage issues, intra- and inter-site neighbour opportunities, missing "
                        "frequency carriers, and long-term capacity gaps — then ranks "
                        "recommendations by recovery potential following the Orange tiered "
                        "action ladder.",
                        style={"margin": 0, "color": TEXT_MUTED, "fontSize": "13px",
                               "lineHeight": "1.6"},
                    ),
                ], style={"flex": 1}),
                _loading("rfa-status-bar",
                         html.Div(id="rfa-status-bar", style={
                             "display": "flex", "gap": "8px",
                             "alignItems": "center", "flexWrap": "wrap",
                         })),
            ], style={
                "display": "flex", "alignItems": "flex-start", "gap": "16px",
                "flexWrap": "wrap", "marginBottom": "16px",
                "padding": "16px 20px",
                "background": "white",
                "border": "1px solid #6366f133",
                "borderRadius": "16px",
                "borderLeft": "4px solid #6366f1",
            }),

            # ── Control panel ─────────────────────────────────────────────────
            html.Div([
                # Filter summary pill
                html.Div([
                    html.Div([
                        html.Span("📡", style={"marginRight": "6px", "fontSize": "14px"}),
                        html.Span("Global filter scope",
                                  style={"fontSize": "12px", "color": "#374151",
                                         "fontWeight": 600}),
                    ], style={"marginBottom": "6px", "display": "flex",
                              "alignItems": "center"}),
                    html.Div(id="rfa-filter-summary",
                             style={"fontSize": "13px", "color": ACCENT,
                                    "fontWeight": 700, "marginBottom": "4px"}),
                    html.Span(
                        "Adjust Region / Site / Cell / Date filters above to change selection.",
                        style={"fontSize": "10px", "color": "#94a3b8"},
                    ),
                ], style={
                    "flex": "0 0 auto", "minWidth": "220px",
                    "padding": "10px 14px",
                    "background": "#f0f4ff",
                    "border": "1px solid #c7d2fe",
                    "borderRadius": "12px",
                }),

                # Neighbour radius slider
                html.Div([
                    html.Label(
                        "Neighbour search radius",
                        style={"fontSize": "12px", "fontWeight": 600,
                               "color": "#374151", "marginBottom": "6px",
                               "display": "block"},
                    ),
                    dcc.Slider(
                        id="rfa-radius-slider",
                        min=500, max=5000, step=250, value=2000,
                        marks={
                            500:  {"label": "500 m"},
                            1000: {"label": "1 km"},
                            2000: {"label": "2 km"},
                            3000: {"label": "3 km"},
                            5000: {"label": "5 km"},
                        },
                        tooltip={"placement": "bottom", "always_visible": True},
                    ),
                    html.Div(
                        "Max distance to search for inter-site neighbour sectors "
                        "that can absorb overflow.",
                        style={"fontSize": "10px", "color": TEXT_MUTED,
                               "marginTop": "4px"},
                    ),
                ], style={"flex": 3, "minWidth": "280px"}),

                # Run button
                html.Div([
                    html.Label("\u00a0",
                               style={"display": "block", "marginBottom": "6px"}),
                    _loading("rfa-run-loading",
                             html.Button(
                                 "🔭  Analyse Cluster",
                                 id="rfa-run-btn", n_clicks=0,
                                 style={
                                     **_PRIMARY_BTN,
                                     "background": "#6366f1",
                                     "border": "1px solid #6366f1",
                                     "fontSize": "13px",
                                     "padding": "10px 26px",
                                     "borderRadius": "10px",
                                     "fontWeight": 700,
                                 },
                             )),
                ], style={"flex": 0, "flexBasis": "200px"}),

            ], style={
                **_SECTION,
                "display": "flex", "gap": "20px", "flexWrap": "wrap",
                "alignItems": "flex-end",
            }),

            # ── Cluster stats bar ──────────────────────────────────────────────
            dcc.Loading(
                type="dot", color=ACCENT,
                children=html.Div(
                    id="rfa-cluster-stats",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(130px, 1fr))",
                        "gap": "12px", "marginBottom": "16px", "minHeight": "80px",
                    },
                ),
            ),

            # ── Warning banner ────────────────────────────────────────────────
            html.Div(id="rfa-warning-banner", style={"marginBottom": "10px"}),

            # ── Topology heatmap grid ──────────────────────────────────────────
            html.Div([
                html.Div([
                    html.Span("📶  Cluster Topology",
                              style={"fontWeight": 700, "fontSize": "14px",
                                     "color": TEXT_PRIMARY}),
                    html.Span(
                        "One row per site · each chip = one cell (frequency + avg PRB load)",
                        style={"fontSize": "11px", "color": TEXT_MUTED,
                               "marginLeft": "10px"},
                    ),
                ], style={
                    "padding": "12px 16px",
                    "borderBottom": f"1px solid {BORDER_COLOR}",
                    "display": "flex", "alignItems": "center",
                    "flexWrap": "wrap", "gap": "4px",
                }),
                dcc.Loading(
                    type="dot", color=ACCENT,
                    children=html.Div(
                        id="rfa-topology-panel",
                        style={
                            "padding": "10px 14px",
                            "maxHeight": "320px",
                            "overflowY": "auto",
                            "minHeight": "60px",
                        },
                    ),
                ),
            ], style={
                "background": SURFACE,
                "borderRadius": "12px",
                "border": f"1px solid {BORDER_COLOR}",
                "marginBottom": "16px",
            }),

            # ── Recommendations — tabbed by priority ──────────────────────────
            html.Div([

                # Header row
                html.Div([
                    html.Span("💡  Recommendations",
                              style={"fontWeight": 700, "fontSize": "14px",
                                     "color": TEXT_PRIMARY}),
                    html.Span(id="rfa-rec-count",
                              style={"fontSize": "11px", "color": TEXT_MUTED,
                                     "marginLeft": "8px"}),
                    html.Div(style={"flex": 1}),
                    # Export button placeholder — wired in callback
                    html.Div(id="rfa-export-bar"),
                ], style={
                    "display": "flex", "alignItems": "center",
                    "padding": "12px 16px",
                    "borderBottom": f"1px solid {BORDER_COLOR}",
                    "gap": "6px",
                }),

                # Priority tabs
                dcc.Loading(
                    type="dot", color="#6366f1",
                    children=dcc.Tabs(
                        id="rfa-priority-tabs",
                        value="short",
                        children=[
                            dcc.Tab(
                                label="🟢 Short-term",
                                value="short",
                                style={
                                    "padding": "8px 18px", "fontSize": "12px",
                                    "fontWeight": 500, "color": TEXT_MUTED,
                                    "border": "none",
                                    "borderBottom": "2px solid transparent",
                                    "background": "transparent",
                                },
                                selected_style={
                                    "padding": "8px 18px", "fontSize": "12px",
                                    "fontWeight": 700, "color": "#15803d",
                                    "border": "none",
                                    "borderBottom": "2px solid #22c55e",
                                    "background": "#f0fdf4",
                                    "borderRadius": "8px 8px 0 0",
                                },
                                children=html.Div(
                                    id="rfa-panel-short",
                                    style={"padding": "0", "minHeight": "60px"},
                                ),
                            ),
                            dcc.Tab(
                                label="🟡 Medium-term",
                                value="medium",
                                style={
                                    "padding": "8px 18px", "fontSize": "12px",
                                    "fontWeight": 500, "color": TEXT_MUTED,
                                    "border": "none",
                                    "borderBottom": "2px solid transparent",
                                    "background": "transparent",
                                },
                                selected_style={
                                    "padding": "8px 18px", "fontSize": "12px",
                                    "fontWeight": 700, "color": "#b45309",
                                    "border": "none",
                                    "borderBottom": "2px solid #f59e0b",
                                    "background": "#fffbeb",
                                    "borderRadius": "8px 8px 0 0",
                                },
                                children=html.Div(
                                    id="rfa-panel-medium",
                                    style={"padding": "0", "minHeight": "60px"},
                                ),
                            ),
                            dcc.Tab(
                                label="🔴 Long-term",
                                value="long",
                                style={
                                    "padding": "8px 18px", "fontSize": "12px",
                                    "fontWeight": 500, "color": TEXT_MUTED,
                                    "border": "none",
                                    "borderBottom": "2px solid transparent",
                                    "background": "transparent",
                                },
                                selected_style={
                                    "padding": "8px 18px", "fontSize": "12px",
                                    "fontWeight": 700, "color": "#b91c1c",
                                    "border": "none",
                                    "borderBottom": "2px solid #ef4444",
                                    "background": "#fef2f2",
                                    "borderRadius": "8px 8px 0 0",
                                },
                                children=html.Div(
                                    id="rfa-panel-long",
                                    style={"padding": "0", "minHeight": "60px"},
                                ),
                            ),
                        ],
                        style={
                            "borderBottom": f"1px solid {BORDER_COLOR}",
                        },
                    ),
                ),

            ], style={
                "background": SURFACE,
                "borderRadius": "12px",
                "border": f"1px solid {BORDER_COLOR}",
                "marginBottom": "16px",
                "overflow": "hidden",
            }),

            # ── Recovery summary chart ────────────────────────────────────────
            html.Div([
                html.Div("📊  Estimated Recovery by Action Type",
                         style={"fontWeight": 700, "fontSize": "14px",
                                "padding": "12px 16px",
                                "borderBottom": f"1px solid {BORDER_COLOR}"}),
                dcc.Loading(
                    type="dot", color=ACCENT,
                    children=dcc.Graph(
                        id="rfa-recovery-chart",
                        config={"displayModeBar": False},
                        style={"minHeight": "200px"},
                    ),
                ),
            ], style={
                "background": SURFACE,
                "borderRadius": "12px",
                "border": f"1px solid {BORDER_COLOR}",
                "marginBottom": "16px",
            }),

            # ── Hidden stores ─────────────────────────────────────────────────
            dcc.Store(id="rfa-results-store", data=None),

        ], style={"padding": "10px 0"}),
    ])
