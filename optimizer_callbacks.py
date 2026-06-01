"""
Optimizer tab callbacks — wires the UI to the optimizer engine.

Reads from:
  precompute.DF                  — historical KPIs (for current mode)
  precompute.CELL_POLYGONS       — sector polygons (for neighbor graph)
  precompute.CELL_GEO_META       — cell coordinates and metadata
  forecast store ("fc-data-store") — XGBoost forecast records (for forecasted mode)

Writes:
  opt-results-store              — full optimizer output, JSON
  opt-card-*                     — headline number cards
  opt-map-container              — Leaflet map showing before/after states
  opt-table-wrap                 — DataTable of recommendations
  opt-top-chart                  — bar chart of biggest reductions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import dash_leaflet as dl
import plotly.graph_objects as go
from dash import (
    Input, Output, State, callback, no_update, html, dash_table, ctx, dcc,
)
from dash.exceptions import PreventUpdate

from config import (
    ACCENT, TEXT_PRIMARY, TEXT_MUTED, BORDER_COLOR, SURFACE,
    P0_COLOR, P1_COLOR,
)
from precompute import DF, CELL_GEO_META, CELL_POLYGONS

from optimizer.optimizer_core   import run_optimizer
from optimizer.neighbor_graph   import build_eta_matrix, neighbor_summary
from optimizer.capacity_predictor import get_current_state, get_forecasted_state


# Map color scheme
_COLOR_CONGESTED = "#dc2626"      # red
_COLOR_NEAR_CAP  = "#f59e0b"      # amber
_COLOR_HEALTHY   = "#10b981"      # green
_COLOR_EMPTY     = "#94a3b8"      # slate (no users)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cells_in_region(region: str) -> list[str]:
    """Pick cells for the chosen region. region='__ALL__' returns all cells."""
    if DF is None or DF.empty or "Cell Name" not in DF.columns:
        return []
    if region == "__ALL__":
        return sorted(DF["Cell Name"].dropna().unique().tolist())
    if "Region" not in DF.columns:
        return []
    sub = DF[DF["Region"] == region]
    return sorted(sub["Cell Name"].dropna().unique().tolist())


def _badge(label: str, color: str, background: str | None = None) -> html.Span:
    return html.Span(label, style={
        "fontSize": "11px", "fontWeight": 600,
        "padding": "4px 10px", "borderRadius": "12px",
        "color": color,
        "background": background or f"{color}1A",
        "border": f"1px solid {color}33",
    })


def _empty_map() -> html.Div:
    return html.Div(
        "Run the optimizer to see the map.",
        style={"display": "flex", "alignItems": "center",
               "justifyContent": "center", "height": "520px",
               "color": TEXT_MUTED, "fontSize": "13px"},
    )


def _empty_table() -> html.Div:
    return html.Div(
        "No recommendations yet.",
        style={"display": "flex", "alignItems": "center",
               "justifyContent": "center", "height": "520px",
               "color": TEXT_MUTED, "fontSize": "13px"},
    )


def _empty_chart() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[{
            "text": "Run the optimizer to see top-relieved cells",
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5, "showarrow": False,
            "font": {"size": 13, "color": TEXT_MUTED},
        }],
        height=320, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis={"visible": False}, yaxis={"visible": False},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK 1: status bar — updates as the user changes the region
# ─────────────────────────────────────────────────────────────────────────────
@callback(
    Output("opt-status-bar", "children"),
    Input("opt-region", "value"),
    Input("opt-mode", "value"),
)
def update_status_bar(region, mode):
    cells = _cells_in_region(region or "__ALL__")
    if not cells:
        return [_badge("No cells", "#dc2626")]
    mode_label = "Current" if mode == "current" else "Forecasted"
    return [
        _badge(f"{len(cells)} cells", ACCENT),
        _badge(f"Mode: {mode_label}", TEXT_PRIMARY),
        _badge("Ready", "#10b981"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK 2: run the optimizer
# ─────────────────────────────────────────────────────────────────────────────
@callback(
    Output("opt-results-store", "data"),
    Output("opt-status-bar",    "children", allow_duplicate=True),
    Input("opt-run-btn", "n_clicks"),
    State("opt-region",    "value"),
    State("opt-mode",      "value"),
    State("opt-algorithm", "value"),
    State("fc-forecast-store", "data"),  # forecast tab's store
    prevent_initial_call=True,
)
def run_optimization(n_clicks, region, mode, algorithm, fc_store):
    if not n_clicks:
        raise PreventUpdate

    cells = _cells_in_region(region or "__ALL__")
    if not cells:
        return None, [_badge("No cells in region", "#dc2626")]

    # Safety cap — running optimizer on tens of thousands of cells in a UI
    # request would block the dashboard. For very large clusters we trim to
    # the most-congested subset, matching what the offline pipeline did.
    if len(cells) > 200:
        # Pick the 200 cells with highest historical max PRB load
        if "DL PRB Load" in DF.columns:
            sub = DF[DF["Cell Name"].isin(cells)].dropna(subset=["DL PRB Load"])
            top = (sub.groupby("Cell Name")["DL PRB Load"].max()
                      .sort_values(ascending=False).head(200).index.tolist())
            cells = top
        else:
            cells = cells[:200]

    # Get state vectors
    try:
        if mode == "forecasted":
            records = (fc_store or {}).get("records", []) if isinstance(fc_store, dict) else []
            if not records:
                return None, [_badge(
                    "No forecast data — run AI Forecasting tab first",
                    "#dc2626",
                )]
            # Guard: the loaded forecast may have been run on a different
            # region. Check overlap before we try to optimize.
            fc_cells = {r.get("Cell Name") for r in records}
            covered = [c for c in cells if c in fc_cells]
            coverage_pct = 100.0 * len(covered) / max(len(cells), 1)
            if not covered:
                return None, [_badge(
                    "Loaded forecast covers different cells — "
                    "re-run AI Forecasting on this region",
                    "#dc2626",
                )]
            if coverage_pct < 50:
                # Partial coverage — proceed but warn, and only optimize
                # the cells the forecast actually covers.
                cells = covered
                partial_warning = _badge(
                    f"Forecast covers only {coverage_pct:.0f}% of region — "
                    f"optimizing {len(covered)} forecasted cells",
                    "#b45309",
                )
            else:
                partial_warning = None
            state = get_forecasted_state(cells, records, DF)
        else:
            partial_warning = None
            state = get_current_state(cells, DF)
    except ValueError as e:
        return None, [_badge(str(e)[:80], "#dc2626")]

    # Build neighbor graph
    eta = build_eta_matrix(cells)
    eta_stats = neighbor_summary(eta)

    # Run optimizer
    try:
        result = run_optimizer(
            lam=state["lam"],
            Lambda=state["Lambda"],
            eta=eta,
            cell_names=cells,
            algorithm=algorithm or "bcdsa",
        )
    except Exception as e:
        return None, [_badge(f"Optimizer error: {e}", "#dc2626")]

    # Attach extra metadata for the UI
    result["region"]        = region or "__ALL__"
    result["mode"]          = mode
    result["snapshot_time"] = state.get("timestamp")
    result["eta_summary"]   = eta_stats

    badge = _badge(
        f"Done · {result['reduction_pct']}% reduction · {result['runtime_s']:.2f}s",
        "#10b981",
    )
    status_children = [
        _badge(f"{result['n_cells']} cells", ACCENT),
        _badge(f"Mode: {'Forecasted' if mode == 'forecasted' else 'Current'}", TEXT_PRIMARY),
        badge,
    ]
    if mode == "forecasted" and partial_warning is not None:
        status_children.insert(2, partial_warning)
    return result, status_children


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK 3: summary cards
# ─────────────────────────────────────────────────────────────────────────────
@callback(
    Output("opt-card-before",    "children"),
    Output("opt-card-after",     "children"),
    Output("opt-card-reduction", "children"),
    Output("opt-card-changed",   "children"),
    Output("opt-card-fixed",     "children"),
    Output("opt-card-runtime",   "children"),
    Output("opt-card-algo",      "children"),
    Input("opt-results-store", "data"),
)
def update_summary_cards(result):
    if not result:
        return "—", "—", "—", "—", "—", "—", "—"
    return (
        f"{result['initial_congestion']:.1f}",
        f"{result['final_congestion']:.1f}",
        f"↓ {result['reduction_pct']:.1f}% reduction",
        f"{result['n_cells_changed']} / {result['n_cells']}",
        f"{result['n_cells_fixed']} fully fixed",
        f"{result['runtime_s']:.2f}s",
        result.get("algorithm", "—").upper(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK 4: map mode toggle (before/after)
# ─────────────────────────────────────────────────────────────────────────────
@callback(
    Output("opt-map-mode",        "data"),
    Output("opt-map-before-btn",  "style"),
    Output("opt-map-after-btn",   "style"),
    Input("opt-map-before-btn",   "n_clicks"),
    Input("opt-map-after-btn",    "n_clicks"),
)
def toggle_map_mode(_b, _a):
    from config import _PILL_BTN
    base = {**_PILL_BTN, "fontSize": "11px", "padding": "4px 12px"}
    active = {**base, "background": TEXT_PRIMARY, "color": "white",
              "border": f"1px solid {TEXT_PRIMARY}"}
    trigger = ctx.triggered_id
    if trigger == "opt-map-after-btn":
        return "after", {**base, "marginRight": "4px"}, active
    return "before", active, {**base, "marginLeft": "4px"}


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK 5: map render (depends on results AND toggle)
# ─────────────────────────────────────────────────────────────────────────────
@callback(
    Output("opt-map-container", "children"),
    Input("opt-results-store",  "data"),
    Input("opt-map-mode",       "data"),
)
def render_map(result, mode):
    if not result:
        return _empty_map()

    per_cell = result.get("per_cell", [])
    if not per_cell:
        return _empty_map()

    # Determine which "excess" to colorize by
    excess_key = "excess_after" if mode == "after" else "excess_before"

    polygons = []
    centers = []
    for r in per_cell:
        cn = r["cell_name"]
        ring = CELL_POLYGONS.get(cn)
        meta = CELL_GEO_META.get(cn, {})
        if not ring or "lat" not in meta:
            continue

        excess = r[excess_key]
        lam = r["lambda"]
        Lambda = r["Lambda"]
        is_changed = r["delta_p_db"] > 0 or r["delta_h_db"] > 0

        if lam < 0.1:
            color = _COLOR_EMPTY
        elif excess > 0.1:
            color = _COLOR_CONGESTED
        elif lam > 0.8 * Lambda:
            color = _COLOR_NEAR_CAP
        else:
            color = _COLOR_HEALTHY

        # Convert [lon, lat] ring to Leaflet's [lat, lon]
        coords = [[pt[1], pt[0]] for pt in ring]

        tooltip = (
            f"{cn}\n"
            f"λ={lam:.1f}  Λ={Lambda:.1f}\n"
            f"Excess: {excess:.2f}"
        )
        if is_changed:
            tooltip += (
                f"\nΔ℘={r['delta_p_db']:.0f}  Δℏ={r['delta_h_db']:.0f}\n"
                f"Offload: {r['total_offload']:.2f}"
            )

        polygons.append(dl.Polygon(
            positions=coords,
            color="#1f2937" if is_changed else "#9ca3af",
            weight=1.5 if is_changed else 0.5,
            fillColor=color,
            fillOpacity=0.55 if is_changed else 0.35,
            children=[dl.Tooltip(tooltip)],
        ))
        centers.append([meta["lat"], meta["lon"]])

    if not centers:
        return _empty_map()

    centers_arr = np.array(centers)
    map_center = [centers_arr[:, 0].mean(), centers_arr[:, 1].mean()]

    return dl.Map(
        center=map_center, zoom=12,
        style={"width": "100%", "height": "520px"},
        children=[
            dl.TileLayer(
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                attribution="© OpenStreetMap",
            ),
            dl.LayerGroup(polygons),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK 6: recommendations table
# ─────────────────────────────────────────────────────────────────────────────
@callback(
    Output("opt-table-wrap",  "children"),
    Output("opt-table-count", "children"),
    Input("opt-results-store", "data"),
)
def render_table(result):
    if not result:
        return _empty_table(), ""

    rows = [r for r in result.get("per_cell", [])
            if r["delta_p_db"] > 0 or r["delta_h_db"] > 0]
    rows.sort(key=lambda r: -r["excess_before"])

    if not rows:
        return html.Div(
            "Optimizer found no changes to recommend for this snapshot.",
            style={"display": "flex", "alignItems": "center",
                   "justifyContent": "center", "height": "520px",
                   "color": TEXT_MUTED, "fontSize": "13px"},
        ), ""

    # ── Build a plain-English action sentence for each cell ──────────────────
    def _action_text(r):
        parts = []
        if r["delta_p_db"] > 0:
            parts.append(f"cut power {int(r['delta_p_db'])} dB")
        if r["delta_h_db"] > 0:
            parts.append(f"raise handover margin {int(r['delta_h_db'])} dB")
        return " + ".join(parts) if parts else "no change"

    _status_label = {
        "fixed":     "Resolved",
        "improved":  "Improved",
        "no-effect": "Still congested",
        "adjusted":  "Tuned (was healthy)",
        "unchanged": "No change",
    }

    table_data = [{
        "Cell":        r["cell_name"],
        "Users now":   r["lambda"],
        "Capacity":    r["Lambda"],
        "Over by":     r["excess_before"],
        "Action":      _action_text(r),
        "Users moved": r["total_offload"],
        "Users after": r["effective_after"],
        "Result":      _status_label.get(r["status"], r["status"].title()),
    } for r in rows]

    # ── Legend explaining every column in plain language ─────────────────────
    legend = html.Div([
        html.Div("What each column means", style={
            "fontSize": "11px", "fontWeight": 700, "color": TEXT_PRIMARY,
            "marginBottom": "8px", "textTransform": "uppercase",
            "letterSpacing": "0.4px",
        }),
        html.Div([
            html.Div([
                html.Span("Users now", style={"fontWeight": 600, "color": TEXT_PRIMARY}),
                html.Span(" — active users on the cell at the optimized moment",
                          style={"color": TEXT_MUTED}),
            ]),
            html.Div([
                html.Span("Capacity", style={"fontWeight": 600, "color": TEXT_PRIMARY}),
                html.Span(" — users the cell can serve before hitting the congestion threshold (the model's Λ prediction)",
                          style={"color": TEXT_MUTED}),
            ]),
            html.Div([
                html.Span("Over by", style={"fontWeight": 600, "color": TEXT_PRIMARY}),
                html.Span(" — how many users the cell is above its capacity (0 means not congested)",
                          style={"color": TEXT_MUTED}),
            ]),
            html.Div([
                html.Span("Action", style={"fontWeight": 600, "color": TEXT_PRIMARY}),
                html.Span(" — what the optimizer recommends. Cutting power shrinks the cell's reach; raising handover margin pushes edge users to neighbors",
                          style={"color": TEXT_MUTED}),
            ]),
            html.Div([
                html.Span("Users moved", style={"fontWeight": 600, "color": TEXT_PRIMARY}),
                html.Span(" — how many users get offloaded to neighboring cells as a result",
                          style={"color": TEXT_MUTED}),
            ]),
            html.Div([
                html.Span("Users after", style={"fontWeight": 600, "color": TEXT_PRIMARY}),
                html.Span(" — users remaining on the cell once the change is applied",
                          style={"color": TEXT_MUTED}),
            ]),
            html.Div([
                html.Span("Result", style={"fontWeight": 600, "color": TEXT_PRIMARY}),
                html.Span(" — Resolved: congestion cleared · Improved: reduced but still over · Still congested: no effect · Tuned: cell was healthy, adjusted to absorb others",
                          style={"color": TEXT_MUTED}),
            ]),
        ], style={"display": "flex", "flexDirection": "column", "gap": "5px",
                  "fontSize": "11px", "lineHeight": "1.5"}),
    ], style={
        "padding": "12px 16px",
        "background": "#f8fafc",
        "borderBottom": f"1px solid {BORDER_COLOR}",
    })

    table = dash_table.DataTable(
        data=table_data,
        columns=[
            {"name": ["", "Cell"],                       "id": "Cell"},
            {"name": ["Current state", "Users now"],     "id": "Users now",
             "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": ["Current state", "Capacity"],      "id": "Capacity",
             "type": "numeric", "format": {"specifier": ".0f"}},
            {"name": ["Current state", "Over by"],       "id": "Over by",
             "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": ["Recommended action", "Action"],   "id": "Action"},
            {"name": ["Predicted outcome", "Users moved"], "id": "Users moved",
             "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": ["Predicted outcome", "Users after"], "id": "Users after",
             "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": ["Predicted outcome", "Result"],    "id": "Result"},
        ],
        merge_duplicate_headers=True,
        page_size=15,
        sort_action="native", filter_action="native",
        style_cell={
            "padding": "8px 12px", "fontSize": "12px",
            "fontFamily": "'Inter', sans-serif", "textAlign": "left",
            "border": f"1px solid {BORDER_COLOR}",
            "maxWidth": "200px", "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_header={
            "backgroundColor": "#f1f5f9", "fontWeight": 600,
            "fontSize": "11px", "color": TEXT_PRIMARY,
            "textAlign": "left",
        },
        style_data_conditional=[
            {"if": {"filter_query": "{Result} = 'Resolved'"},
             "backgroundColor": "#f0fdf4", "color": "#15803d"},
            {"if": {"filter_query": "{Result} = 'Improved'"},
             "backgroundColor": "#fffbeb", "color": "#b45309"},
            {"if": {"filter_query": "{Result} = 'Still congested'"},
             "backgroundColor": "#fef2f2", "color": "#dc2626"},
            {"if": {"column_id": "Action"},
             "fontStyle": "italic", "color": ACCENT},
        ],
        style_cell_conditional=[
            {"if": {"column_id": "Cell"}, "fontFamily": "monospace",
             "fontSize": "11px"},
            {"if": {"column_id": "Action"}, "minWidth": "180px"},
        ],
        style_table={"overflowY": "auto", "maxHeight": "440px"},
        tooltip_delay=0,
        tooltip_duration=None,
    )

    return html.Div([legend, table]), f"{len(rows)} cells changed"


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK 7: top relieved cells chart
# ─────────────────────────────────────────────────────────────────────────────
@callback(
    Output("opt-top-chart", "figure"),
    Input("opt-results-store", "data"),
)
def render_top_chart(result):
    if not result:
        return _empty_chart()

    rows = [r for r in result.get("per_cell", []) if r["excess_before"] > 0]
    if not rows:
        return _empty_chart()

    rows.sort(key=lambda r: -(r["excess_before"] - r["excess_after"]))
    top = rows[:10]
    names = [r["cell_name"][-15:] for r in top]   # last 15 chars to keep short
    before = [r["excess_before"] for r in top]
    after = [r["excess_after"] for r in top]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names, x=before, name="Before",
        orientation="h",
        marker_color="#fca5a5",
        hovertemplate="%{y}<br>Excess before: %{x:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=names, x=after, name="After",
        orientation="h",
        marker_color="#86efac",
        hovertemplate="%{y}<br>Excess after: %{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=350, barmode="overlay",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Excess users", yaxis_title=None,
        font={"family": "'Inter', sans-serif", "size": 11},
        legend=dict(orientation="h", y=1.06, x=1, xanchor="right"),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


# =============================================================================
# RF CLUSTER ANALYSIS CALLBACKS
# (reads from global filter, not the region/mode picker above)
# =============================================================================

from cluster_analyzer import analyse_cluster
from precompute import get_filtered_df as _rfa_get_filtered


# ── colour helpers ────────────────────────────────────────────────────────────
_RFA_SHORT  = "#15803d"   # green
_RFA_MEDIUM = "#b45309"   # amber
_RFA_LONG   = "#b91c1c"   # red
_RFA_BG     = {"short": "#f0fdf4", "medium": "#fffbeb", "long": "#fef2f2"}
_RFA_BORDER = {"short": "#86efac", "medium": "#fcd34d", "long": "#fca5a5"}
_RFA_DOT    = {"short": "🟢",      "medium": "🟡",       "long": "🔴"}

_ACTION_LABELS = {
    "freq_balance":              "Frequency-band rebalancing",
    "tilt_up":                   "Uptilt (TA too low)",
    "tilt_down":                 "Downtilt (TA too high)",
    "intra_site_neighbor_tilt":  "Intra-site sector relief",
    "inter_site_neighbor_tilt":  "Inter-site neighbour tilt",
    "add_band":                  "Deploy missing frequency band",
    "add_sector":                "Add new sector",
    "new_site":                  "New capacitive site",
}

_RECOVERY_COLOR = {
    "freq_balance":              "#6366f1",
    "tilt_up":                   "#0ea5e9",
    "tilt_down":                 "#0ea5e9",
    "intra_site_neighbor_tilt":  "#8b5cf6",
    "inter_site_neighbor_tilt":  "#8b5cf6",
    "add_band":                  "#f59e0b",
    "add_sector":                "#f97316",
    "new_site":                  "#ef4444",
}


def _rfa_badge(label: str, color: str) -> html.Span:
    return html.Span(label, style={
        "fontSize": "11px", "fontWeight": 600,
        "padding": "4px 10px", "borderRadius": "12px",
        "color": color,
        "background": color + "1A",
        "border": f"1px solid {color}33",
    })


def _rfa_stat_card(icon: str, label: str, value, color: str = TEXT_PRIMARY) -> html.Div:
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "16px", "marginRight": "6px"}),
            html.Span(label, style={"fontSize": "10px", "fontWeight": 600,
                                    "color": TEXT_MUTED, "textTransform": "uppercase",
                                    "letterSpacing": "0.4px"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        html.Div(str(value), style={
            "fontSize": "22px", "fontWeight": 700, "color": color,
        }),
    ], style={
        "padding": "12px 16px",
        "background": SURFACE,
        "borderRadius": "12px",
        "border": f"1px solid {BORDER_COLOR}",
    })


def _rfa_prb_bar(prb: float | None, width: str = "100%") -> html.Div:
    """Thin coloured PRB bar."""
    if prb is None:
        return html.Div()
    pct  = max(0.0, min(100.0, prb))
    clr  = P0_COLOR if pct >= 90 else (P1_COLOR if pct >= 80 else
           "#f59e0b" if pct >= 65 else "#22c55e")
    return html.Div(
        html.Div(style={
            "width": f"{pct}%", "height": "100%",
            "background": clr, "borderRadius": "3px",
            "transition": "width 0.4s ease",
        }),
        style={
            "height": "5px", "background": "#f1f5f9",
            "borderRadius": "3px", "marginTop": "4px",
            "width": width,
        },
    )


def _build_topology_panel(result: dict) -> html.Div:
    """
    Compact topology grid.
    One row per site: site name | sector azimuths | one chip per cell
    (frequency label + coloured PRB bar).  All sites fit in a scrollable 320 px box.
    """
    topology = result.get("topology", {})
    stats    = result.get("stats", {})
    if not topology:
        return html.Div("No topology data.", style={"color": TEXT_MUTED, "fontSize": "13px"})

    rows = []
    for sk, site in topology.items():
        # Determine worst level for the site header badge
        site_cells = [cn for sec in site["sectors"].values() for cn in sec["cells"]]
        worst = "healthy"
        for cn in site_cells:
            lv = stats.get(cn, {}).get("congestion_level", "no_data")
            if lv == "P0":
                worst = "P0"; break
            if lv == "P1" and worst != "P0":
                worst = "P1"

        hdr_color = (P0_COLOR if worst == "P0" else
                     P1_COLOR if worst == "P1" else
                     "#22c55e" if worst == "healthy" else "#94a3b8")

        # Build sector columns — each column = one sector direction
        sector_cols = []
        for sid, sector in sorted(site["sectors"].items(),
                                  key=lambda x: x[1]["azimuth"]):
            cell_chips = []
            for cn, info in sorted(sector["cells"].items(),
                                   key=lambda x: x[1]["frequency"]):
                s   = stats.get(cn, {})
                prb = s.get("avg_prb")
                lv  = s.get("congestion_level", "no_data")
                clr = (P0_COLOR if lv == "P0" else
                       P1_COLOR if lv == "P1" else
                       "#22c55e" if lv == "healthy" else "#94a3b8")
                pct = max(0.0, min(100.0, prb)) if prb is not None else 0.0
                freq_label = info["frequency"].upper() if info["frequency"] else "?"
                cell_chips.append(html.Div([
                    # Frequency label + PRB number on one line
                    html.Div([
                        html.Span(freq_label, style={
                            "fontSize": "9px", "fontWeight": 700,
                            "color": clr, "minWidth": "28px",
                        }),
                        html.Span(
                            f"{prb:.0f}%" if prb is not None else "—",
                            style={"fontSize": "10px", "color": TEXT_MUTED,
                                   "marginLeft": "3px"},
                        ),
                    ], style={"display": "flex", "alignItems": "center"}),
                    # Mini PRB bar
                    html.Div(
                        html.Div(style={
                            "width": f"{pct}%", "height": "100%",
                            "background": clr, "borderRadius": "2px",
                        }),
                        style={
                            "height": "3px", "background": "#f1f5f9",
                            "borderRadius": "2px", "marginTop": "2px",
                        },
                    ),
                ], style={
                    "padding": "3px 6px",
                    "background": clr + "0D",
                    "border": f"1px solid {clr}28",
                    "borderRadius": "5px",
                    "minWidth": "58px",
                }))

            sector_cols.append(html.Div([
                html.Span(f"{sector['azimuth']:.0f}°", style={
                    "fontSize": "9px", "fontWeight": 600, "color": TEXT_MUTED,
                    "display": "block", "marginBottom": "3px",
                    "textAlign": "center",
                }),
                html.Div(cell_chips, style={"display": "flex", "flexDirection": "column",
                                            "gap": "3px"}),
            ], style={"flex": "0 0 auto"}))

        rows.append(html.Div([
            # Site label with worst-level dot
            html.Div([
                html.Span("●", style={
                    "color": hdr_color, "fontSize": "10px", "marginRight": "5px",
                }),
                html.Span(site["site"], style={
                    "fontSize": "12px", "fontWeight": 700, "color": TEXT_PRIMARY,
                    "whiteSpace": "nowrap",
                }),
                html.Span(
                    f"  {site.get('region', '')}",
                    style={"fontSize": "10px", "color": TEXT_MUTED,
                           "marginLeft": "4px", "whiteSpace": "nowrap"},
                ),
            ], style={
                "minWidth": "130px", "maxWidth": "150px",
                "paddingRight": "10px",
                "borderRight": f"1px solid {BORDER_COLOR}",
                "display": "flex", "alignItems": "center",
                "flexWrap": "wrap",
            }),
            # Sector columns
            html.Div(sector_cols, style={
                "display": "flex", "gap": "10px",
                "flexWrap": "wrap", "flex": 1,
                "paddingLeft": "10px",
            }),
        ], style={
            "display": "flex", "alignItems": "flex-start",
            "padding": "8px 10px",
            "borderBottom": f"1px solid {BORDER_COLOR}",
            "gap": "0",
        }))

    return html.Div(rows, style={"fontFamily": "'Inter', monospace, sans-serif"})


# ── Per-priority panel builder ────────────────────────────────────────────────

def _build_priority_panel(recs: list[dict], priority: str) -> html.Div:
    """
    Build the content for one priority tab (short / medium / long).

    Layout inside the tab:
      ┌──────────────────────────────────────────────────────────────────┐
      │ SITE grouping header (site name + worst PRB + # actions)        │
      │  ┌── action row (compact, one line) ──────────────────────────┐ │
      │  │  [action chip]  [sector °]  [affected cells]  [→ delta]    │ │
      │  │  ▼ expand → full rationale + evidence                       │ │
      │  └────────────────────────────────────────────────────────────┘ │
      └──────────────────────────────────────────────────────────────────┘
    """
    pri_recs = [r for r in recs if r.get("priority") == priority]
    if not pri_recs:
        empty_msg = {
            "short":  "No short-term actions needed for this cluster.",
            "medium": "No medium-term actions identified.",
            "long":   "No long-term capacity changes recommended.",
        }
        return html.Div(
            html.Div(
                f"✅  {empty_msg.get(priority, 'No recommendations.')}",
                style={"padding": "18px 20px", "fontSize": "13px",
                       "color": "#64748b", "fontStyle": "italic"},
            )
        )

    # Color scheme per priority
    pri_cfg = {
        "short":  {"chip_bg": "#dcfce7", "chip_cl": "#15803d", "hdr_bg": "#f0fdf4",
                   "hdr_bdr": "#86efac"},
        "medium": {"chip_bg": "#fef9c3", "chip_cl": "#854d0e", "hdr_bg": "#fefce8",
                   "hdr_bdr": "#fde047"},
        "long":   {"chip_bg": "#fee2e2", "chip_cl": "#991b1b", "hdr_bg": "#fff1f2",
                   "hdr_bdr": "#fca5a5"},
    }
    cfg = pri_cfg.get(priority, pri_cfg["short"])

    # Group by site
    by_site: dict[str, list[dict]] = {}
    for r in pri_recs:
        s = r.get("site", "—")
        by_site.setdefault(s, []).append(r)

    site_blocks = []
    for site_name, site_recs in by_site.items():
        # Worst PRB across affected cells in this site's recs
        all_prbs = [
            r["evidence"].get("congested_prb") or
            r["evidence"].get("high_prb") or
            r["evidence"].get("avg_prb")
            for r in site_recs
            if isinstance(r.get("evidence"), dict)
        ]
        worst_prb = max((p for p in all_prbs if p is not None), default=None)
        prb_tag = (
            html.Span(
                f"max PRB {worst_prb:.0f}%",
                style={
                    "fontSize": "10px", "fontWeight": 700,
                    "padding": "1px 7px", "borderRadius": "10px",
                    "color": (P0_COLOR if worst_prb >= 90 else
                              P1_COLOR if worst_prb >= 80 else "#64748b"),
                    "background": (P0_COLOR + "18" if worst_prb >= 90 else
                                   P1_COLOR + "18" if worst_prb >= 80 else "#f1f5f9"),
                    "border": f"1px solid {'#fca5a5' if worst_prb >= 90 else '#fed7aa' if worst_prb >= 80 else '#e2e8f0'}",
                    "marginLeft": "8px",
                },
            ) if worst_prb is not None else html.Span()
        )

        # Individual action rows
        action_rows = []
        for rec_idx, r in enumerate(site_recs):
            label       = _ACTION_LABELS.get(r["action_type"],
                                             r["action_type"].replace("_", " ").title())
            delta       = r.get("suggested_delta", "")
            est_prb     = r.get("estimated_prb_after")
            rec_pct     = r.get("recovery_pct", 0)
            az          = r.get("sector_az")
            aff_cells   = r.get("affected_cells", [])
            helper_cells = r.get("helper_cells", [])
            ev          = r.get("evidence", {}) or {}

            # Build evidence detail lines
            ev_lines = []
            for k, v in ev.items():
                if v is None:
                    continue
                label_map = {
                    "avg_prb": "Avg PRB", "max_prb": "Max PRB",
                    "high_prb": "High-band PRB", "low_prb": "Low-band PRB",
                    "high_band": "High band", "low_band": "Low band",
                    "avg_ta_m": "Avg TA", "ta_threshold_m": "TA threshold",
                    "congested_prb": "Congested PRB", "helper_prb": "Helper PRB",
                    "headroom_pp": "Headroom", "distance_m": "Distance",
                    "relationship": "Relationship", "az_alignment_deg": "AZ alignment",
                    "recommended_az_deg": "Suggested azimuth",
                    "largest_gap_deg": "Largest gap",
                    "suggested_lat": "Suggest. lat", "suggested_lon": "Suggest. lon",
                    "offset_direction_deg": "Offset direction",
                    "max_ta_m": "Max TA",
                    "missing_band": "Missing band",
                    "present_bands": "Present bands",
                    "n_sectors": "Sectors on site",
                    "congested_sectors": "Congested sectors",
                }
                display_k = label_map.get(k, k.replace("_", " ").title())
                if isinstance(v, list):
                    v = ", ".join(str(x) for x in v)
                elif isinstance(v, float):
                    v = f"{v:.1f}"
                ev_lines.append(
                    html.Div([
                        html.Span(display_k + ": ",
                                  style={"color": TEXT_MUTED, "fontWeight": 500}),
                        html.Span(str(v),
                                  style={"color": TEXT_PRIMARY, "fontWeight": 600}),
                    ], style={"fontSize": "11px", "marginBottom": "2px"})
                )

            # Expandable detail using html.Details / html.Summary
            detail_block = html.Details([
                html.Summary(
                    "▸  Details & rationale",
                    style={
                        "fontSize": "11px", "color": ACCENT,
                        "cursor": "pointer", "fontWeight": 600,
                        "listStyle": "none", "outline": "none",
                        "padding": "4px 0",
                        "userSelect": "none",
                    },
                ),
                html.Div([
                    # Rationale paragraph
                    html.P(r.get("rationale", ""),
                           style={"fontSize": "12px", "color": TEXT_PRIMARY,
                                  "lineHeight": "1.6", "margin": "0 0 8px 0"}),
                    # Evidence chips
                    html.Div(ev_lines, style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fill, minmax(200px, 1fr))",
                        "gap": "2px 16px",
                        "padding": "8px 10px",
                        "background": "#f8fafc",
                        "borderRadius": "6px",
                        "border": f"1px solid {BORDER_COLOR}",
                    }) if ev_lines else html.Div(),
                    # Helper cells
                    html.Div([
                        html.Span("Sectors/cells to adjust: ",
                                  style={"fontSize": "11px", "fontWeight": 600,
                                         "color": TEXT_MUTED}),
                        html.Span(", ".join(helper_cells),
                                  style={"fontSize": "11px", "color": ACCENT,
                                         "fontStyle": "italic"}),
                    ], style={"marginTop": "6px"}) if helper_cells else html.Div(),
                ], style={"padding": "6px 0 2px 10px",
                          "borderLeft": f"2px solid {BORDER_COLOR}",
                          "marginTop": "4px"}),
            ])

            # Recovery badge
            rec_badge = html.Span(f"+{rec_pct}%", style={
                "fontSize": "10px", "fontWeight": 700,
                "padding": "1px 6px", "borderRadius": "8px",
                "color": "#15803d", "background": "#dcfce7",
                "border": "1px solid #86efac", "whiteSpace": "nowrap",
            })

            # Target PRB badge
            tgt_badge = (
                html.Span(f"→ ~{est_prb:.0f}%", style={
                    "fontSize": "10px", "fontWeight": 700,
                    "padding": "1px 6px", "borderRadius": "8px",
                    "color": ("#15803d" if est_prb < 80 else
                              "#b45309" if est_prb < 90 else "#b91c1c"),
                    "background": ("#dcfce7" if est_prb < 80 else
                                   "#fef9c3" if est_prb < 90 else "#fee2e2"),
                    "border": "1px solid #e2e8f0",
                    "whiteSpace": "nowrap",
                }) if est_prb is not None else html.Span()
            )

            # Affected cells pill (show first 2 + count)
            cell_pill = html.Span(
                (", ".join(aff_cells[:2]) + (f" +{len(aff_cells)-2}" if len(aff_cells) > 2 else "")),
                style={"fontSize": "10px", "color": TEXT_MUTED,
                       "fontFamily": "monospace"},
            ) if aff_cells else html.Span()

            action_rows.append(html.Div([
                # ── Compact summary row ───────────────────────────────────────
                html.Div([
                    # Action type chip
                    html.Span(label, style={
                        "fontSize": "11px", "fontWeight": 700,
                        "padding": "2px 8px", "borderRadius": "5px",
                        "color": cfg["chip_cl"],
                        "background": cfg["chip_bg"],
                        "whiteSpace": "nowrap", "flexShrink": 0,
                    }),
                    # Sector azimuth
                    html.Span(
                        f"  {az:.0f}°" if az is not None else "",
                        style={"fontSize": "11px", "color": TEXT_MUTED,
                               "fontWeight": 600, "whiteSpace": "nowrap"},
                    ),
                    # Affected cells
                    cell_pill,
                    html.Div(style={"flex": 1}),
                    # Delta
                    html.Span(delta, style={
                        "fontSize": "11px", "color": "#6366f1",
                        "fontStyle": "italic", "textAlign": "right",
                        "maxWidth": "220px",
                    }),
                    rec_badge,
                    tgt_badge,
                ], style={
                    "display": "flex", "alignItems": "center",
                    "gap": "8px", "flexWrap": "wrap",
                    "marginBottom": "4px",
                }),
                # ── Expandable detail ─────────────────────────────────────────
                detail_block,
            ], style={
                "padding": "10px 14px",
                "borderBottom": f"1px solid {BORDER_COLOR}",
                "background": "white",
            }))

        site_blocks.append(html.Div([
            # Site header
            html.Div([
                html.Span("📍", style={"marginRight": "6px"}),
                html.Span(site_name, style={
                    "fontWeight": 700, "fontSize": "13px",
                    "color": TEXT_PRIMARY,
                }),
                prb_tag,
                html.Div(style={"flex": 1}),
                html.Span(
                    f"{len(site_recs)} action{'s' if len(site_recs) != 1 else ''}",
                    style={"fontSize": "11px", "color": TEXT_MUTED,
                           "fontWeight": 500},
                ),
            ], style={
                "display": "flex", "alignItems": "center",
                "padding": "9px 14px",
                "background": cfg["hdr_bg"],
                "borderBottom": f"1px solid {cfg['hdr_bdr']}",
                "gap": "6px",
            }),
            html.Div(action_rows),
        ], style={
            "border": f"1px solid {cfg['hdr_bdr']}",
            "borderRadius": "8px",
            "overflow": "hidden",
            "marginBottom": "10px",
        }))

    return html.Div(site_blocks, style={"padding": "12px 14px"})


def _build_rec_cards(result: dict) -> tuple[html.Div, str]:
    """Legacy stub — no longer called; panels are now split by priority tab."""
    recs = result.get("recommendations", [])
    count_label = f"{len(recs)} recommendation{'s' if len(recs) != 1 else ''}"
    return html.Div(), count_label


def _build_recovery_chart(result: dict) -> go.Figure:
    """Horizontal bar chart: action type vs estimated recovery %."""
    recs = result.get("recommendations", [])
    if not recs:
        fig = go.Figure()
        fig.add_annotation(
            text="No recommendations to display",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 13, "color": TEXT_MUTED},
        )
        fig.update_layout(
            height=160, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis={"visible": False}, yaxis={"visible": False},
        )
        return fig

    # Aggregate: best recovery per action type
    agg: dict[str, dict] = {}
    for r in recs:
        at = r["action_type"]
        if at not in agg or r["recovery_pct"] > agg[at]["pct"]:
            agg[at] = {
                "label": _ACTION_LABELS.get(at, at.replace("_", " ").title()),
                "pct":   r["recovery_pct"],
                "color": _RECOVERY_COLOR.get(at, ACCENT),
                "pri":   r["priority"],
            }

    # Sort by priority then pct desc
    pri_order = {"short": 0, "medium": 1, "long": 2}
    sorted_items = sorted(agg.values(),
                          key=lambda x: (pri_order.get(x["pri"], 9), -x["pct"]))

    labels = [x["label"] for x in sorted_items]
    values = [x["pct"]   for x in sorted_items]
    colors = [x["color"] for x in sorted_items]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}<br>Est. recovery: %{x}%<extra></extra>",
    ))
    fig.update_layout(
        height=max(160, 38 * len(sorted_items) + 60),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        margin=dict(l=10, r=60, t=16, b=20),
        xaxis={"title": "Estimated PRB recovery (%)", "range": [0, 110],
               "gridcolor": "#f1f5f9"},
        yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
        font={"family": "'Inter', sans-serif", "size": 11},
        showlegend=False,
    )
    return fig


# ── RFA CALLBACK 1: filter summary pill ──────────────────────────────────────
@callback(
    Output("rfa-filter-summary", "children"),
    Input("region-filter",  "value"),
    Input("site-filter",    "value"),
    Input("cell-filter",    "value"),
    Input("mode-filter",    "value"),
    Input("date-filter",    "start_date"),
    Input("date-filter",    "end_date"),
)
def rfa_update_filter_summary(regions, sites, cells, modes, start, end):
    filtered = _rfa_get_filtered(regions, sites, cells, modes, start, end)
    if filtered.empty or "Cell Name" not in filtered.columns:
        return "⚠️  No cells match current filter"
    n = filtered["Cell Name"].nunique()
    if n == 0:
        return "⚠️  No cells match current filter"
    preview_cells = filtered["Cell Name"].dropna().unique()[:3].tolist()
    suffix = f" … +{n - 3} more" if n > 3 else ""
    return f"{n} cell{'s' if n != 1 else ''}: {', '.join(preview_cells)}{suffix}"


# ── RFA CALLBACK 2: run analysis ──────────────────────────────────────────────
@callback(
    Output("rfa-results-store",        "data"),
    Output("rfa-status-bar",           "children"),
    Input("rfa-run-btn",               "n_clicks"),
    State("region-filter",             "value"),
    State("site-filter",               "value"),
    State("cell-filter",               "value"),
    State("mode-filter",               "value"),
    State("date-filter",               "start_date"),
    State("date-filter",               "end_date"),
    State("rfa-radius-slider",         "value"),
    prevent_initial_call=True,
)
def rfa_run_analysis(n_clicks, regions, sites, cells, modes, start, end, radius_m):
    if not n_clicks:
        raise PreventUpdate

    filtered = _rfa_get_filtered(regions, sites, cells, modes, start, end)
    if filtered.empty or "Cell Name" not in filtered.columns:
        return None, [_badge("No cells match current filter", "#dc2626")]

    cell_list = filtered["Cell Name"].dropna().unique().tolist()
    if not cell_list:
        return None, [_badge("No cells available", "#dc2626")]

    try:
        result = analyse_cluster(
            cell_list        = cell_list,
            df               = filtered,
            cell_geo_meta    = CELL_GEO_META,
            cell_polygons    = CELL_POLYGONS,
            neighbor_radius_m = float(radius_m or 2000),
        )
    except Exception as exc:
        return None, [_badge(f"Analysis error: {str(exc)[:80]}", "#dc2626")]

    s = result.get("summary", {})
    status = [
        _badge(f"{s.get('n_sites', 0)} sites", ACCENT),
        _badge(f"{s.get('n_cells', 0)} cells", TEXT_PRIMARY),
        _badge(f"P0: {s.get('p0_cells', 0)}", P0_COLOR),
        _badge(f"P1: {s.get('p1_cells', 0)}", P1_COLOR),
        _badge(f"{result.get('n_recs', 0)} recommendations", "#6366f1"),
    ]

    # Store topology/stats as JSON-safe — convert keys to strings
    def _jsonify(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(i) for i in obj]
        if isinstance(obj, float) and (obj != obj):   # NaN
            return None
        return obj

    return _jsonify(result), status


# ── RFA CALLBACK 3: cluster stats cards ──────────────────────────────────────
@callback(
    Output("rfa-cluster-stats",   "children"),
    Output("rfa-warning-banner",  "children"),
    Input("rfa-results-store",    "data"),
)
def rfa_update_cluster_stats(result):
    if not result:
        return [], html.Div()

    s = result.get("summary", {})
    freq = s.get("freq_counts", {})

    freq_str = "  ·  ".join(
        f"{k.upper()}: {v}"
        for k, v in sorted(freq.items())
        if k not in ("", "unknown", "none")
    ) or "—"

    cards = [
        _rfa_stat_card("🏗️",  "Sites",           s.get("n_sites", 0),    TEXT_PRIMARY),
        _rfa_stat_card("📡",  "Sectors",         s.get("n_sectors", 0),  TEXT_PRIMARY),
        _rfa_stat_card("📶",  "Cells",           s.get("n_cells", 0),    TEXT_PRIMARY),
        _rfa_stat_card("🔴",  "P0 Cells",        s.get("p0_cells", 0),   P0_COLOR),
        _rfa_stat_card("🟠",  "P1 Cells",        s.get("p1_cells", 0),   P1_COLOR),
        _rfa_stat_card("✅",  "Healthy Cells",   s.get("healthy_cells", 0), "#15803d"),
        _rfa_stat_card("📻",  "Frequency bands", freq_str,               ACCENT),
    ]

    warn_div = html.Div()
    if result.get("warning"):
        warn_div = html.Div(
            f"⚠️  {result['warning']}",
            style={
                "padding": "10px 16px",
                "background": "#fffbeb", "border": "1px solid #fcd34d",
                "borderRadius": "10px", "fontSize": "12px",
                "color": "#b45309", "fontWeight": 600,
                "marginBottom": "10px",
            },
        )

    return cards, warn_div


# ── RFA CALLBACK 4: topology panel ───────────────────────────────────────────
@callback(
    Output("rfa-topology-panel", "children"),
    Input("rfa-results-store",   "data"),
)
def rfa_update_topology(result):
    if not result:
        return html.Div(
            "Run the analysis to see the cluster topology.",
            style={"color": TEXT_MUTED, "fontSize": "13px",
                   "padding": "10px 14px"},
        )
    return _build_topology_panel(result)


# ── RFA CALLBACK 5: three priority tab panels + count label ──────────────────
@callback(
    Output("rfa-panel-short",  "children"),
    Output("rfa-panel-medium", "children"),
    Output("rfa-panel-long",   "children"),
    Output("rfa-rec-count",    "children"),
    Output("rfa-export-bar",   "children"),
    Input("rfa-results-store", "data"),
)
def rfa_update_recommendations(result):
    empty_panel = html.Div(
        "Run the analysis to see recommendations.",
        style={"color": TEXT_MUTED, "fontSize": "13px",
               "padding": "18px 20px", "fontStyle": "italic"},
    )
    if not result:
        return empty_panel, empty_panel, empty_panel, "", html.Div()

    recs  = result.get("recommendations", [])
    n     = len(recs)
    count = f"{n} recommendation{'s' if n != 1 else ''}"

    n_short  = sum(1 for r in recs if r.get("priority") == "short")
    n_medium = sum(1 for r in recs if r.get("priority") == "medium")
    n_long   = sum(1 for r in recs if r.get("priority") == "long")

    # Update tab labels with counts — done via children of the panels,
    # tabs themselves are static in layout; count shown in header instead.
    count_full = (
        f"{n} total  ·  "
        f"🟢 {n_short} short  ·  "
        f"🟡 {n_medium} medium  ·  "
        f"🔴 {n_long} long"
    )

    short_panel  = _build_priority_panel(recs, "short")
    medium_panel = _build_priority_panel(recs, "medium")
    long_panel   = _build_priority_panel(recs, "long")

    export_bar = html.Div(
        html.Span(
            f"🟢 {n_short}  🟡 {n_medium}  🔴 {n_long}",
            style={"fontSize": "11px", "color": TEXT_MUTED, "fontWeight": 500},
        )
    )

    return short_panel, medium_panel, long_panel, count_full, export_bar


# ── RFA CALLBACK 6: recovery chart ───────────────────────────────────────────
@callback(
    Output("rfa-recovery-chart", "figure"),
    Input("rfa-results-store",   "data"),
)
def rfa_update_recovery_chart(result):
    if not result:
        fig = go.Figure()
        fig.add_annotation(
            text="Run the analysis to see the recovery chart",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font={"size": 13, "color": TEXT_MUTED},
        )
        fig.update_layout(
            height=160, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis={"visible": False}, yaxis={"visible": False},
        )
        return fig
    return _build_recovery_chart(result)
