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
                    "No forecast data — run AI Forecasting first",
                    "#dc2626",
                )]
            state = get_forecasted_state(cells, records, DF)
        else:
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
    return result, [
        _badge(f"{result['n_cells']} cells", ACCENT),
        _badge(f"Mode: {'Forecasted' if mode == 'forecasted' else 'Current'}", TEXT_PRIMARY),
        badge,
    ]


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

    table_data = [{
        "Cell":     r["cell_name"],
        "λ":        r["lambda"],
        "Λ":        r["Lambda"],
        "Excess":   r["excess_before"],
        "Δ℘ dB":    int(r["delta_p_db"]) if r["delta_p_db"] > 0 else None,
        "Δℏ dB":    int(r["delta_h_db"]) if r["delta_h_db"] > 0 else None,
        "Offload":  r["total_offload"],
        "New λ":    r["effective_after"],
        "Status":   r["status"].upper(),
    } for r in rows]

    return dash_table.DataTable(
        data=table_data,
        columns=[
            {"name": "Cell",    "id": "Cell"},
            {"name": "λ",       "id": "λ",       "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Λ",       "id": "Λ",       "type": "numeric", "format": {"specifier": ".0f"}},
            {"name": "Excess",  "id": "Excess",  "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Δ℘ dB",   "id": "Δ℘ dB"},
            {"name": "Δℏ dB",   "id": "Δℏ dB"},
            {"name": "Offload", "id": "Offload", "type": "numeric", "format": {"specifier": ".2f"}},
            {"name": "New λ",   "id": "New λ",   "type": "numeric", "format": {"specifier": ".1f"}},
            {"name": "Status",  "id": "Status"},
        ],
        page_size=18,
        sort_action="native", filter_action="native",
        style_cell={
            "padding": "8px 10px", "fontSize": "12px",
            "fontFamily": "'Inter', sans-serif", "textAlign": "left",
            "border": f"1px solid {BORDER_COLOR}",
        },
        style_header={
            "backgroundColor": "#f8fafc", "fontWeight": 600,
            "fontSize": "11px", "textTransform": "uppercase",
            "letterSpacing": "0.4px", "color": TEXT_MUTED,
        },
        style_data_conditional=[
            {"if": {"filter_query": "{Status} = 'FIXED'"},     "color": "#15803d", "fontWeight": 500},
            {"if": {"filter_query": "{Status} = 'IMPROVED'"},  "color": "#b45309"},
            {"if": {"filter_query": "{Status} = 'NO-EFFECT'"}, "color": "#dc2626"},
        ],
        style_table={"overflowY": "auto", "maxHeight": "520px"},
    ), f"{len(rows)} cells"


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
