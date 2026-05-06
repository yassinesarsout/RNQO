"""
All Dash callbacks — registered via the global @callback decorator (Dash 2.x).
Import this module AFTER creating the app so the callbacks attach correctly.
"""

import numpy as np
import pandas as pd
import plotly.express as px
from dash import (
    Input, Output, State,
    callback, clientside_callback,
    ctx, no_update,
)
from dash.exceptions import PreventUpdate

from config import (
    CONGESTION_COLORS,
    DEFAULT_PRB_THRESHOLD, DEFAULT_THROUGHPUT_THRESHOLD, DEFAULT_ACTIVE_USER_THRESHOLD,
    MAX_SCATTER_POINTS, MAX_HIST_POINTS, P0_COLOR, P1_COLOR,
    SECTOR_ZOOM_THRESHOLD,
)
from data_utils import categorical_options, downsample, make_empty_figure, numeric_columns
from geo_utils import split_site_key, point_in_polygon
from precompute import (
    DF, CELL_GEO_META, SITE_SUMMARY,
    get_filtered_df, get_geo_visible_cells, get_cell_congestion_lookup,
    _df_filter_cache,
)
from map_builder import (
    build_sites_geojson, build_sectors_geojson,
    find_nearest_site, find_sector_at_point,
)
from ui_components import build_summary_cards, build_congestion_cards, build_map_panel


# ============================================================
# Clientside: track map zoom in a store
# ============================================================
clientside_callback(
    """function(ze) {
        if (ze == null) return window.dash_clientside.no_update;
        if (typeof ze === 'number') return ze;
        return ze.zoom || ze.newZoom || 8;
    }""",
    Output("map-zoom-store", "data"),
    Input("main-map", "zoom"),
)


# ============================================================
# Cascading dropdowns: Region → Site → Cell Name
# ============================================================
@callback(
    Output("site-filter", "options"),
    Output("cell-filter", "options"),
    Input("region-filter", "value"),
    Input("site-filter",   "value"),
)
def update_site_and_cell_options(selected_regions, selected_sites):
    if DF.empty:
        return [], []
    temp = DF
    if selected_regions and "Region" in temp.columns:
        temp = temp[temp["Region"].isin(selected_regions)]
    site_options = categorical_options(temp["Radio Site"]) if "Radio Site" in temp.columns else []
    if selected_sites and "Radio Site" in temp.columns:
        temp = temp[temp["Radio Site"].isin(selected_sites)]
    cell_options = categorical_options(temp["Cell Name"]) if "Cell Name" in temp.columns else []
    return site_options, cell_options


# ============================================================
# Map render — sites GeoJSON + info panel
# ============================================================
@callback(
    Output("sites-geojson",   "data"),
    Output("map-info-panel",  "children"),
    Input("region-filter",         "value"),
    Input("site-filter",           "value"),
    Input("cell-filter",           "value"),
    Input("mode-filter",           "value"),
    Input("date-filter",           "start_date"),
    Input("date-filter",           "end_date"),
    Input("selected-sites-store",  "data"),
    Input("selected-cells-store",  "data"),
    Input("animation-hour-store",  "data"),
)
def update_map_sites(
    regions, sites, cells, modes, start, end,
    sel_sites, sel_cells, animation_hour,
):
    visible  = get_geo_visible_cells(regions, sites, cells)
    cong     = get_cell_congestion_lookup(regions, sites, cells, modes, start, end, animation_hour)
    filtered = get_filtered_df(regions, sites, cells, modes, start, end)

    sites_geojson = build_sites_geojson(visible, cong, sel_sites, sel_cells)

    # Build a lightweight site_df for the panel
    if not SITE_SUMMARY.empty:
        rows = []
        for _, row in SITE_SUMMARY.iterrows():
            site_cells = [c for c in row["cells"] if c in visible]
            if not site_cells:
                continue
            levels = [cong.get(c) for c in site_cells if cong.get(c)]
            worst  = ("P0" if "P0" in levels
                      else "P1" if "P1" in levels
                      else "Not Congested" if levels
                      else "No Data")
            rows.append({**row.to_dict(), "congestion": worst})
        site_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    else:
        site_df = pd.DataFrame()

    panel = build_map_panel(
        sel_sites or [], sel_cells or [], visible, cong, site_df, filtered
    )
    return sites_geojson, panel


# ============================================================
# Map render — sectors GeoJSON (zoom-aware)
# ============================================================
@callback(
    Output("sectors-geojson", "data"),
    Input("region-filter",        "value"),
    Input("site-filter",          "value"),
    Input("cell-filter",          "value"),
    Input("mode-filter",          "value"),
    Input("date-filter",          "start_date"),
    Input("date-filter",          "end_date"),
    Input("selected-sites-store", "data"),
    Input("selected-cells-store", "data"),
    Input("animation-hour-store", "data"),
    Input("map-zoom-store",       "data"),
)
def update_map_sectors(
    regions, sites, cells, modes, start, end,
    sel_sites, sel_cells, animation_hour, zoom,
):
    current_zoom   = zoom or 8
    show_all       = current_zoom >= SECTOR_ZOOM_THRESHOLD
    show_selected  = bool(sel_sites or sel_cells)

    if not show_all and not show_selected:
        return {"type": "FeatureCollection", "features": []}

    visible  = get_geo_visible_cells(regions, sites, cells)
    cong     = get_cell_congestion_lookup(regions, sites, cells, modes, start, end, animation_hour)

    if not show_all and show_selected:
        # Only show sectors for selected sites / cells
        sel_site_set = set(sel_sites or [])
        sel_cell_set = set(sel_cells or [])
        visible = frozenset(
            cn for cn in visible
            if cn in CELL_GEO_META
            and (CELL_GEO_META[cn]["site_key"] in sel_site_set or cn in sel_cell_set)
        )

    return build_sectors_geojson(visible, cong, sel_sites, sel_cells)


# ============================================================
# Map click handler — site dot or sector cone
# ============================================================
def _site_keys_to_filter_vals(site_keys: list) -> tuple:
    regions, site_names = [], []
    for key in (site_keys or []):
        region, site = split_site_key(key)
        if region and region not in regions:       regions.append(region)
        if site   and site   not in site_names:    site_names.append(site)
    return sorted(regions), sorted(site_names)


@callback(
    Output("selected-sites-store", "data"),
    Output("selected-cells-store", "data"),
    Output("region-filter",        "value"),
    Output("site-filter",          "value"),
    Output("cell-filter",          "value"),
    Output("filter-update-source", "data"),
    Input("sites-geojson",   "clickData"),
    Input("sectors-geojson", "clickData"),
    State("selected-sites-store", "data"),
    State("selected-cells-store", "data"),
    State("cell-filter",           "value"),
    State("region-filter",         "value"),
    State("site-filter",           "value"),
    State("map-zoom-store",        "data"),
    prevent_initial_call=True,
)
def handle_map_click(
    site_click, sector_click,
    cur_sites, cur_cells, cur_cell_vals,
    cur_regions, cur_site_vals, zoom,
):
    # siteOnEachFeature / sectorOnEachFeature in map_functions.js inject
    # site_key / cell_name directly into clickData via context.setProps.
    cur_sites = cur_sites or []
    cur_cells = cur_cells or []
    triggered = ctx.triggered_id

    # ── Site marker clicked ───────────────────────────────────────────────
    if triggered == "sites-geojson" and site_click:
        # Primary: enriched clickData set by siteOnEachFeature
        site_key = site_click.get("site_key")

        # Fallback: coordinate-based lookup (cluster click or plain clickData)
        if not site_key:
            latlng = site_click.get("latlng") or []
            lat = latlng[0] if isinstance(latlng, (list, tuple)) and len(latlng) >= 2 \
                  else latlng.get("lat") if isinstance(latlng, dict) else None
            lon = latlng[1] if isinstance(latlng, (list, tuple)) and len(latlng) >= 2 \
                  else latlng.get("lng") if isinstance(latlng, dict) else None
            if lat is not None and lon is not None:
                visible = get_geo_visible_cells(cur_regions, cur_site_vals, cur_cell_vals)
                site_key = find_nearest_site(lat, lon, visible, zoom or 8)

        if not site_key:
            raise PreventUpdate

        if site_key in cur_sites:
            new_sites = [s for s in cur_sites if s != site_key]
        else:
            new_sites = sorted(set(cur_sites + [site_key]))
        regions, site_names = _site_keys_to_filter_vals(new_sites)
        return (new_sites, [], regions or no_update, site_names or no_update, [], "map")

    # ── Sector cone clicked ───────────────────────────────────────────────
    if triggered == "sectors-geojson" and sector_click:
        # Primary: enriched clickData set by sectorOnEachFeature
        cell_name = sector_click.get("cell_name")
        site_key  = sector_click.get("site_key")

        # Fallback: point-in-polygon lookup
        if not cell_name:
            latlng = sector_click.get("latlng") or []
            lat = latlng[0] if isinstance(latlng, (list, tuple)) and len(latlng) >= 2 \
                  else latlng.get("lat") if isinstance(latlng, dict) else None
            lon = latlng[1] if isinstance(latlng, (list, tuple)) and len(latlng) >= 2 \
                  else latlng.get("lng") if isinstance(latlng, dict) else None
            if lat is not None and lon is not None:
                visible = get_geo_visible_cells(cur_regions, cur_site_vals, cur_cell_vals)
                cell_name, site_key = find_sector_at_point(lat, lon, visible)

        if not cell_name:
            raise PreventUpdate

        new_sites = sorted(set(cur_sites + ([site_key] if site_key else [])))
        if cell_name in cur_cells:
            new_cells = [c for c in cur_cells if c != cell_name]
            if not new_cells and len(new_sites) == 1 and site_key in new_sites:
                return ([], [], no_update, [], [], "map")
        else:
            new_cells = sorted(set(cur_cells + [cell_name]))
        regions, site_names = _site_keys_to_filter_vals(new_sites)
        return (new_sites, new_cells,
                regions or no_update, site_names or no_update,
                new_cells or no_update, "map")

    raise PreventUpdate


# ============================================================
# Lasso / rectangle draw → select sites within shape
# ============================================================
@callback(
    Output("selected-sites-store", "data",  allow_duplicate=True),
    Output("selected-cells-store", "data",  allow_duplicate=True),
    Output("region-filter",        "value", allow_duplicate=True),
    Output("site-filter",          "value", allow_duplicate=True),
    Output("cell-filter",          "value", allow_duplicate=True),
    Output("filter-update-source", "data",  allow_duplicate=True),
    Input("lasso-control", "geojson"),
    prevent_initial_call=True,
)
def handle_lasso(geojson):
    if not geojson or not geojson.get("features"):
        raise PreventUpdate

    drawn = geojson["features"][-1]
    geom  = drawn.get("geometry", {})
    gtype = geom.get("type", "")

    # Extract outer ring coordinates from Polygon or convert Rectangle bbox
    if gtype == "Polygon":
        ring = geom["coordinates"][0]
    elif gtype == "Rectangle" or (gtype == "Polygon" and len(geom["coordinates"][0]) == 5):
        ring = geom["coordinates"][0]
    else:
        raise PreventUpdate

    if SITE_SUMMARY.empty:
        raise PreventUpdate

    selected = []
    for _, row in SITE_SUMMARY.iterrows():
        if point_in_polygon(float(row["lat"]), float(row["lon"]), ring):
            selected.append(row["SiteKey"])

    if not selected:
        raise PreventUpdate

    regions, site_names = _site_keys_to_filter_vals(selected)
    return (sorted(selected), [],
            regions or no_update, site_names or no_update, [], "map")


# ============================================================
# Filter bar changes → clear map selection (when driven from UI)
# ============================================================
@callback(
    Output("selected-sites-store", "data", allow_duplicate=True),
    Output("selected-cells-store", "data", allow_duplicate=True),
    Input("region-filter", "value"),
    Input("site-filter",   "value"),
    Input("cell-filter",   "value"),
    State("filter-update-source", "data"),
    prevent_initial_call=True,
)
def sync_map_selection_from_filters(regions, sites, cells, source):
    _ = (regions, sites, cells)   # trigger inputs; values not inspected
    if source == "map":
        return no_update, no_update
    return [], []


# ============================================================
# Reset map
# ============================================================
@callback(
    Output("selected-sites-store",    "data",     allow_duplicate=True),
    Output("selected-cells-store",    "data",     allow_duplicate=True),
    Output("region-filter",           "value",    allow_duplicate=True),
    Output("site-filter",             "value",    allow_duplicate=True),
    Output("cell-filter",             "value",    allow_duplicate=True),
    Output("animation-hour-store",    "data",     allow_duplicate=True),
    Output("hour-slider",             "value",    allow_duplicate=True),
    Output("hour-display",            "children", allow_duplicate=True),
    Output("animation-playing-store", "data",     allow_duplicate=True),
    Output("anim-interval",           "disabled", allow_duplicate=True),
    Output("anim-play-btn",           "children", allow_duplicate=True),
    Output("filter-update-source",    "data",     allow_duplicate=True),
    Output("map-reset-token",         "data"),
    Input("map-reset-btn", "n_clicks"),
    State("map-reset-token", "data"),
    prevent_initial_call=True,
)
def reset_map(n_clicks, token):
    if not n_clicks:
        raise PreventUpdate
    _df_filter_cache.clear()
    return (
        [],            # selected-sites-store
        [],            # selected-cells-store
        [],            # region-filter
        [],            # site-filter
        [],            # cell-filter
        0,             # animation-hour-store
        0,             # hour-slider
        "Hour 00:00",  # hour-display
        False,         # animation-playing-store
        True,          # anim-interval disabled
        "▶ Play",      # anim-play-btn
        "filters",     # filter-update-source
        (token or 0) + 1,  # map-reset-token (bumping forces map re-center)
    )


# ============================================================
# Animation controls
# ============================================================
@callback(
    Output("anim-interval",           "disabled"),
    Output("anim-play-btn",           "children"),
    Output("animation-playing-store", "data"),
    Input("anim-play-btn", "n_clicks"),
    State("animation-playing-store", "data"),
    prevent_initial_call=True,
)
def toggle_animation(n, playing):
    _ = n   # n_clicks triggers the callback; value not inspected
    now_playing = not playing
    return (not now_playing), ("⏸ Pause" if now_playing else "▶ Play"), now_playing


@callback(
    Output("animation-hour-store", "data"),
    Output("hour-slider",          "value"),
    Output("hour-display",         "children"),
    Input("anim-interval",  "n_intervals"),
    Input("hour-slider",    "value"),
    Input("anim-reset-btn", "n_clicks"),
    State("animation-hour-store",    "data"),
    State("animation-playing-store", "data"),
    prevent_initial_call=True,
)
def update_animation_hour(n_intervals, slider_val, reset_clicks, current_hour, playing):
    _ = (n_intervals, reset_clicks)   # trigger inputs; values not inspected directly
    triggered = ctx.triggered_id
    if triggered == "anim-reset-btn":
        return 0, 0, "Hour 00:00"
    if triggered == "anim-interval" and playing:
        next_hour = ((current_hour or 0) + 1) % 24
        return next_hour, next_hour, f"Hour {next_hour:02d}:00"
    if triggered == "hour-slider":
        return slider_val, slider_val, f"Hour {slider_val:02d}:00"
    return current_hour, (current_hour or 0), f"Hour {(current_hour or 0):02d}:00"


# ============================================================
# Main analytics callback — KPI tab + Congestion tab
# ============================================================
@callback(
    Output("summary-cards",               "children"),
    Output("timeseries-chart",            "figure"),
    Output("timeseries-chart-2",          "figure"),
    Output("timeseries-chart-3",          "figure"),
    Output("histogram-chart",             "figure"),
    Output("top-cells-chart",             "figure"),
    Output("scatter-chart",               "figure"),
    Output("correlation-chart",           "figure"),
    Output("detail-table",                "data"),
    Output("detail-table",                "columns"),
    Output("congestion-cards",            "children"),
    Output("congestion-level-chart",      "figure"),
    Output("congestion-timeseries-chart", "figure"),
    Output("top-congested-cells-chart",   "figure"),
    Output("busy-hour-chart",             "figure"),
    Output("congested-scatter-chart",     "figure"),
    Output("congestion-table",            "data"),
    Output("congestion-table",            "columns"),
    Input("region-filter",  "value"),
    Input("site-filter",    "value"),
    Input("cell-filter",    "value"),
    Input("mode-filter",    "value"),
    Input("date-filter",    "start_date"),
    Input("date-filter",    "end_date"),
    Input("main-kpi",       "value"),
    Input("main-kpi-2",     "value"),
    Input("main-kpi-3",     "value"),
    Input("hist-kpi",       "value"),
    Input("top-kpi",        "value"),
    Input("scatter-x",      "value"),
    Input("scatter-y",      "value"),
    Input("scatter-color",  "value"),
)
def update_dashboard(
    selected_regions, selected_sites, selected_cells, selected_modes,
    start_date, end_date,
    main_kpi, main_kpi_2, main_kpi_3,
    hist_kpi, top_kpi,
    scatter_x, scatter_y, scatter_color,
):
    if DF.empty:
        empty = make_empty_figure("No data")
        return ([], empty, empty, empty, empty, empty, empty, empty,
                [], [], [], empty, empty, empty, empty, empty, [], [])

    filtered = get_filtered_df(
        selected_regions, selected_sites, selected_cells,
        selected_modes, start_date, end_date,
    )
    cards = build_summary_cards(filtered)

    # ---- Time-series ----
    def _timeseries(kpi_name, suffix=""):
        if filtered.empty or kpi_name is None or kpi_name not in filtered.columns:
            return make_empty_figure("Time Series")
        n_sel = len(selected_cells) if selected_cells else 0
        show_cells = 1 <= n_sel <= 10 and "Cell Name" in filtered.columns
        if show_cells:
            plot_df = (filtered.groupby(["Time", "Cell Name"], as_index=False)[kpi_name]
                       .mean().sort_values("Time"))
            fig = px.line(plot_df, x="Time", y=kpi_name, color="Cell Name",
                          title=f"{kpi_name} over Time{suffix}")
        else:
            agg = (filtered.groupby(["Time"], as_index=False)[kpi_name]
                   .mean().sort_values("Time"))
            fig = px.line(agg, x="Time", y=kpi_name,
                          title=f"{kpi_name} over Time{suffix} (Average)")
        threshold = None
        if kpi_name == "DL PRB Load":
            threshold = DEFAULT_PRB_THRESHOLD
        elif kpi_name in {"DL Avg Throughput", "LTE DL User Throughput"}:
            threshold = DEFAULT_THROUGHPUT_THRESHOLD
        elif kpi_name == "Active Users":
            threshold = DEFAULT_ACTIVE_USER_THRESHOLD
        if threshold is not None:
            fig.add_hline(y=threshold, line_dash="dash",
                          annotation_text=f"Threshold {threshold}")
        fig.update_layout(template="plotly_white")
        return fig

    fig_time   = _timeseries(main_kpi,   " · Panel 1")
    fig_time_2 = _timeseries(main_kpi_2, " · Panel 2")
    fig_time_3 = _timeseries(main_kpi_3, " · Panel 3")

    # ---- Histogram ----
    if filtered.empty or hist_kpi is None or hist_kpi not in filtered.columns:
        fig_hist = make_empty_figure("Distribution")
    else:
        hist_df = downsample(filtered[[hist_kpi]].dropna(), MAX_HIST_POINTS)
        fig_hist = px.histogram(hist_df, x=hist_kpi, nbins=40,
                                title=f"Distribution of {hist_kpi}")
        fig_hist.update_layout(template="plotly_white")

    # ---- Top cells ----
    if (filtered.empty or top_kpi is None or top_kpi not in filtered.columns
            or "Cell Name" not in filtered.columns):
        fig_top = make_empty_figure("Top Cells")
    else:
        top = (filtered.groupby("Cell Name", as_index=False)[top_kpi]
               .mean().sort_values(top_kpi, ascending=False).head(15))
        fig_top = px.bar(top, x=top_kpi, y="Cell Name", orientation="h",
                         title=f"Top 15 Cells by Avg {top_kpi}")
        fig_top.update_layout(template="plotly_white",
                              yaxis={"categoryorder": "total ascending"})

    # ---- Scatter ----
    if filtered.empty or scatter_x not in filtered.columns or scatter_y not in filtered.columns:
        fig_scatter = make_empty_figure("Scatter")
    else:
        scatter_cols = [scatter_x, scatter_y, "Cell Name", "Region", "Radio Site", "Time"]
        color_arg = None
        if scatter_color and scatter_color != "__none__" and scatter_color in filtered.columns:
            scatter_cols.append(scatter_color)
            color_arg = scatter_color
        elif "CongestionLevel" in filtered.columns:
            color_arg = "CongestionLevel"
        scatter_df = downsample(
            filtered[[c for c in scatter_cols if c in filtered.columns]]
            .dropna(subset=[scatter_x, scatter_y]),
            MAX_SCATTER_POINTS,
        )
        fig_scatter = px.scatter(
            scatter_df, x=scatter_x, y=scatter_y, color=color_arg,
            hover_data=[c for c in ["Cell Name", "Region", "Radio Site", "Time",
                                     "FinalCongestionScore", "CongestionLevel",
                                     "P0_Congestion", "P1_Congestion"]
                        if c in scatter_df.columns],
            title=f"{scatter_y} vs {scatter_x}",
        )
        fig_scatter.update_layout(template="plotly_white")

    # ---- Correlation heatmap ----
    num_cols = [c for c in numeric_columns(filtered)
                if c not in {"LocalCell Id", "Radio Site", "Hour"}]
    preferred = ["DL PRB Load", "UL PRB Load", "Active Users",
                 "DL Avg Throughput", "LTE DL User Throughput", "Average CQI"]
    corr_cols = [c for c in preferred if c in num_cols] or num_cols[:12]
    if len(corr_cols) < 2 or filtered.empty:
        fig_corr = make_empty_figure("Correlation")
    else:
        corr = filtered[corr_cols].corr(numeric_only=True)
        fig_corr = px.imshow(corr, aspect="auto", title="Correlation Heatmap")
        fig_corr.update_layout(template="plotly_white")

    # ---- Detail table ----
    display_cols = [c for c in [
        "Time", "Region", "Radio Site", "eNodeB Name", "Cell Name", "LocalCell Id",
        "Cell FDD TDD Indication", "DL PRB Load", "UL PRB Load", "Connected Users",
        "Active Users", "DL Avg Throughput", "LTE DL User Throughput", "Average CQI",
        "UL Interference", "Drop Call Rate",
        "P0_Congestion", "P1_Congestion",
        "CongestionLevelHour", "Hourly_Congestion_Score_0_1", "Hourly_Congestion_Score",
        "BusyHour", "BusyHour_Score", "BusyHour_Level", "IsBusyHourRow",
        "FinalCongestionScore", "CongestionLevel",
        "DaysObserved", "DaysP0", "DaysP1", "MaxConsecutiveCongestedDays",
    ] if c in filtered.columns]
    table_df      = filtered[display_cols].copy().head(5000)
    table_data    = table_df.to_dict("records")
    table_columns = [{"name": c, "id": c} for c in table_df.columns]

    # ============================================================
    # Congestion tab charts
    # ============================================================
    congestion_cards = build_congestion_cards(filtered)
    has_p0 = "P0_Congestion" in filtered.columns
    has_p1 = "P1_Congestion" in filtered.columns

    # P0 vs P1 bar
    if filtered.empty or (not has_p0 and not has_p1):
        fig_cong_level = make_empty_figure("P0 / P1 Congestion Comparison")
    else:
        level_rows = []
        if "Cell Name" in filtered.columns:
            if has_p0:
                level_rows.append({"Tier": "P0 (Severe)",
                                    "Congested Cells": filtered[filtered["P0_Congestion"] == True]["Cell Name"].nunique()})
            if has_p1:
                level_rows.append({"Tier": "P1 (Moderate)",
                                    "Congested Cells": filtered[filtered["P1_Congestion"] == True]["Cell Name"].nunique()})
        if not level_rows:
            fig_cong_level = make_empty_figure("P0 / P1 Congestion Comparison")
        else:
            level_df = pd.DataFrame(level_rows)
            fig_cong_level = px.bar(
                level_df, x="Tier", y="Congested Cells",
                title="Distinct Congested Cells by Tier (P0 vs P1)",
                color="Tier",
                color_discrete_map={"P0 (Severe)": P0_COLOR, "P1 (Moderate)": P1_COLOR},
                text="Congested Cells",
            )
            fig_cong_level.update_traces(textposition="outside")
            fig_cong_level.update_layout(template="plotly_white", showlegend=False)

    # P0 vs P1 timeseries
    if filtered.empty or "Time" not in filtered.columns or (not has_p0 and not has_p1):
        fig_cong_time = make_empty_figure("P0 / P1 Congestion Over Time")
    else:
        time_traces = []
        if has_p0:
            p0t = (filtered.groupby("Time", as_index=False)["P0_Congestion"].sum()
                   .rename(columns={"P0_Congestion": "Congested Rows"}).sort_values("Time"))
            p0t["Tier"] = "P0 (Severe)"
            time_traces.append(p0t)
        if has_p1:
            p1t = (filtered.groupby("Time", as_index=False)["P1_Congestion"].sum()
                   .rename(columns={"P1_Congestion": "Congested Rows"}).sort_values("Time"))
            p1t["Tier"] = "P1 (Moderate)"
            time_traces.append(p1t)
        if not time_traces:
            fig_cong_time = make_empty_figure("P0 / P1 Congestion Over Time")
        else:
            combined = pd.concat(time_traces, ignore_index=True)
            fig_cong_time = px.line(
                combined, x="Time", y="Congested Rows", color="Tier",
                title="Congested Rows Over Time — P0 vs P1",
                color_discrete_map={"P0 (Severe)": P0_COLOR, "P1 (Moderate)": P1_COLOR},
            )
            fig_cong_time.update_layout(template="plotly_white")

    # Top congested cells
    if (filtered.empty or "Cell Name" not in filtered.columns
            or "FinalCongestionScore" not in filtered.columns):
        fig_top_cong = make_empty_figure("Top Congested Cells")
    else:
        top_cong = (filtered[["Cell Name", "FinalCongestionScore"]]
                    .dropna().drop_duplicates()
                    .groupby("Cell Name", as_index=False)["FinalCongestionScore"].max()
                    .sort_values("FinalCongestionScore", ascending=False).head(15))
        if top_cong.empty:
            fig_top_cong = make_empty_figure("Top Congested Cells")
        else:
            if has_p0 and has_p1:
                cell_tier = (filtered[["Cell Name", "P0_Congestion", "P1_Congestion"]]
                             .dropna(subset=["Cell Name"])
                             .groupby("Cell Name", as_index=False)
                             .agg({"P0_Congestion": "any", "P1_Congestion": "any"}))
                cell_tier["Tier"] = np.where(
                    cell_tier["P0_Congestion"], "P0 (Severe)",
                    np.where(cell_tier["P1_Congestion"], "P1 (Moderate)", "Not Congested"),
                )
                top_cong = top_cong.merge(cell_tier[["Cell Name", "Tier"]],
                                          on="Cell Name", how="left")
                top_cong["Tier"] = top_cong["Tier"].fillna("Not Congested")
                color_arg = "Tier"
                cmap = {"P0 (Severe)": P0_COLOR, "P1 (Moderate)": P1_COLOR,
                        "Not Congested": CONGESTION_COLORS["Not Congested"]}
            else:
                color_arg, cmap = None, {}
            fig_top_cong = px.bar(
                top_cong, x="FinalCongestionScore", y="Cell Name", orientation="h",
                title="Top 15 Cells by Final Congestion Score",
                color=color_arg,
                color_discrete_map=cmap if cmap else None,
            )
            fig_top_cong.update_layout(template="plotly_white",
                                       yaxis={"categoryorder": "total ascending"})

    # Busy hour distribution
    if filtered.empty or "BusyHour" not in filtered.columns:
        fig_busy_hour = make_empty_figure("Busy Hour Distribution")
    else:
        busy_df = (filtered[filtered["IsBusyHourRow"] == True].copy()
                   if "IsBusyHourRow" in filtered.columns else filtered.copy())
        if busy_df.empty:
            fig_busy_hour = make_empty_figure("Busy Hour Distribution")
        elif "BusyHour_Level" in busy_df.columns:
            hour_counts = (busy_df.groupby(["BusyHour", "BusyHour_Level"], as_index=False)
                           .size().rename(columns={"size": "Count"}).sort_values("BusyHour"))
            fig_busy_hour = px.bar(
                hour_counts, x="BusyHour", y="Count", color="BusyHour_Level",
                title="Busy Hour Distribution by Level",
                color_discrete_map={"P0": P0_COLOR, "P1": P1_COLOR,
                                    "Not Congested": CONGESTION_COLORS["Not Congested"]},
            )
            fig_busy_hour.update_layout(template="plotly_white")
        else:
            hour_counts = (busy_df.groupby("BusyHour", as_index=False).size()
                           .rename(columns={"size": "Count"}).sort_values("BusyHour"))
            fig_busy_hour = px.bar(hour_counts, x="BusyHour", y="Count",
                                   title="Busy Hour Distribution")
            fig_busy_hour.update_layout(template="plotly_white")

    # PRB vs Throughput scatter
    throughput_col = next(
        (c for c in ["LTE DL User Throughput", "DL Avg Throughput"] if c in filtered.columns), None)
    if filtered.empty or "DL PRB Load" not in filtered.columns or throughput_col is None:
        fig_cong_scatter = make_empty_figure("PRB vs Throughput")
    else:
        cdf = filtered.copy()
        if has_p0 and has_p1:
            cdf["CongestionFlag"] = np.select(
                [cdf["P0_Congestion"] == True, cdf["P1_Congestion"] == True],
                ["P0 (Severe)", "P1 (Moderate)"], default="Not Congested")
        elif has_p1:
            cdf["CongestionFlag"] = np.where(cdf["P1_Congestion"] == True, "P1 (Moderate)", "Not Congested")
        elif has_p0:
            cdf["CongestionFlag"] = np.where(cdf["P0_Congestion"] == True, "P0 (Severe)", "Not Congested")
        else:
            cdf["CongestionFlag"] = "Not Congested"
        cdf = downsample(
            cdf[[c for c in ["DL PRB Load", throughput_col, "Cell Name", "Time", "CongestionFlag"]
                 if c in cdf.columns]].dropna(subset=["DL PRB Load", throughput_col]),
            MAX_SCATTER_POINTS,
        )
        fig_cong_scatter = px.scatter(
            cdf, x="DL PRB Load", y=throughput_col, color="CongestionFlag",
            color_discrete_map={"P0 (Severe)": P0_COLOR, "P1 (Moderate)": P1_COLOR,
                                 "Not Congested": CONGESTION_COLORS["Not Congested"]},
            hover_data=[c for c in ["Cell Name", "Time"] if c in cdf.columns],
            title=f"{throughput_col} vs DL PRB Load — P0 / P1 Congestion",
        )
        fig_cong_scatter.update_layout(template="plotly_white")
        fig_cong_scatter.add_vline(x=DEFAULT_PRB_THRESHOLD, line_dash="dash",
                                    annotation_text=f"PRB threshold {DEFAULT_PRB_THRESHOLD}")
        fig_cong_scatter.add_hline(y=DEFAULT_THROUGHPUT_THRESHOLD, line_dash="dash",
                                    annotation_text=f"Throughput threshold {DEFAULT_THROUGHPUT_THRESHOLD}")

    # Congestion table
    cong_cols = [c for c in [
        "Time", "Region", "Radio Site", "Cell Name",
        "DL PRB Load", "Active Users", "LTE DL User Throughput", "DL Avg Throughput",
        "P0_Congestion", "P1_Congestion",
        "CongestionLevelHour", "Hourly_Congestion_Score_0_1", "Hourly_Congestion_Score",
        "BusyHour", "BusyHour_Score", "BusyHour_Level", "IsBusyHourRow",
        "FinalCongestionScore", "CongestionLevel",
        "DaysObserved", "DaysP0", "DaysP1", "MaxConsecutiveCongestedDays",
    ] if c in filtered.columns]

    if has_p0 and has_p1:
        mask = (filtered["P0_Congestion"] == True) | (filtered["P1_Congestion"] == True)
        cong_table_df = filtered[mask][cong_cols].copy().head(5000)
    elif has_p1:
        cong_table_df = filtered[filtered["P1_Congestion"] == True][cong_cols].copy().head(5000)
    elif has_p0:
        cong_table_df = filtered[filtered["P0_Congestion"] == True][cong_cols].copy().head(5000)
    else:
        cong_table_df = filtered[cong_cols].copy().head(5000)

    return (
        cards, fig_time, fig_time_2, fig_time_3, fig_hist, fig_top, fig_scatter, fig_corr,
        table_data, table_columns,
        congestion_cards, fig_cong_level, fig_cong_time, fig_top_cong, fig_busy_hour,
        fig_cong_scatter,
        cong_table_df.to_dict("records"),
        [{"name": c, "id": c} for c in cong_table_df.columns],
    )
