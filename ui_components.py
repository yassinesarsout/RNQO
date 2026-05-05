"""
Reusable UI component builders — summary cards, congestion cards,
and the map side panel.
"""

import pandas as pd
from dash import html

from config import CARD_STYLE, CONGESTION_COLORS, NO_DATA_COLOR, P0_COLOR, P1_COLOR
from precompute import CELL_GEO_META


# ── Shared helpers ────────────────────────────────────────────────────────

def _kpi_row(label, value):
    return html.Div([
        html.Span(f"{label}: ", style={"fontSize": "12px", "color": "#475569"}),
        html.Span(str(value),  style={"fontSize": "12px", "color": "#0f172a", "fontWeight": 600}),
    ], style={"marginBottom": "3px"})


def _congestion_badge(level: str):
    color = CONGESTION_COLORS.get(level, NO_DATA_COLOR)
    return html.Span(level, style={
        "fontSize": "11px", "fontWeight": 700, "color": color,
        "background": color + "22",
        "border": f"1px solid {color}",
        "borderRadius": "6px", "padding": "1px 7px",
    })


# ── KPI analytics cards ───────────────────────────────────────────────────

def build_summary_cards(df: pd.DataFrame) -> list:
    metrics = [
        ("Rows",    f"{len(df):,}"),
        ("Regions", f"{df['Region'].nunique(dropna=True):,}"
                    if "Region" in df.columns else "0"),
        ("Sites",   f"{df['Radio Site'].nunique(dropna=True):,}"
                    if "Radio Site" in df.columns else "0"),
        ("Cells",   f"{df['Cell Name'].nunique(dropna=True):,}"
                    if "Cell Name" in df.columns else "0"),
        ("Avg DL PRB",
         f"{df['DL PRB Load'].mean():.2f}"
         if "DL PRB Load" in df.columns and not df.empty else "N/A"),
    ]
    return [
        html.Div([
            html.Div(label, style={"fontSize": "12px", "color": "#64748b", "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": 700, "color": "#0f172a"}),
        ], style=CARD_STYLE)
        for label, value in metrics
    ]


def build_congestion_cards(df: pd.DataFrame) -> list:
    if df.empty:
        metrics = [
            ("P0 Congested Rows",  "0"), ("P1 Congested Rows",  "0"),
            ("P0 Congested Cells", "0"), ("P1 Congested Cells", "0"),
            ("Avg Final Score",    "N/A"), ("Busy Hour Rows",   "0"),
        ]
    else:
        p0_rows  = int(df["P0_Congestion"].fillna(False).sum()) if "P0_Congestion" in df.columns else 0
        p0_cells = (df.loc[df["P0_Congestion"] == True, "Cell Name"].nunique()
                    if "P0_Congestion" in df.columns and "Cell Name" in df.columns else 0)
        p1_rows  = int(df["P1_Congestion"].fillna(False).sum()) if "P1_Congestion" in df.columns else 0
        p1_cells = (df.loc[df["P1_Congestion"] == True, "Cell Name"].nunique()
                    if "P1_Congestion" in df.columns and "Cell Name" in df.columns else 0)
        avg_score = (f"{df['FinalCongestionScore'].dropna().mean():.2f}"
                     if "FinalCongestionScore" in df.columns
                     and df["FinalCongestionScore"].notna().any() else "N/A")
        busy_rows = (int(df["IsBusyHourRow"].fillna(False).sum())
                     if "IsBusyHourRow" in df.columns else 0)
        metrics = [
            ("P0 Congested Rows",  f"{p0_rows:,}"),  ("P1 Congested Rows",  f"{p1_rows:,}"),
            ("P0 Congested Cells", f"{p0_cells:,}"), ("P1 Congested Cells", f"{p1_cells:,}"),
            ("Avg Final Score",    avg_score),        ("Busy Hour Rows",     f"{busy_rows:,}"),
        ]

    color_map = {
        "P0 Congested Rows":  P0_COLOR, "P1 Congested Rows":  P1_COLOR,
        "P0 Congested Cells": P0_COLOR, "P1 Congested Cells": P1_COLOR,
    }
    return [
        html.Div([
            html.Div(label, style={"fontSize": "12px", "color": "#64748b", "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "24px", "fontWeight": 700,
                                    "color": color_map.get(label, "#0f172a")}),
        ], style={
            **CARD_STYLE,
            "borderLeft": (f"4px solid {color_map[label]}"
                           if label in color_map else CARD_STYLE.get("border", "")),
        })
        for label, value in metrics
    ]


# ── KPI metric helpers ────────────────────────────────────────────────────

_KPI_COLS = [
    ("DL PRB Load",              "%",    "Avg PRB Load"),
    ("UL PRB Load",              "%",    "Avg UL PRB Load"),
    ("Active Users",             "",     "Avg Active Users"),
    ("LTE DL User Throughput",   "kbps", "Avg DL Throughput"),
    ("DL Avg Throughput",        "kbps", "Avg DL Throughput"),
    ("Average CQI",              "",     "Avg CQI"),
]

def _cell_kpi_block(cell_name: str, df: pd.DataFrame) -> list:
    """Return a list of _kpi_row() elements for the given cell's KPI data."""
    if df.empty or "Cell Name" not in df.columns:
        return []
    sub = df[df["Cell Name"] == cell_name]
    if sub.empty:
        return [html.Div("No KPI data for current filter.",
                         style={"fontSize": "11px", "color": "#94a3b8",
                                "marginTop": "6px"})]
    rows = []
    seen_labels = set()
    for col, unit, label in _KPI_COLS:
        if col in sub.columns and label not in seen_labels:
            val = sub[col].dropna()
            if val.empty:
                continue
            mean_val = val.mean()
            max_val  = val.max()
            suffix   = f" {unit}" if unit else ""
            rows.append(_kpi_row(label, f"{mean_val:.1f}{suffix}  (max {max_val:.1f}{suffix})"))
            seen_labels.add(label)

    # Congestion hours
    for flag_col, tier_label in [("P0_Congestion", "P0 hours"), ("P1_Congestion", "P1 hours")]:
        if flag_col in sub.columns:
            count = int(sub[flag_col].fillna(False).sum())
            if count > 0:
                rows.append(_kpi_row(tier_label, str(count)))

    if "BusyHour" in sub.columns:
        busy_vals = sub["BusyHour"].dropna().unique()
        if len(busy_vals) > 0:
            bh = ", ".join(f"{int(h):02d}:00" for h in sorted(busy_vals)[:3])
            rows.append(_kpi_row("Busy hours", bh))

    return rows


# ── Map side panel ────────────────────────────────────────────────────────

def build_map_panel(
    selected_sites: list,
    selected_cells: list,
    visible_cells,
    cong_lookup: dict,
    site_df: pd.DataFrame,
    filtered_df: pd.DataFrame | None = None,
) -> html.Div:
    """Build the right-hand info panel for the map tab."""

    # ── Cell-level selection ──────────────────────────────────────────────
    if selected_cells:
        cards = []
        for cn in selected_cells:
            meta = CELL_GEO_META.get(cn)
            if not meta:
                continue
            level      = cong_lookup.get(cn, "No Data")
            border_clr = CONGESTION_COLORS.get(level, NO_DATA_COLOR)
            kpi_rows   = _cell_kpi_block(cn, filtered_df if filtered_df is not None
                                         else pd.DataFrame())
            cards.append(html.Div([
                html.Div(cn, style={"fontWeight": 700, "fontSize": "13px",
                                     "color": "#0f172a", "marginBottom": "6px"}),
                _kpi_row("Site",     meta["site"]),
                _kpi_row("Region",   meta["region"]),
                _kpi_row("Azimuth",  f"{meta['azimuth']:.0f}°"),
                _kpi_row("Range",    f"{meta['range_m']:.0f} m"),
                _kpi_row("Coverage", meta["coverage"] or "—"),
                html.Div([
                    html.Span("Congestion: ",
                              style={"fontSize": "12px", "color": "#475569"}),
                    _congestion_badge(level),
                ], style={"marginTop": "4px", "marginBottom": "6px"}),
                # KPI metrics from the filtered DF
                html.Div([
                    html.Div("KPI metrics (filtered period)",
                             style={"fontSize": "11px", "fontWeight": 600,
                                    "color": "#64748b", "marginBottom": "4px",
                                    "borderTop": "1px dashed #e5e7eb",
                                    "paddingTop": "6px"}),
                    *kpi_rows,
                ]) if kpi_rows else html.Div(),
            ], style={
                "padding": "10px 12px",
                "border":  f"2px solid {border_clr}",
                "borderRadius": "10px", "marginBottom": "10px", "background": "#fafbff",
            }))
        return html.Div([
            html.Div(f"Selected cells ({len(selected_cells)})",
                     style={"fontSize": "13px", "fontWeight": 700, "marginBottom": "10px"}),
            *cards,
            html.P("Click the same cell again to deselect.",
                   style={"fontSize": "11px", "color": "#94a3b8", "marginTop": "6px"}),
        ])

    # ── Site-level selection ──────────────────────────────────────────────
    if selected_sites:
        sel_set = set(selected_sites)
        rows = site_df[site_df["SiteKey"].isin(sel_set)] if not site_df.empty else pd.DataFrame()
        cards = []
        for _, row in rows.iterrows():
            worst = row.get("congestion", "No Data")
            color = CONGESTION_COLORS.get(worst, NO_DATA_COLOR)
            site_cells = [c for c, m in CELL_GEO_META.items()
                          if m["site_key"] == row["SiteKey"] and c in visible_cells]
            bullets = []
            for cn in sorted(site_cells, key=lambda c: CELL_GEO_META[c]["azimuth"]):
                meta = CELL_GEO_META[cn]
                cl   = cong_lookup.get(cn, "No Data")
                cl_color = CONGESTION_COLORS.get(cl, NO_DATA_COLOR)
                bullets.append(html.Div([
                    html.Span("● ", style={"color": cl_color, "fontWeight": 700}),
                    html.Span(cn,   style={"fontSize": "12px", "color": "#0f172a"}),
                    html.Span(
                        f"  az {meta['azimuth']:.0f}°  {meta['range_m']:.0f} m",
                        style={"fontSize": "11px", "color": "#64748b"},
                    ),
                    html.Span(
                        f"  {cl}",
                        style={"fontSize": "11px", "fontWeight": 600,
                               "color": cl_color, "marginLeft": "4px"},
                    ),
                ], style={"marginBottom": "4px"}))
            cards.append(html.Div([
                html.Div(row["site_name"],
                         style={"fontWeight": 700, "color": "#0f172a", "marginBottom": "4px"}),
                _kpi_row("Region", row["region"]),
                _kpi_row("Cells",  int(row["cell_count"])),
                html.Div([
                    html.Span("Worst: ", style={"fontSize": "12px", "color": "#475569"}),
                    _congestion_badge(worst),
                ], style={"marginTop": "4px", "marginBottom": "6px"}),
                html.Div(bullets, style={
                    "borderTop": "1px dashed #e5e7eb", "paddingTop": "6px",
                }) if bullets else html.Div(),
            ], style={
                "padding": "10px 12px",
                "border": f"2px solid {color}",
                "borderRadius": "10px", "marginBottom": "10px", "background": "#fafbff",
            }))
        return html.Div([
            html.Div(f"Selected sites ({len(selected_sites)})",
                     style={"fontSize": "13px", "fontWeight": 700, "marginBottom": "10px"}),
            *cards,
            html.P("Click a sector cone to inspect a specific cell.",
                   style={"fontSize": "11px", "color": "#94a3b8", "marginTop": "4px"}),
        ])

    # ── Overview (nothing selected) ───────────────────────────────────────
    counts = {}
    if not site_df.empty and "congestion" in site_df.columns:
        counts = site_df["congestion"].value_counts().to_dict()

    legend_items = [
        ("P0 — Severe",        CONGESTION_COLORS["P0"]),
        ("P1 — Moderate",      CONGESTION_COLORS["P1"]),
        ("Not Congested",      CONGESTION_COLORS["Not Congested"]),
        ("No Data",            NO_DATA_COLOR),
    ]
    legend = html.Div([
        html.Div("Legend", style={"fontSize": "12px", "fontWeight": 600, "marginBottom": "6px"}),
        *[html.Div([
            html.Div(style={"width": "12px", "height": "12px", "borderRadius": "50%",
                            "background": c, "display": "inline-block", "marginRight": "8px",
                            "flexShrink": 0}),
            html.Span(label, style={"fontSize": "12px", "color": "#475569"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "5px"})
          for label, c in legend_items],
    ], style={"marginBottom": "14px"})

    return html.Div([
        html.Div("Map overview",
                 style={"fontSize": "13px", "fontWeight": 700, "marginBottom": "10px"}),
        _kpi_row("Visible sites",
                 f"{len(site_df):,}" if not site_df.empty else "0"),
        _kpi_row("P0 sites",       counts.get("P0", 0)),
        _kpi_row("P1 sites",       counts.get("P1", 0)),
        _kpi_row("Not congested",  counts.get("Not Congested", 0)),
        _kpi_row("No data",        counts.get("No Data", 0)),
        html.Div(style={"height": "12px"}),
        legend,
        html.Div("Tips", style={"fontSize": "12px", "fontWeight": 600, "marginBottom": "6px"}),
        html.Ul([
            html.Li("Clusters split into individual sites as you zoom in.",
                    style={"fontSize": "12px", "color": "#64748b", "marginBottom": "4px"}),
            html.Li("Sector cones appear automatically at zoom ≥ 13.",
                    style={"fontSize": "12px", "color": "#64748b", "marginBottom": "4px"}),
            html.Li("Click a site dot to select it and reveal its sectors.",
                    style={"fontSize": "12px", "color": "#64748b", "marginBottom": "4px"}),
            html.Li("Click a cone to select that specific cell.",
                    style={"fontSize": "12px", "color": "#64748b", "marginBottom": "4px"}),
            html.Li("Draw a polygon / rectangle to lasso-select multiple sites.",
                    style={"fontSize": "12px", "color": "#64748b"}),
        ], style={"paddingLeft": "18px", "margin": 0}),
    ])
