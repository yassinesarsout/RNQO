"""
Reusable UI component builders — summary cards, congestion cards,
and the map side panel.
"""

import pandas as pd
from dash import html

from config import (
    CARD_STYLE, CONGESTION_COLORS, NO_DATA_COLOR, P0_COLOR, P1_COLOR,
    ACCENT, TEXT_MUTED, TEXT_PRIMARY, BORDER_COLOR,
)
from precompute import CELL_GEO_META


# ── Shared helpers ─────────────────────────────────────────────────────────

def _row(label, value, muted=False):
    return html.Div([
        html.Span(label, style={"fontSize": "11px", "color": TEXT_MUTED,
                                "minWidth": "90px", "display": "inline-block"}),
        html.Span(str(value), style={"fontSize": "12px",
                                     "color": TEXT_MUTED if muted else TEXT_PRIMARY,
                                     "fontWeight": 500}),
    ], style={"marginBottom": "4px", "display": "flex", "alignItems": "center"})


def _badge(level: str):
    color = CONGESTION_COLORS.get(level, NO_DATA_COLOR)
    return html.Span(level, style={
        "fontSize": "10px", "fontWeight": 700, "color": color,
        "background": color + "18",
        "border": f"1px solid {color}55",
        "borderRadius": "20px", "padding": "2px 8px",
        "letterSpacing": "0.3px",
    })


def _sparkline_bar(ratio: float, color: str):
    """A thin progress bar to visualise a 0–1 ratio."""
    pct = max(0.0, min(1.0, ratio)) * 100
    return html.Div(
        html.Div(style={
            "width": f"{pct}%", "height": "100%",
            "background": color, "borderRadius": "4px",
            "transition": "width 0.4s ease",
        }),
        style={
            "height": "4px", "background": "#f1f5f9",
            "borderRadius": "4px", "marginTop": "8px",
        },
    )


# ── KPI Analytics summary cards ────────────────────────────────────────────

_SUMMARY_META = [
    ("Rows",         "📊", ACCENT,   lambda df: f"{len(df):,}"),
    ("Regions",      "🗺️",  "#0ea5e9", lambda df: f"{df['Region'].nunique():,}"
                                        if "Region" in df.columns else "0"),
    ("Sites",        "📡", "#8b5cf6", lambda df: f"{df['Radio Site'].nunique():,}"
                                        if "Radio Site" in df.columns else "0"),
    ("Cells",        "📶", "#06b6d4", lambda df: f"{df['Cell Name'].nunique():,}"
                                        if "Cell Name" in df.columns else "0"),
    ("Avg DL PRB",   "📈", "#f59e0b", lambda df: f"{df['DL PRB Load'].mean():.1f} %"
                                        if "DL PRB Load" in df.columns and not df.empty else "N/A"),
]


def build_summary_cards(df: pd.DataFrame) -> list:
    cards = []
    for label, icon, color, val_fn in _SUMMARY_META:
        try:
            value = val_fn(df)
        except Exception:
            value = "N/A"
        cards.append(html.Div([
            html.Div([
                html.Span(icon, style={"fontSize": "20px"}),
                html.Span(label, style={
                    "fontSize": "11px", "fontWeight": 600,
                    "color": TEXT_MUTED, "textTransform": "uppercase",
                    "letterSpacing": "0.5px",
                }),
            ], style={"display": "flex", "alignItems": "center",
                      "gap": "8px", "marginBottom": "10px"}),
            html.Div(value, style={
                "fontSize": "26px", "fontWeight": 800,
                "color": TEXT_PRIMARY, "lineHeight": 1.1,
                "fontVariantNumeric": "tabular-nums",
            }),
        ], style={
            **CARD_STYLE,
            "borderTop": f"3px solid {color}",
        }))
    return cards


# ── Congestion Analytics cards ─────────────────────────────────────────────

def build_congestion_cards(df: pd.DataFrame) -> list:
    if df.empty:
        p0r = p1r = p0c = p1c = busy = 0
        avg_score = "N/A"
    else:
        p0r  = int(df["P0_Congestion"].fillna(False).sum())  if "P0_Congestion" in df.columns else 0
        p1r  = int(df["P1_Congestion"].fillna(False).sum())  if "P1_Congestion" in df.columns else 0
        p0c  = (df.loc[df["P0_Congestion"] == True, "Cell Name"].nunique()
                if "P0_Congestion" in df.columns and "Cell Name" in df.columns else 0)
        p1c  = (df.loc[df["P1_Congestion"] == True, "Cell Name"].nunique()
                if "P1_Congestion" in df.columns and "Cell Name" in df.columns else 0)
        avg_score = (f"{df['FinalCongestionScore'].dropna().mean():.2f}"
                     if "FinalCongestionScore" in df.columns
                     and df["FinalCongestionScore"].notna().any() else "N/A")
        busy = (int(df["IsBusyHourRow"].fillna(False).sum())
                if "IsBusyHourRow" in df.columns else 0)

    total_rows  = max(len(df), 1)
    total_cells = max(df["Cell Name"].nunique() if "Cell Name" in df.columns else 1, 1)

    specs = [
        ("P0 Rows",       f"{p0r:,}",  P0_COLOR,   p0r  / total_rows,   "🔴"),
        ("P1 Rows",       f"{p1r:,}",  P1_COLOR,   p1r  / total_rows,   "🟠"),
        ("P0 Cells",      f"{p0c:,}",  P0_COLOR,   p0c  / total_cells,  "📵"),
        ("P1 Cells",      f"{p1c:,}",  P1_COLOR,   p1c  / total_cells,  "📶"),
        ("Avg Score",     avg_score,   ACCENT,     None,                 "🎯"),
        ("Busy-Hour Rows",f"{busy:,}", "#0ea5e9",  busy / total_rows,   "⏱️"),
    ]
    cards = []
    for label, value, color, ratio, icon in specs:
        cards.append(html.Div([
            html.Div([
                html.Span(icon, style={"fontSize": "16px"}),
                html.Span(label, style={
                    "fontSize": "11px", "fontWeight": 600,
                    "color": TEXT_MUTED, "textTransform": "uppercase",
                    "letterSpacing": "0.5px",
                }),
            ], style={"display": "flex", "alignItems": "center",
                      "gap": "6px", "marginBottom": "6px"}),
            html.Div(value, style={
                "fontSize": "24px", "fontWeight": 800,
                "color": color, "lineHeight": 1.1,
                "fontVariantNumeric": "tabular-nums",
            }),
            _sparkline_bar(ratio, color) if ratio is not None else html.Div(),
        ], style={
            **CARD_STYLE,
            "borderLeft": f"4px solid {color}",
            "paddingLeft": "16px",
        }))
    return cards


# ── Map side panel ─────────────────────────────────────────────────────────

def _section(title: str, children: list) -> html.Div:
    return html.Div([
        html.Div(title, style={
            "fontSize": "10px", "fontWeight": 700, "color": TEXT_MUTED,
            "textTransform": "uppercase", "letterSpacing": "0.8px",
            "marginBottom": "8px", "paddingBottom": "6px",
            "borderBottom": f"1px solid {BORDER_COLOR}",
        }),
        *children,
    ], style={"marginBottom": "14px"})


def build_map_panel(
    selected_sites: list,
    selected_cells: list,
    visible_cells,
    cong_lookup: dict,
    site_df: pd.DataFrame,
    filtered_df: pd.DataFrame | None = None,
) -> html.Div:

    # ── Cell-level detail ──────────────────────────────────────────────────
    if selected_cells:
        cards = []
        for cn in selected_cells:
            meta = CELL_GEO_META.get(cn)
            if not meta:
                continue
            level      = cong_lookup.get(cn, "No Data")
            border_clr = CONGESTION_COLORS.get(level, NO_DATA_COLOR)

            # KPI block from filtered DF
            kpi_items = []
            if filtered_df is not None and not filtered_df.empty and "Cell Name" in filtered_df.columns:
                sub = filtered_df[filtered_df["Cell Name"] == cn]
                for col, unit, lbl in [
                    ("DL PRB Load", "%", "DL PRB"),
                    ("Active Users", "", "Active Users"),
                    ("LTE DL User Throughput", "kbps", "DL Tput"),
                    ("Average CQI", "", "CQI"),
                ]:
                    if col in sub.columns and not sub[col].dropna().empty:
                        avg = sub[col].dropna().mean()
                        mx  = sub[col].dropna().max()
                        kpi_items.append(_row(lbl, f"{avg:.1f}{' '+unit if unit else ''}  (max {mx:.0f})"))
                for flag, lbl in [("P0_Congestion", "P0 hours"), ("P1_Congestion", "P1 hours")]:
                    if flag in sub.columns:
                        n = int(sub[flag].fillna(False).sum())
                        if n:
                            kpi_items.append(_row(lbl, str(n)))

            cards.append(html.Div([
                html.Div([
                    html.Span(cn, style={"fontWeight": 700, "fontSize": "13px",
                                         "color": TEXT_PRIMARY}),
                    _badge(level),
                ], style={"display": "flex", "alignItems": "center",
                          "justifyContent": "space-between", "marginBottom": "8px"}),
                _row("Site",     meta["site"]),
                _row("Region",   meta["region"]),
                _row("Azimuth",  f"{meta['azimuth']:.0f}°"),
                _row("Avg TA",   f"{meta['ta_m']:.0f} m" if meta.get("ta_m") else "—"),
                _row("Frequency", meta.get("frequency") or "—"),
                html.Div(kpi_items, style={"marginTop": "8px", "paddingTop": "8px",
                                           "borderTop": f"1px dashed {BORDER_COLOR}"})
                if kpi_items else html.Div(),
            ], style={
                "padding": "12px 14px",
                "border": f"1px solid {border_clr}44",
                "borderLeft": f"3px solid {border_clr}",
                "borderRadius": "12px", "marginBottom": "10px",
                "background": border_clr + "08",
            }))
        return html.Div([
            _section(f"Selected cells ({len(selected_cells)})", cards),
            html.P("Click the same cell again to deselect.",
                   style={"fontSize": "11px", "color": TEXT_MUTED, "margin": 0}),
        ])

    # ── Site-level detail ──────────────────────────────────────────────────
    if selected_sites:
        sel_set = set(selected_sites)
        rows = site_df[site_df["SiteKey"].isin(sel_set)] if not site_df.empty else pd.DataFrame()
        cards = []
        for _, row in rows.iterrows():
            worst = row.get("congestion", "No Data")
            color = CONGESTION_COLORS.get(worst, NO_DATA_COLOR)
            site_cells = sorted(
                [c for c, m in CELL_GEO_META.items()
                 if m["site_key"] == row["SiteKey"] and c in visible_cells],
                key=lambda c: CELL_GEO_META[c]["azimuth"],
            )
            bullets = [html.Div([
                html.Span("●", style={"color": CONGESTION_COLORS.get(cong_lookup.get(cn, "No Data"),
                                                                       NO_DATA_COLOR),
                                      "marginRight": "6px", "fontSize": "10px"}),
                html.Span(cn, style={"fontSize": "12px", "color": TEXT_PRIMARY,
                                     "fontWeight": 500}),
                html.Span(f"  {CELL_GEO_META[cn]['azimuth']:.0f}°  "
                          f"{CELL_GEO_META[cn].get('frequency', '')}",
                          style={"fontSize": "11px", "color": TEXT_MUTED}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"})
                       for cn in site_cells]
            cards.append(html.Div([
                html.Div([
                    html.Span(row["site_name"], style={"fontWeight": 700,
                                                        "color": TEXT_PRIMARY, "fontSize": "13px"}),
                    _badge(worst),
                ], style={"display": "flex", "alignItems": "center",
                          "justifyContent": "space-between", "marginBottom": "8px"}),
                _row("Region", row["region"]),
                _row("Cells",  int(row["cell_count"])),
                html.Div(bullets, style={"marginTop": "8px", "paddingTop": "8px",
                                         "borderTop": f"1px dashed {BORDER_COLOR}"})
                if bullets else html.Div(),
            ], style={
                "padding": "12px 14px",
                "border": f"1px solid {color}44",
                "borderLeft": f"3px solid {color}",
                "borderRadius": "12px", "marginBottom": "10px",
                "background": color + "08",
            }))
        return html.Div([
            _section(f"Selected sites ({len(selected_sites)})", cards),
            html.P("Click a sector cone to inspect a specific cell.",
                   style={"fontSize": "11px", "color": TEXT_MUTED, "margin": 0}),
        ])

    # ── Overview ───────────────────────────────────────────────────────────
    counts = site_df["congestion"].value_counts().to_dict() if (
        not site_df.empty and "congestion" in site_df.columns) else {}

    legend = [
        html.Div([
            html.Div(style={
                "width": "10px", "height": "10px", "borderRadius": "50%",
                "background": c, "flexShrink": 0,
            }),
            html.Span(lbl, style={"fontSize": "12px", "color": TEXT_MUTED}),
        ], style={"display": "flex", "alignItems": "center",
                  "gap": "8px", "marginBottom": "6px"})
        for lbl, c in [
            ("P0 — Severe",   CONGESTION_COLORS["P0"]),
            ("P1 — Moderate", CONGESTION_COLORS["P1"]),
            ("Not Congested", CONGESTION_COLORS["Not Congested"]),
            ("No Data",       NO_DATA_COLOR),
        ]
    ]

    stat_items = [
        ("Visible sites", f"{len(site_df):,}" if not site_df.empty else "0"),
        ("P0",            counts.get("P0", 0)),
        ("P1",            counts.get("P1", 0)),
        ("Not congested", counts.get("Not Congested", 0)),
    ]

    tips = [
        "Zoom in — clusters split into individual sites.",
        f"Sectors appear at zoom ≥ SECTOR_ZOOM_THRESHOLD.",
        "Click a site dot to reveal its sectors.",
        "Click a cone to inspect that cell.",
        "Draw a polygon to lasso-select multiple sites.",
    ]

    return html.Div([
        _section("Network overview", [_row(k, v) for k, v in stat_items]),
        _section("Legend",           legend),
        _section("Tips", [
            html.P(t, style={"fontSize": "11px", "color": TEXT_MUTED,
                             "margin": "0 0 4px 0"})
            for t in tips
        ]),
    ])
