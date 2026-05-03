"""
Reusable UI component builders — summary cards and congestion cards.
"""

import pandas as pd
from dash import html

from config import CARD_STYLE, P0_COLOR, P1_COLOR


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
