"""
Pure data utilities — loading, column helpers, sampling, empty figures,
and the single-source filter implementation.
None of these depend on global DataFrames.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def load_dataframe(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path_obj.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path_obj, low_memory=False)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path_obj)
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_options(series: pd.Series) -> list[dict]:
    values = sorted(series.dropna().unique().tolist())
    return [{"label": str(v), "value": v} for v in values]


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.sample(max_points, random_state=42)


def make_empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title, template="plotly_white",
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[{
            "text": "No data after filtering",
            "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
            "showarrow": False, "font": {"size": 16},
        }],
    )
    return fig


def recompute_congestion_flags(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """
    Re-derive P0/P1 congestion flags from raw KPI columns using custom thresholds.
    Uses the same 3-way AND logic as congestion_engine / congestion.py.
    Only rewrites the flag + level columns; leaves all other columns intact.
    Returns a copy — does not mutate the input.
    """
    needed = ["DL PRB Load", "LTE DL User Throughput", "Active Users"]
    if df.empty or not all(c in df.columns for c in needed):
        return df

    df = df.copy()
    prb = df["DL PRB Load"].fillna(0)
    thp = df["LTE DL User Throughput"].fillna(9_999_999)
    usr = df["Active Users"].fillna(0)

    p0_prb   = float(thresholds.get("p0_prb",   90))
    p0_thp   = float(thresholds.get("p0_thp",   2000))
    p0_users = float(thresholds.get("p0_users", 8))
    p1_prb   = float(thresholds.get("p1_prb",   80))
    p1_thp   = float(thresholds.get("p1_thp",   4000))
    p1_users = float(thresholds.get("p1_users", 4))

    df["P0_Congestion"] = (prb > p0_prb) & (thp < p0_thp) & (usr > p0_users)
    df["P1_Congestion"] = (prb > p1_prb) & (thp < p1_thp) & (usr > p1_users)
    df.loc[df["P0_Congestion"], "P1_Congestion"] = False

    df["CongestionLevelHour"] = "None"
    df.loc[df["P1_Congestion"], "CongestionLevelHour"] = "P1"
    df.loc[df["P0_Congestion"], "CongestionLevelHour"] = "P0"
    return df


def _filter_dataframe_impl(df, regions, radio_sites, cells, modes, start, end):
    """Single-source filter logic. Filters by Cell Name (not Cell Key)."""
    out = df
    if regions and "Region" in out.columns:
        out = out[out["Region"].isin(regions)]
    if radio_sites and "Radio Site" in out.columns:
        out = out[out["Radio Site"].isin(radio_sites)]
    if cells and "Cell Name" in out.columns:
        out = out[out["Cell Name"].isin(cells)]
    if modes and "Cell FDD TDD Indication" in out.columns:
        out = out[out["Cell FDD TDD Indication"].isin(modes)]
    if start and "Time" in out.columns:
        out = out[out["Time"] >= pd.to_datetime(start)]
    if end and "Time" in out.columns:
        out = out[out["Time"] <= pd.to_datetime(end)]
    return out
