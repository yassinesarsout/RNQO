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
