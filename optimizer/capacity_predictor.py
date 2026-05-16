"""
capacity_predictor.py — derive cell capacity ceilings Λᵢ from KPIs.

The optimizer needs to know, for each cell, the active-user count at which it
would hit the congestion threshold. In the offline pipeline we trained an MLP
on historical KPIs and swept Active Users to find the crossing. In the
dashboard, the existing XGBoost forecaster already does the heavy lifting; we
just need to extract the right snapshots and apply the same threshold logic.

Two modes are supported:

  "current"     → use the most recent observed KPIs from DF as the cell state,
                  taking active users directly from the data and using the
                  congestion-engine thresholds to define Λᵢ implicitly.

  "forecasted"  → use the XGBoost forecast records produced by the AI tab,
                  picking the time slice the engineer wants to optimize for.

Both modes return the same kind of dict so the optimizer doesn't care which
one fed it.

This module knows about the dashboard's data structures (DF, forecast records)
but expects them to be passed in — it does not import them directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Default thresholds — match the dashboard's congestion_engine.DEFAULT_THRESHOLDS
# These can be overridden per-call to keep the UI consistent with the user's
# settings panel.
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_PRB_THRESHOLD_PCT = 80.0    # the industry-standard congestion threshold
SAFETY_MARGIN_PCT = 3.0             # paper uses 3% to absorb DL prediction error


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1: current — read from observed KPIs
# ─────────────────────────────────────────────────────────────────────────────
def get_current_state(
    cell_names: list[str],
    df: pd.DataFrame,
    prb_threshold_pct: float = DEFAULT_PRB_THRESHOLD_PCT,
    safety_margin_pct: float = SAFETY_MARGIN_PCT,
) -> dict:
    """
    For each cell, derive (current_lambda, capacity_Lambda) from observed KPIs.

    The capacity Λᵢ is the active-user count where this cell historically hits
    the PRB threshold. We estimate it directly from data: find observations
    where PRB load is near the threshold and average the corresponding
    active-user counts.

    Parameters
    ----------
    cell_names : list of str
        Cells to compute for.
    df : pd.DataFrame
        The dashboard's main KPI dataframe. Must contain columns
        "Cell Name", "Active Users", "DL PRB Load".
    prb_threshold_pct : float
        The PRB-load percentage that defines congestion (default 80).
    safety_margin_pct : float
        Subtracted from the threshold to absorb prediction noise.

    Returns
    -------
    {
        "lam":         np.ndarray of current users (latest hour) per cell,
        "Lambda":      np.ndarray of capacity Λᵢ per cell,
        "cell_names":  list of str (same order),
        "timestamp":   ISO string of the snapshot time used,
        "note":        diagnostic message,
    }
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty — cannot derive current state.")

    needed = {"Cell Name", "Active Users", "DL PRB Load"}
    if not needed.issubset(df.columns):
        missing = needed - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    threshold = prb_threshold_pct - safety_margin_pct

    # Use the most recent observation per cell as the "current" λ
    sub = df[df["Cell Name"].isin(cell_names)].dropna(
        subset=["Cell Name", "Active Users", "DL PRB Load"]
    )
    if sub.empty:
        raise ValueError("No rows match the requested cells.")

    # Most recent timestamp per cell
    if "Time" in sub.columns:
        latest_idx = sub.groupby("Cell Name")["Time"].idxmax()
        latest = sub.loc[latest_idx].set_index("Cell Name")
        snapshot_time = sub["Time"].max()
    else:
        latest = sub.drop_duplicates("Cell Name", keep="last").set_index("Cell Name")
        snapshot_time = None

    # Capacity Λᵢ — derived per cell from the historical relationship
    Lambda_by_cell = _estimate_capacity_from_history(sub, threshold)

    lam_vec = np.zeros(len(cell_names), dtype=float)
    Lambda_vec = np.zeros(len(cell_names), dtype=float)

    for i, cn in enumerate(cell_names):
        if cn in latest.index:
            row = latest.loc[cn]
            lam_vec[i] = float(row["Active Users"]) if pd.notna(row["Active Users"]) else 0.0
        else:
            lam_vec[i] = 0.0
        Lambda_vec[i] = Lambda_by_cell.get(cn, _fallback_capacity(sub, cn))

    return {
        "lam":        lam_vec,
        "Lambda":     Lambda_vec,
        "cell_names": list(cell_names),
        "timestamp":  snapshot_time.isoformat() if snapshot_time is not None else None,
        "note":       f"Current state from observed KPIs at {snapshot_time}",
    }


def _estimate_capacity_from_history(
    df: pd.DataFrame, threshold_pct: float, window_pct: float = 5.0,
) -> dict[str, float]:
    """
    For each cell, estimate Λᵢ by averaging the Active Users at observations
    where PRB Load was close to the threshold.

    If a cell has never hit the threshold, fall back to extrapolating from
    the slope of PRB vs Active Users on the cell's own data.
    """
    out: dict[str, float] = {}
    lo = threshold_pct - window_pct
    hi = threshold_pct + window_pct

    for cn, grp in df.groupby("Cell Name"):
        near_threshold = grp[(grp["DL PRB Load"] >= lo) & (grp["DL PRB Load"] <= hi)]
        if len(near_threshold) >= 3:
            out[cn] = float(near_threshold["Active Users"].median())
        elif (grp["DL PRB Load"] >= threshold_pct).any():
            # Cell did hit the threshold at least once — use earliest such crossing
            crossed = grp[grp["DL PRB Load"] >= threshold_pct]
            out[cn] = float(crossed["Active Users"].min())
        else:
            # Never hit — linear extrapolation from peak observed
            peak_prb = grp["DL PRB Load"].max()
            peak_users = grp.loc[grp["DL PRB Load"].idxmax(), "Active Users"] \
                         if not grp["DL PRB Load"].isna().all() else np.nan
            if pd.notna(peak_prb) and peak_prb > 5 and pd.notna(peak_users) and peak_users > 0:
                # Assume linearity: scale users in the same ratio as PRB
                out[cn] = float(peak_users * threshold_pct / peak_prb)
            # else: leave out of dict, caller will use fallback
    return out


def _fallback_capacity(df: pd.DataFrame, cell_name: str) -> float:
    """Last-resort capacity estimate when nothing else works."""
    cell_max = df.loc[df["Cell Name"] == cell_name, "Active Users"].max()
    if pd.notna(cell_max) and cell_max > 0:
        return float(cell_max * 1.2)   # 20% above peak observed
    return 5.0  # match the median from the offline ARI cluster


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2: forecasted — read from XGBoost forecast records
# ─────────────────────────────────────────────────────────────────────────────
def get_forecasted_state(
    cell_names: list[str],
    forecast_records: list[dict],
    df_for_capacity: pd.DataFrame,
    forecast_hour_offset: int | None = None,
    prb_threshold_pct: float = DEFAULT_PRB_THRESHOLD_PCT,
    safety_margin_pct: float = SAFETY_MARGIN_PCT,
) -> dict:
    """
    For each cell, derive future (lambda, Lambda) from forecast records.

    Parameters
    ----------
    cell_names : list of str
        Cells to compute for.
    forecast_records : list of dict
        Records produced by run_forecast() in forecast_callbacks.py.
        Each record has keys: Time, Cell Name, DL PRB Load, Active Users,
        LTE DL User Throughput, Hour, Risk Level.
    df_for_capacity : pd.DataFrame
        Historical KPI dataframe — used to compute Lambda (which doesn't
        change with the forecast; it's an intrinsic property of each cell).
    forecast_hour_offset : int or None
        Which forecast hour to optimize for.
        - None or 0 → optimize for the peak-congestion hour in the forecast
        - 1, 2, ... → optimize for that specific hour ahead

    Returns the same dict shape as get_current_state().
    """
    if not forecast_records:
        raise ValueError(
            "No forecast records provided. Run the AI Forecasting tab first."
        )

    fc_df = pd.DataFrame(forecast_records)
    if "Time" in fc_df.columns:
        fc_df["Time"] = pd.to_datetime(fc_df["Time"], errors="coerce")

    cells_set = set(cell_names)
    fc_df = fc_df[fc_df["Cell Name"].isin(cells_set)]
    if fc_df.empty:
        raise ValueError(
            "Forecast records do not include any of the requested cells. "
            "Run the AI Forecasting tab on this cluster first."
        )

    # Pick the time slice to optimize for
    if forecast_hour_offset is None or forecast_hour_offset <= 0:
        # Default: the most-congested future moment in the forecast
        # We sum predicted PRB-over-threshold across cells per timestamp
        threshold = prb_threshold_pct
        scored = (fc_df.assign(over=(fc_df["DL PRB Load"] - threshold).clip(lower=0))
                       .groupby("Time")["over"].sum()
                       .sort_values(ascending=False))
        chosen_time = scored.index[0] if not scored.empty else fc_df["Time"].min()
    else:
        unique_times = sorted(fc_df["Time"].dropna().unique())
        idx = min(forecast_hour_offset - 1, len(unique_times) - 1)
        chosen_time = unique_times[idx]

    snapshot = fc_df[fc_df["Time"] == chosen_time].set_index("Cell Name")

    # Lambda is still derived from history — it's a cell property, not time-dependent
    threshold = prb_threshold_pct - safety_margin_pct
    Lambda_by_cell = _estimate_capacity_from_history(df_for_capacity, threshold)

    lam_vec = np.zeros(len(cell_names), dtype=float)
    Lambda_vec = np.zeros(len(cell_names), dtype=float)
    for i, cn in enumerate(cell_names):
        if cn in snapshot.index:
            v = snapshot.loc[cn, "Active Users"]
            lam_vec[i] = float(v) if pd.notna(v) else 0.0
        else:
            lam_vec[i] = 0.0
        Lambda_vec[i] = Lambda_by_cell.get(
            cn, _fallback_capacity(df_for_capacity, cn)
        )

    return {
        "lam":        lam_vec,
        "Lambda":     Lambda_vec,
        "cell_names": list(cell_names),
        "timestamp":  chosen_time.isoformat() if pd.notna(chosen_time) else None,
        "note":       f"Forecasted state for {chosen_time}",
    }
