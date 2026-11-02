"""
Congestion computation engine — mirrors congestion.py but is importable.

Public API
----------
DEFAULT_THRESHOLDS  : dict with all 6 threshold keys + 3 weight keys
compute_congestion(df, thresholds=None) -> dict
    {"kpi_df": pd.DataFrame, "busy_hours": pd.DataFrame, "cell_summary": pd.DataFrame}
run_and_save(raw_path, *, output_kpi, output_busy, output_summary, thresholds=None) -> str
    status message

Congestion persistence rule (v2)
---------------------------------
A cell earns a CongestionLevel of P0 or P1 only when the hourly flags show
congestion on **≥ MIN_CONGESTED_DAYS_PER_WEEK distinct days-of-week** AND
that weekly pattern repeats across **≥ MIN_CONGESTED_WEEKS calendar weeks**.

Defaults:  MIN_CONGESTED_DAYS_PER_WEEK = 4,  MIN_CONGESTED_WEEKS = 3
(i.e. 4 days/week × 3 weeks).

Hourly P0_Congestion / P1_Congestion flags are kept as-is — they reflect
raw threshold crossings. Only the cell-level CongestionLevel verdict in the
summary (and the column merged back into kpi_df) is affected by this rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

# ── Columns ───────────────────────────────────────────────────────────────────
COL_TIME  = "Time"
COL_CELL  = "Cell Name"
COL_PRB   = "DL PRB Load"
COL_USERS = "Active Users"
COL_THP   = "LTE DL User Throughput"

REQUIRED_COLS = [COL_TIME, COL_CELL, COL_PRB, COL_USERS, COL_THP]

# ── Persistence thresholds (new rule) ─────────────────────────────────────────
MIN_CONGESTED_DAYS_PER_WEEK = 4   # distinct days-of-week in a single ISO week
MIN_CONGESTED_WEEKS         = 2   # calendar weeks that must show that pattern

# ── Defaults (matching the original congestion.py) ────────────────────────────
DEFAULT_THRESHOLDS: dict = {
    # P0 — severe congestion (all three conditions must hold simultaneously)
    "p0_prb":   90,     # DL PRB Load   >  90 %
    "p0_thp":   2000,   # DL Throughput <  2 000 kbps
    "p0_users": 8,      # Active Users  >  8
    # P1 — moderate congestion
    "p1_prb":   80,     # DL PRB Load   >  80 %
    "p1_thp":   4000,   # DL Throughput <  4 000 kbps
    "p1_users": 4,      # Active Users  >  4
    # Severity-score weights (must sum to 1)
    "w_prb":    0.45,
    "w_thp":    0.40,
    "w_active": 0.15,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _max_consecutive_days(date_series: pd.Series) -> int:
    if date_series.empty:
        return 0
    d = (pd.to_datetime(date_series)
         .dt.normalize().drop_duplicates().sort_values().values)
    streak = best = 1
    for i in range(1, len(d)):
        gap = (d[i] - d[i - 1]).astype("timedelta64[D]").astype(int)
        if gap == 1:
            streak += 1
            best = max(best, streak)
        else:
            streak = 1
    return int(best)


def _weeks_meeting_day_threshold(
    cong_dates: pd.Series,
    min_days: int = MIN_CONGESTED_DAYS_PER_WEEK,
) -> int:
    """
    Count how many ISO calendar weeks contain ≥ min_days distinct
    congested days-of-week.

    Parameters
    ----------
    cong_dates : Series of date strings or date objects  (days when congestion occurred)
    min_days   : minimum distinct days-of-week required per week

    Returns
    -------
    int — number of qualifying weeks
    """
    if cong_dates.empty:
        return 0
    dt = pd.to_datetime(cong_dates).dt.normalize()
    # ISO week key = (year, week_number)
    week_keys = dt.dt.isocalendar()[["year", "week"]].apply(
        lambda r: (int(r["year"]), int(r["week"])), axis=1
    )
    # For each week, count distinct days-of-week
    qualifying = 0
    for _wk, group in dt.groupby(week_keys):
        if group.dt.dayofweek.nunique() >= min_days:
            qualifying += 1
    return qualifying


def _congestion_level_from_persistence(
    busy_df: pd.DataFrame,
    min_days: int = MIN_CONGESTED_DAYS_PER_WEEK,
    min_weeks: int = MIN_CONGESTED_WEEKS,
) -> pd.Series:
    """
    For each cell, apply the persistence rule and return a Series of
    CongestionLevel values ("P0", "P1", or "None") indexed by cell name.

    Rule:
      • Candidate level = P0 if the cell has any P0 busy-hour days,
        P1 if it has P1 days (and no P0 ones that survive).
      • A cell reaches that level only when, for its candidate level,
        the qualifying-weeks count ≥ min_weeks.
      • P0 is checked first; if it qualifies → P0, else fall through to P1.
    """
    records = []
    for cell_name, grp in busy_df.groupby(COL_CELL):
        p0_dates = grp.loc[grp["BusyHour_P0"] == True, "Date"]
        p1_dates = grp.loc[grp["BusyHour_P1"] == True, "Date"]

        p0_weeks = _weeks_meeting_day_threshold(p0_dates, min_days)
        p1_weeks = _weeks_meeting_day_threshold(p1_dates, min_days)

        if p0_weeks >= min_weeks:
            level = "P0"
        elif p1_weeks >= min_weeks:
            level = "P1"
        else:
            level = "None"

        records.append({
            COL_CELL:             cell_name,
            "CongestionLevel":    level,
            "WeeksP0":            p0_weeks,
            "WeeksP1":            p1_weeks,
        })

    result = pd.DataFrame(records)
    if result.empty:
        return pd.Series(dtype=str)
    return result.set_index(COL_CELL)


# ── Core engine ───────────────────────────────────────────────────────────────

def _compute_summary_from_busy(
    busy_df: pd.DataFrame,
    min_days: int = MIN_CONGESTED_DAYS_PER_WEEK,
    min_weeks: int = MIN_CONGESTED_WEEKS,
) -> pd.DataFrame:
    """Derive cell_summary from the busy_hours DataFrame (full or partial).

    The CongestionLevel column now uses the persistence rule:
    ≥ min_days congested days-of-week per calendar week, repeated across
    ≥ min_weeks calendar weeks.
    """
    cell_day = busy_df[[COL_CELL, "Date", "BusyHour_Score_0_1", "BusyHour_Score",
                         "BusyHour_P0", "BusyHour_P1"]].copy()
    summary = (
        cell_day.groupby(COL_CELL, as_index=False).agg(
            DaysObserved       =("Date",               "nunique"),
            DaysP0             =("BusyHour_P0",        "sum"),
            DaysP1             =("BusyHour_P1",        "sum"),
            AvgBusyHourScore01 =("BusyHour_Score_0_1", "mean"),
            AvgBusyHourScore   =("BusyHour_Score",     "mean"),
            MaxBusyHourScore   =("BusyHour_Score",     "max"),
        )
    )
    summary["PersistenceRatio"] = (
        (summary["DaysP0"] + summary["DaysP1"]) / summary["DaysObserved"]
    )
    summary["FinalCongestionScore01"] = (
        0.7 * summary["AvgBusyHourScore01"] + 0.3 * summary["PersistenceRatio"]
    )
    summary["FinalCongestionScore"] = (100 * summary["FinalCongestionScore01"]).round(2)

    streaks = [
        (cell, _max_consecutive_days(
            grp.loc[(grp["BusyHour_P0"]) | (grp["BusyHour_P1"]), "Date"]
        ))
        for cell, grp in cell_day.groupby(COL_CELL)
    ]
    summary = summary.merge(
        pd.DataFrame(streaks, columns=[COL_CELL, "MaxConsecutiveCongestedDays"]),
        on=COL_CELL, how="left",
    )

    # ── Apply persistence rule for CongestionLevel ────────────────────────────
    persistence = _congestion_level_from_persistence(busy_df, min_days, min_weeks)
    if not persistence.empty:
        summary["CongestionLevel"] = summary[COL_CELL].map(
            persistence["CongestionLevel"]
        ).fillna("None")
        summary["WeeksP0"] = summary[COL_CELL].map(
            persistence.get("WeeksP0", pd.Series(dtype=int))
        ).fillna(0).astype(int)
        summary["WeeksP1"] = summary[COL_CELL].map(
            persistence.get("WeeksP1", pd.Series(dtype=int))
        ).fillna(0).astype(int)
    else:
        summary["CongestionLevel"] = "None"
        summary["WeeksP0"] = 0
        summary["WeeksP1"] = 0

    return summary


def compute_congestion(
    df: pd.DataFrame,
    thresholds: dict | None = None,
) -> dict:
    """
    Full congestion pipeline on a raw KPI DataFrame.

    Parameters
    ----------
    df          : raw KPI data (must contain REQUIRED_COLS)
    thresholds  : override any key from DEFAULT_THRESHOLDS; None = use defaults

    Returns
    -------
    {
        "kpi_df"      : hourly rows with P0/P1 flags + busy-hour + summary columns
        "busy_hours"  : one row per (cell, date) = the cell's busiest hour
        "cell_summary": one row per cell with persistence / final score
    }
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    df = df.copy()

    # ── Parse time ────────────────────────────────────────────────────────────
    df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
    df = df[df[COL_TIME].notna()].copy()
    df["Date"] = df[COL_TIME].dt.date.astype(str)
    df["Hour"] = df[COL_TIME].dt.hour

    prb  = df[COL_PRB].fillna(0)
    thp  = df[COL_THP].fillna(9_999_999)
    usr  = df[COL_USERS].fillna(0)

    # ── P0 / P1 flags (three-way AND — matching the script exactly) ───────────
    df["P0_Congestion"] = (prb > t["p0_prb"]) & (thp < t["p0_thp"]) & (usr > t["p0_users"])
    df["P1_Congestion"] = (prb > t["p1_prb"]) & (thp < t["p1_thp"]) & (usr > t["p1_users"])
    df.loc[df["P0_Congestion"], "P1_Congestion"] = False   # P0 overrides P1

    df["CongestionLevelHour"] = "None"
    df.loc[df["P1_Congestion"], "CongestionLevelHour"] = "P1"
    df.loc[df["P0_Congestion"], "CongestionLevelHour"] = "P0"

    # ── Severity score ────────────────────────────────────────────────────────
    prb_sev = np.clip((prb  - t["p1_prb"]) / 20,          0, 1)
    thp_sev = np.clip((t["p1_thp"] - thp)  / t["p1_thp"], 0, 1)
    act_sev = np.clip((usr  - t["p1_users"]) / 20,         0, 1)

    df["Hourly_Congestion_Score_0_1"] = (
        t["w_prb"] * prb_sev + t["w_thp"] * thp_sev + t["w_active"] * act_sev
    )
    df["Hourly_Congestion_Score"] = (100 * df["Hourly_Congestion_Score_0_1"]).round(2)

    # ── Busy hour per (cell, date) ────────────────────────────────────────────
    df["_pick"] = (
        df["P0_Congestion"].astype(int) * 20
        + df["P1_Congestion"].astype(int) * 10
        + df["Hourly_Congestion_Score_0_1"]
    )
    busy_idx = df.groupby([COL_CELL, "Date"])["_pick"].idxmax()
    busy = df.loc[busy_idx].copy()
    df.drop(columns=["_pick"], inplace=True)
    busy.drop(columns=["_pick"], inplace=True)

    busy = busy.rename(columns={
        "Hour":                        "BusyHour",
        "Hourly_Congestion_Score":     "BusyHour_Score",
        "Hourly_Congestion_Score_0_1": "BusyHour_Score_0_1",
    })
    busy["BusyHour_Level"] = busy["CongestionLevelHour"]
    busy["BusyHour_P0"]    = busy["P0_Congestion"]
    busy["BusyHour_P1"]    = busy["P1_Congestion"]

    # ── Cell summary (with persistence rule) ──────────────────────────────────
    summary = _compute_summary_from_busy(busy)

    # ── Merge busy-hour columns back to hourly rows ───────────────────────────
    df = df.merge(
        busy[[COL_CELL, "Date", "BusyHour", "BusyHour_Score", "BusyHour_Level"]],
        on=[COL_CELL, "Date"], how="left",
    )
    df["IsBusyHourRow"] = df["Hour"] == df["BusyHour"]

    df = df.merge(
        summary[[COL_CELL, "FinalCongestionScore", "CongestionLevel",
                 "DaysObserved", "DaysP0", "DaysP1", "MaxConsecutiveCongestedDays",
                 "WeeksP0", "WeeksP1"]],
        on=COL_CELL, how="left",
    )

    return {"kpi_df": df, "busy_hours": busy, "cell_summary": summary}


# ── Save wrapper (called from dashboard button or CLI) ────────────────────────

def run_and_save(
    raw_path: str,
    *,
    output_kpi:     str = "datasets/kpi_with_congestion.csv",
    output_busy:    str = "datasets/busy_hours_by_cell_day.csv",
    output_summary: str = "datasets/cell_congestion_summary.csv",
    thresholds: dict | None = None,
    mode: str = "replace",
) -> str:
    """Compute and persist all three outputs. Returns a human-readable status string.

    mode="replace" : full recompute, overwrites all three files.
    mode="append"  : only processes rows newer than the latest Time already saved,
                     appends to kpi and busy_hours CSVs, then rewrites cell_summary
                     from the full busy_hours so aggregates stay accurate.
    """
    p = Path(raw_path)
    if not p.exists():
        return f"❌  Raw file not found: {raw_path}"

    try:
        df = (pd.read_csv(p, low_memory=False)
              if p.suffix.lower() == ".csv"
              else pd.read_excel(p))
    except Exception as exc:
        return f"❌  Could not read file: {exc}"

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return f"❌  Missing columns: {missing}"

    # ── Append path ───────────────────────────────────────────────────────────
    if mode == "append":
        kpi_path = Path(output_kpi)
        if not kpi_path.exists():
            # Nothing to append to — fall back to replace and warn
            mode = "replace"
        else:
            try:
                existing = pd.read_csv(kpi_path, usecols=[COL_TIME], low_memory=False)
                max_t = pd.to_datetime(existing[COL_TIME], errors="coerce").max()
            except Exception as exc:
                return f"❌  Could not read existing file to find cutoff: {exc}"

            df[COL_TIME] = pd.to_datetime(df[COL_TIME], errors="coerce")
            new_raw = df[df[COL_TIME] > max_t].copy()

            if new_raw.empty:
                return f"ℹ️  No new data found after {max_t} — nothing appended."

            try:
                result = compute_congestion(new_raw, thresholds)

                for path_str, key in [
                    (output_kpi,  "kpi_df"),
                    (output_busy, "busy_hours"),
                ]:
                    path_obj = Path(path_str)
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                    result[key].to_csv(path_str, mode="a", header=False, index=False)

                # Recompute summary from the FULL busy_hours so aggregates stay accurate
                full_busy = pd.read_csv(output_busy, low_memory=False)
                new_summary = _compute_summary_from_busy(full_busy)
                Path(output_summary).parent.mkdir(parents=True, exist_ok=True)
                new_summary.to_csv(output_summary, index=False)

                n    = len(result["kpi_df"])
                dmin = result["kpi_df"][COL_TIME].min()
                dmax = result["kpi_df"][COL_TIME].max()
                return (
                    f"✅  Appended {n:,} new rows "
                    f"({str(dmin)[:10]} → {str(dmax)[:10]}) → {output_kpi}"
                )
            except Exception as exc:
                return f"❌  Append failed: {exc}"

    # ── Replace path (default) ────────────────────────────────────────────────
    try:
        result = compute_congestion(df, thresholds)
        for path, key in [
            (output_kpi,     "kpi_df"),
            (output_busy,    "busy_hours"),
            (output_summary, "cell_summary"),
        ]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            result[key].to_csv(path, index=False)
        n = len(result["kpi_df"])
        return f"✅  Congestion computed for {n:,} rows → saved to {output_kpi}"
    except Exception as exc:
        return f"❌  Computation failed: {exc}"
