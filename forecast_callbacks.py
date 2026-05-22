"""
Forecast tab callbacks — AI-powered forecasting for RAN KPIs.  (v2)

Changes from v1:
  - Cells now come from the global filter (region/site/cell/mode/date)
    instead of the local fc-cell-selector dropdown.
  - "Save Week Forecast" runs a background thread, persists to
    datasets/forecast_cache.csv, and surfaces live progress messages.
  - When the global filter changes and a saved cache exists the store is
    reloaded automatically so charts update without re-running the model.
  - fc-filter-summary pill shows the count of cells currently in scope.
"""

import os
import json
import sys
import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import (
    Input, Output, State, callback,
    no_update, html, ctx, dcc,
)
from dash.exceptions import PreventUpdate

from config import (
    SECTION_STYLE, CARD_STYLE, P0_COLOR, P1_COLOR,
    DEFAULT_PRB_THRESHOLD, DEFAULT_ACTIVE_USER_THRESHOLD,
)
from precompute import DF, get_filtered_df

_ACCENT  = "#6366f1"
_SUCCESS = "#22c55e"
_WARN    = "#f97316"
_DANGER  = "#ef4444"
_INFO    = "#0891b2"

# ── Cache path ─────────────────────────────────────────────────────────────────
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "forecast_cache.csv")
_XGB_DIR    = os.path.join(os.path.dirname(__file__), "Models")

_XGB_MODEL_FILES = {
    "DL PRB Load":            "xgb_prb.json",
    "Active Users":           "xgb_users.json",
    "LTE DL User Throughput": "xgb_throughput.json",
}
_XGB_SCALER_FILES = {
    "DL PRB Load":            "scaler_prb.pkl",
    "Active Users":           "scaler_users.pkl",
    "LTE DL User Throughput": "scaler_throughput.pkl",
}

_XGB_MODELS:  dict = {}
_XGB_SCALERS: dict = {}
_XGB_META:    dict = {}
_XGB_FEATURES: dict = {}


def _xgb_feature_list(rolling_col: str, fdd_cols: list) -> list:
    return [
        "cell_id", "site_id", "region_id",
        "prb_lag_1d", "throughput_lag_1d", "users_lag_1d",
        "prb_lag_1w", "throughput_lag_1w", "users_lag_1w",
        "prb_lag_2w", "throughput_lag_2w", "users_lag_2w",
        rolling_col,
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_weekend",
    ] + list(fdd_cols)


def _try_load_xgb():
    meta_path = os.path.join(_XGB_DIR, "meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                _XGB_META.update(json.load(f))
        except Exception:
            pass

    fdd_cols = _XGB_META.get("fdd_cols", [])
    _XGB_FEATURES.update({
        "DL PRB Load":            _xgb_feature_list("prb_rolling_24h",       fdd_cols),
        "Active Users":           _xgb_feature_list("users_rolling_24h",      fdd_cols),
        "LTE DL User Throughput": _xgb_feature_list("throughput_rolling_24h", fdd_cols),
    })

    for kpi, fname in _XGB_MODEL_FILES.items():
        mp = os.path.join(_XGB_DIR, fname)
        if os.path.exists(mp):
            try:
                import xgboost as xgb
                m = xgb.XGBRegressor()
                m.load_model(mp)
                _XGB_MODELS[kpi] = m
            except Exception:
                pass

    for kpi, fname in _XGB_SCALER_FILES.items():
        sp = os.path.join(_XGB_DIR, fname)
        if os.path.exists(sp):
            try:
                import joblib
                _XGB_SCALERS[kpi] = joblib.load(sp)
            except Exception:
                pass

_try_load_xgb()


# ─────────────────────────────────────────────────────────────────────────────
# Model retraining — background thread + status  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_LOCK   = threading.Lock()
_TRAIN_STATUS = {"running": False, "message": "Idle", "error": False}


def _reload_xgb_models():
    _XGB_MODELS.clear(); _XGB_SCALERS.clear()
    _XGB_META.clear();   _XGB_FEATURES.clear()
    _try_load_xgb()


def _run_training_thread(data_path: str):
    def _progress(msg: str):
        with _TRAIN_LOCK:
            _TRAIN_STATUS["message"] = msg

    _XGB_MODELS.clear()
    try:
        if _XGB_DIR not in sys.path:
            sys.path.insert(0, _XGB_DIR)
        import importlib
        tx = importlib.import_module("train_xgboost")
        importlib.reload(tx)
        result  = tx.train_all(data_path=data_path, models_dir=_XGB_DIR, progress_cb=_progress)
        success = result.startswith("✅")
        if success:
            _reload_xgb_models()
    except Exception as exc:
        result  = f"❌  Thread error: {exc}"
        success = False

    with _TRAIN_LOCK:
        _TRAIN_STATUS["running"] = False
        _TRAIN_STATUS["message"] = result
        _TRAIN_STATUS["error"]   = not success


# ─────────────────────────────────────────────────────────────────────────────
# Week-forecast background thread + status
# ─────────────────────────────────────────────────────────────────────────────

_FC_LOCK   = threading.Lock()
_FC_STATUS = {
    "running":     False,
    "message":     "Ready.",
    "error":       False,
    "progress":    0,
    "total":       0,
    "saved_ts":    None,   # ISO timestamp of last successful week-save
    "saved_cells": 0,
    "mode":        None,   # "live" = Run Forecast  |  "week" = Save Week
    "live_result": None,   # filled by _run_live_forecast_thread when done
}


def _run_week_forecast_thread(cell_list: list, thresholds: dict):
    """Background thread: 168-h forecast for every cell, saved to CSV."""
    kpis = ["DL PRB Load", "Active Users", "LTE DL User Throughput"]

    def _prog(msg: str):
        with _FC_LOCK:
            _FC_STATUS["message"] = msg

    try:
        n = len(cell_list)
        with _FC_LOCK:
            _FC_STATUS["total"]    = n
            _FC_STATUS["progress"] = 0

        _prog(f"⏳ Building cell index for {n} cells…")
        cell_index = _make_cell_index(DF)

        records = []
        for idx, cn in enumerate(cell_list):
            with _FC_LOCK:
                _FC_STATUS["progress"] = idx + 1
            _prog(f"⏳ Forecasting cell {idx + 1}/{n}: {cn}")

            hist      = cell_index.get(cn)
            cell_last = hist.index.max() if (hist is not None and not hist.empty) else DF["Time"].max()
            ftimes    = [cell_last + pd.Timedelta(hours=h + 1) for h in range(168)]

            for kpi in kpis:
                if kpi not in DF.columns:
                    continue
                preds = _forecast_one_cell_one_kpi(kpi, cn, ftimes, cell_index)
                for i, t in enumerate(ftimes):
                    v = float(preds[i]) if not np.isnan(preds[i]) else None
                    records.append({
                        "Time":      t.isoformat(),
                        "Cell Name": cn,
                        "KPI":       kpi,
                        "Value":     v,
                        "Hour":      t.hour,
                    })

        if not records:
            with _FC_LOCK:
                _FC_STATUS["running"] = False
                _FC_STATUS["message"] = "❌  No data produced — check that cells have history."
                _FC_STATUS["error"]   = True
            return

        _prog("💾 Saving forecast to disk…")
        fc_df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        fc_df.to_csv(_CACHE_PATH, index=False)

        ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        with _FC_LOCK:
            _FC_STATUS["running"]      = False
            _FC_STATUS["message"]      = f"✅  Week forecast saved — {n} cells · {ts}"
            _FC_STATUS["error"]        = False
            _FC_STATUS["saved_ts"]     = ts
            _FC_STATUS["saved_cells"]  = n

    except Exception as exc:
        with _FC_LOCK:
            _FC_STATUS["running"] = False
            _FC_STATUS["message"] = f"❌  Error: {exc}"
            _FC_STATUS["error"]   = True


def _run_live_forecast_thread(cell_list: list, horizon: int, thresholds: dict):
    """
    Background thread for the 'Run Forecast' button.
    Calls run_forecast() with a progress callback, stores the result in
    _FC_STATUS['live_result'] so the poll callback can push it to the store.
    """
    def _prog(done, total, cn):
        with _FC_LOCK:
            _FC_STATUS["progress"] = done
            _FC_STATUS["total"]    = total
            _FC_STATUS["message"]  = f"⏳ Forecasting cell {done}/{total}: {cn}"

    try:
        result = run_forecast(cell_list, horizon, thresholds, progress_cb=_prog)
        with _FC_LOCK:
            _FC_STATUS["running"]     = False
            _FC_STATUS["live_result"] = result
            n = len(cell_list)
            _FC_STATUS["message"] = (
                f"✅  Forecast complete — {n} cell{'s' if n != 1 else ''}, "
                f"{horizon} h horizon."
            )
            _FC_STATUS["error"] = False
    except Exception as exc:
        with _FC_LOCK:
            _FC_STATUS["running"]     = False
            _FC_STATUS["live_result"] = None
            _FC_STATUS["message"]     = f"❌  Forecast error: {exc}"
            _FC_STATUS["error"]       = True


# ─────────────────────────────────────────────────────────────────────────────
# Core forecast engine  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _make_cell_index(df: pd.DataFrame) -> dict:
    kpis = ["DL PRB Load", "Active Users", "LTE DL User Throughput"]
    idx  = {}
    for cn, grp in df.groupby("Cell Name"):
        g = grp.sort_values("Time").drop_duplicates("Time").set_index("Time")
        idx[str(cn)] = g[[k for k in kpis if k in g.columns]]
    return idx


def _lag_lookup(series: pd.Series, t: pd.Timestamp, h: int):
    """Single-point lag lookup — kept for compatibility but not used in hot path."""
    try:
        return series.at[t - pd.Timedelta(hours=h)]
    except KeyError:
        return np.nan


def _rolling24(series: pd.Series, t: pd.Timestamp) -> float:
    """Single-point rolling mean — kept for compatibility but not used in hot path."""
    lo = t - pd.Timedelta(hours=24)
    hi = t - pd.Timedelta(hours=1)
    window = series.loc[lo:hi]
    return float(window.mean()) if not window.empty else 0.0


def _cell_fdd(cell_name: str) -> str:
    if DF.empty or "Cell FDD TDD Indication" not in DF.columns:
        return ""
    rows = DF.loc[DF["Cell Name"] == cell_name, "Cell FDD TDD Indication"]
    return str(rows.iloc[0]) if not rows.empty else ""


# ── Vectorized lag helper ─────────────────────────────────────────────────────
def _lag_vec(series: pd.Series, ft_idx: pd.DatetimeIndex, h: int) -> np.ndarray:
    """
    For every timestamp in ft_idx, look up series[timestamp - h hours].
    Uses .reindex() — one C-level operation instead of len(ft_idx) Python calls.
    """
    return series.reindex(ft_idx - pd.Timedelta(hours=h)).fillna(0.0).values


def _rolling_vec(series: pd.Series, ft_idx: pd.DatetimeIndex) -> np.ndarray:
    """
    For every timestamp in ft_idx, compute mean of the 24 h window ending 1 h before it.
    Extends the series with NaN placeholders so .rolling() can cover future timestamps
    — one pandas pass instead of len(ft_idx) individual .loc[] slices.
    """
    combined_idx = series.index.union(ft_idx)
    extended     = series.reindex(combined_idx)          # NaN for future slots
    # shift(1) → window ends at t-1h; rolling(24) → covers t-1h … t-24h
    rolled = extended.rolling(24, min_periods=1).mean().shift(1)
    return rolled.reindex(ft_idx).fillna(0.0).values


def _build_xgb_feature_matrix(kpi: str, cell_name: str,
                               ft_idx: pd.DatetimeIndex, hist) -> pd.DataFrame:
    """
    Build the full (n_timesteps × n_features) feature matrix for one cell using
    vectorized pandas/numpy — no Python loops over timestamps.

    Replaces the old _build_xgb_rows() which looped row-by-row.
    """
    meta     = _XGB_META
    fdd_cols = meta.get("fdd_cols", [])
    cell_id   = meta.get("cell_codes",    {}).get(cell_name, 0)
    site_name = meta.get("cell_to_site",  {}).get(cell_name, "")
    reg_name  = meta.get("cell_to_region",{}).get(cell_name, "")
    site_id   = meta.get("site_codes",    {}).get(site_name, 0)
    region_id = meta.get("region_codes",  {}).get(reg_name,  0)
    fdd_val   = _cell_fdd(cell_name)

    _rolling_kpi = {
        "DL PRB Load":            "DL PRB Load",
        "Active Users":           "Active Users",
        "LTE DL User Throughput": "LTE DL User Throughput",
    }[kpi]
    _rolling_col = {
        "DL PRB Load":            "prb_rolling_24h",
        "Active Users":           "users_rolling_24h",
        "LTE DL User Throughput": "throughput_rolling_24h",
    }[kpi]

    n = len(ft_idx)

    def _s(col):
        """Return historical series for col, or empty series if unavailable."""
        if hist is not None and col in hist.columns:
            return hist[col]
        return pd.Series(dtype=float)

    # ── All lags — one .reindex() per lag instead of n individual .at[] calls ──
    prb_s  = _s("DL PRB Load")
    tput_s = _s("LTE DL User Throughput")
    usr_s  = _s("Active Users")

    prb_lag_1d  = _lag_vec(prb_s,  ft_idx, 24)
    tput_lag_1d = _lag_vec(tput_s, ft_idx, 24)
    usr_lag_1d  = _lag_vec(usr_s,  ft_idx, 24)
    prb_lag_1w  = _lag_vec(prb_s,  ft_idx, 168)
    tput_lag_1w = _lag_vec(tput_s, ft_idx, 168)
    usr_lag_1w  = _lag_vec(usr_s,  ft_idx, 168)
    prb_lag_2w  = _lag_vec(prb_s,  ft_idx, 336)
    tput_lag_2w = _lag_vec(tput_s, ft_idx, 336)
    usr_lag_2w  = _lag_vec(usr_s,  ft_idx, 336)

    # ── Rolling mean — one pandas pass instead of n slice operations ──────────
    roll_s   = _s(_rolling_kpi)
    roll_val = _rolling_vec(roll_s, ft_idx) if not roll_s.empty else np.zeros(n)

    # ── Cyclical time features — pure numpy, no lookups needed ────────────────
    hours   = ft_idx.hour.values
    dow     = ft_idx.dayofweek.values
    months  = ft_idx.month.values

    df_feat = pd.DataFrame({
        "cell_id":           cell_id,
        "site_id":           site_id,
        "region_id":         region_id,
        "prb_lag_1d":        prb_lag_1d,
        "throughput_lag_1d": tput_lag_1d,
        "users_lag_1d":      usr_lag_1d,
        "prb_lag_1w":        prb_lag_1w,
        "throughput_lag_1w": tput_lag_1w,
        "users_lag_1w":      usr_lag_1w,
        "prb_lag_2w":        prb_lag_2w,
        "throughput_lag_2w": tput_lag_2w,
        "users_lag_2w":      usr_lag_2w,
        _rolling_col:        roll_val,
        "hour_sin":          np.sin(2 * np.pi * hours  / 24),
        "hour_cos":          np.cos(2 * np.pi * hours  / 24),
        "dow_sin":           np.sin(2 * np.pi * dow    / 7),
        "dow_cos":           np.cos(2 * np.pi * dow    / 7),
        "month_sin":         np.sin(2 * np.pi * months / 12),
        "month_cos":         np.cos(2 * np.pi * months / 12),
        "is_weekend":        (dow >= 5).astype(int),
    })

    # ── FDD/TDD one-hot ───────────────────────────────────────────────────────
    for col in fdd_cols:
        v = col.replace("Cell FDD TDD Indication_", "")
        df_feat[col] = int(fdd_val == v)

    return df_feat


def _forecast_one_cell_one_kpi(kpi, cell_name, forecast_times, cell_index) -> np.ndarray:
    hist   = cell_index.get(cell_name)
    model  = _XGB_MODELS.get(kpi)
    scaler = _XGB_SCALERS.get(kpi)
    feats  = _XGB_FEATURES.get(kpi, [])

    ft_idx = pd.DatetimeIndex(forecast_times)

    # ── XGBoost path: fully vectorized feature matrix ────────────────────────
    if model is not None and scaler is not None and feats:
        X_df = _build_xgb_feature_matrix(kpi, cell_name, ft_idx, hist)
        for fn in feats:
            if fn not in X_df.columns:
                X_df[fn] = 0
        X_df = X_df[feats].fillna(0)
        try:
            return model.predict(scaler.transform(X_df.values))
        except Exception:
            pass

    # ── Lag-blend fallback: vectorized reindex instead of loop ───────────────
    if hist is None or kpi not in hist.columns:
        return np.full(len(forecast_times), np.nan)

    s = hist[kpi]
    lag_24  = s.reindex(ft_idx - pd.Timedelta(hours=24)).values
    lag_168 = s.reindex(ft_idx - pd.Timedelta(hours=168)).values
    lag_336 = s.reindex(ft_idx - pd.Timedelta(hours=336)).values
    return np.nanmean(np.column_stack([lag_24, lag_168, lag_336]), axis=1)


def run_forecast(cells: list, horizon_hours: int,
                 thresholds: dict | None = None,
                 progress_cb=None) -> dict:
    """
    Forecast `cells` for `horizon_hours` into the future.

    Per-cell anchor:  each cell uses its own last known timestamp as the
      forecast origin, so cells with older data don't produce gaps.

    24-hour real-data overlap:  the 24 h ending at cell_last_time are taken
      directly from historical data (is_forecast=False) and prepended to the
      output.  This bridges the historical trace to the prediction line so the
      chart is always continuous.

    progress_cb(done, total, cell_name) is called once per cell when provided.
    """
    if not cells or DF.empty:
        return {}

    from congestion_engine import DEFAULT_THRESHOLDS as _DT
    t = {**_DT, **(thresholds or {})}

    cell_index    = _make_cell_index(DF)
    kpis          = ["DL PRB Load", "Active Users", "LTE DL User Throughput"]
    OVERLAP_HOURS = 24

    records = []
    for idx, cn in enumerate(cells):
        if progress_cb:
            progress_cb(idx + 1, len(cells), cn)

        hist = cell_index.get(cn)

        # ── Per-cell anchor time ──────────────────────────────────────────
        cell_last = hist.index.max() if (hist is not None and not hist.empty) else DF["Time"].max()

        # ── 24 h real-data overlap (no prediction, just real values) ──────
        overlap_start = cell_last - pd.Timedelta(hours=OVERLAP_HOURS - 1)
        for ot in pd.date_range(overlap_start, cell_last, freq="h"):
            def _real(kpi, _ot=ot):
                if hist is None or kpi not in hist.columns:
                    return None
                v = hist[kpi].get(_ot, np.nan)
                return float(v) if pd.notna(v) else None

            records.append({
                "Time": ot.isoformat(), "Cell Name": cn,
                "DL PRB Load":            _real("DL PRB Load"),
                "Active Users":           _real("Active Users"),
                "LTE DL User Throughput": _real("LTE DL User Throughput"),
                "Hour": ot.hour, "Risk Level": "Historical",
                "is_forecast": False,
            })

        # ── Future predictions starting from cell_last + 1 h ─────────────
        future_times = [cell_last + pd.Timedelta(hours=h + 1) for h in range(horizon_hours)]

        kpi_preds = {}
        for kpi in kpis:
            if kpi in DF.columns:
                kpi_preds[kpi] = _forecast_one_cell_one_kpi(kpi, cn, future_times, cell_index)
            else:
                kpi_preds[kpi] = np.full(len(future_times), np.nan)

        for i, ft in enumerate(future_times):
            prb  = float(kpi_preds["DL PRB Load"][i])              if not np.isnan(kpi_preds["DL PRB Load"][i])              else None
            usr  = float(kpi_preds["Active Users"][i])             if not np.isnan(kpi_preds["Active Users"][i])             else None
            tput = float(kpi_preds["LTE DL User Throughput"][i])   if not np.isnan(kpi_preds["LTE DL User Throughput"][i])   else None

            if (prb is not None and prb > t["p0_prb"]
                    and tput is not None and tput < t["p0_thp"]
                    and usr  is not None and usr  > t["p0_users"]):
                risk = "P0"
            elif (prb is not None and prb > t["p1_prb"]
                    and tput is not None and tput < t["p1_thp"]
                    and usr  is not None and usr  > t["p1_users"]):
                risk = "P1"
            else:
                risk = "Not Congested"

            records.append({
                "Time": ft.isoformat(), "Cell Name": cn,
                "DL PRB Load": prb, "Active Users": usr,
                "LTE DL User Throughput": tput,
                "Hour": ft.hour, "Risk Level": risk,
                "is_forecast": True,
            })

    if not records:
        return {}

    fc_df    = pd.DataFrame(records)
    # summary stats only over predicted rows
    pred_df  = fc_df[fc_df["is_forecast"] == True]
    p0_cells = pred_df.loc[pred_df["Risk Level"] == "P0", "Cell Name"].nunique()
    p1_cells = pred_df.loc[pred_df["Risk Level"] == "P1", "Cell Name"].nunique()
    max_prb  = pred_df["DL PRB Load"].max()
    mean_prb = pred_df["DL PRB Load"].mean()

    return {
        "records": records,
        "summary": {
            "n_cells":  len(cells),
            "p0_cells": int(p0_cells),
            "p1_cells": int(p1_cells),
            "max_prb":  round(float(max_prb),  2) if pd.notna(max_prb)  else None,
            "mean_prb": round(float(mean_prb), 2) if pd.notna(mean_prb) else None,
        },
        "method":          "model+lag" if _XGB_MODELS else "lag-only",
        "horizon":         horizon_hours,
        "thresholds":      t,
        "prb_threshold":   t["p0_prb"],
        "users_threshold": t["p1_users"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _records_to_df(store: dict) -> pd.DataFrame:
    if not store or "records" not in store:
        return pd.DataFrame()
    df = pd.DataFrame(store["records"])
    if not df.empty:
        df["Time"] = pd.to_datetime(df["Time"])
    return df


def _empty_fig(msg="Run a forecast first"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font={"size": 15, "color": "#94a3b8"})
    fig.update_layout(
        template="plotly_white",
        xaxis={"visible": False}, yaxis={"visible": False},
        margin={"t": 10, "b": 10, "l": 10, "r": 10}, height=300,
    )
    return fig


def _badge(label, color, bg=None):
    bg = bg or (color + "18")
    return html.Span(label, style={
        "fontSize": "11px", "fontWeight": 700, "color": color,
        "background": bg, "border": f"1px solid {color}",
        "borderRadius": "6px", "padding": "2px 9px",
    })


def _forecast_status_ui(message: str, running: bool, error: bool):
    """Build the status div shown below the forecast controls."""
    if running:
        color, icon = _INFO, "⏳ "
    elif error:
        color, icon = _DANGER, ""
    elif message.startswith("✅"):
        color, icon = _SUCCESS, ""
    else:
        color, icon = "#64748b", "ℹ️  "

    # Progress bar when running
    with _FC_LOCK:
        prog  = _FC_STATUS.get("progress", 0)
        total = _FC_STATUS.get("total", 0)

    bar = None
    if running and total > 0:
        pct = int(100 * prog / total)
        bar = html.Div([
            html.Div(style={
                "height": "6px", "borderRadius": "3px",
                "background": "#e2e8f0", "overflow": "hidden",
                "width": "220px", "display": "inline-block",
                "verticalAlign": "middle", "marginLeft": "10px",
            }, children=[
                html.Div(style={
                    "height": "100%", "borderRadius": "3px",
                    "background": color, "width": f"{pct}%",
                    "transition": "width 0.4s ease",
                }),
            ]),
            html.Span(f"  {prog}/{total}", style={
                "fontSize": "11px", "color": "#64748b",
                "marginLeft": "8px", "verticalAlign": "middle",
            }),
        ], style={"display": "inline-flex", "alignItems": "center"})

    children = [
        html.Span(icon + message, style={"color": color, "fontWeight": 600}),
    ]
    if bar:
        children.append(bar)

    return children


def _cells_from_filter(regions, sites, cells, modes, start, end) -> list:
    """Derive the list of cell names from the global filter."""
    df = get_filtered_df(regions, sites, cells, modes, start, end)
    if df.empty or "Cell Name" not in df.columns:
        return []
    return df["Cell Name"].dropna().unique().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Model status bar ───────────────────────────────────────────────────────
@callback(
    Output("fc-model-status-bar", "children"),
    Input("fc-model-status-bar", "id"),
)
def update_model_status_bar(_):
    labels = {
        "DL PRB Load":            "PRB Model",
        "Active Users":           "Active Users Model",
        "LTE DL User Throughput": "Throughput Model",
    }
    badges = []
    for kpi, label in labels.items():
        loaded = kpi in _XGB_MODELS
        badges.append(
            _badge(f"✓ {label} (XGBoost)", _SUCCESS) if loaded
            else _badge(f"◌ {label} (lag fallback)", "#94a3b8")
        )
    return badges


# ── 2. Global-filter summary pill ─────────────────────────────────────────────
@callback(
    Output("fc-filter-summary", "children"),
    Input("region-filter",  "value"),
    Input("site-filter",    "value"),
    Input("cell-filter",    "value"),
    Input("mode-filter",    "value"),
    Input("date-filter",    "start_date"),
    Input("date-filter",    "end_date"),
)
def update_filter_summary(regions, sites, cells, modes, start, end):
    cell_list = _cells_from_filter(regions, sites, cells, modes, start, end)
    n = len(cell_list)
    if n == 0:
        return "⚠️  No cells match current filter"
    preview = ", ".join(cell_list[:3])
    suffix  = f" … +{n - 3} more" if n > 3 else ""
    return f"{n} cell{'s' if n != 1 else ''} selected: {preview}{suffix}"


# ── 3. Initial status message on load ─────────────────────────────────────────
@callback(
    Output("fc-forecast-status", "children"),
    Input("fc-forecast-status", "id"),
)
def init_forecast_status(_):
    with _FC_LOCK:
        ts    = _FC_STATUS.get("saved_ts")
        n     = _FC_STATUS.get("saved_cells", 0)
        cache = os.path.exists(_CACHE_PATH)
    if cache and ts:
        return _forecast_status_ui(f"✅  Week cache available — {n} cells · saved {ts}. "
                                   "Change filter to explore, or click Save Week Forecast to refresh.",
                                   False, False)
    if cache:
        return _forecast_status_ui("ℹ️  Week cache file found. Click Save Week Forecast to rebuild, "
                                   "or Run Forecast for on-demand results.",
                                   False, False)
    return _forecast_status_ui("Ready. Use Run Forecast for on-demand results, or "
                               "Save Week Forecast to pre-compute 7 days for all filtered cells.",
                               False, False)


# ── 4. Run Forecast → background thread with live progress ────────────────────
@callback(
    Output("fc-forecast-status", "children", allow_duplicate=True),
    Output("fc-forecast-poll",   "disabled", allow_duplicate=True),
    Input("fc-run-btn",          "n_clicks"),
    State("region-filter",       "value"),
    State("site-filter",         "value"),
    State("cell-filter",         "value"),
    State("mode-filter",         "value"),
    State("date-filter",         "start_date"),
    State("date-filter",         "end_date"),
    State("fc-horizon",          "value"),
    State("fc-p0-prb",           "value"),
    State("fc-p0-thp",           "value"),
    State("fc-p0-users",         "value"),
    State("fc-p1-prb",           "value"),
    State("fc-p1-thp",           "value"),
    State("fc-p1-users",         "value"),
    prevent_initial_call=True,
)
def run_forecast_callback(n_clicks,
                          regions, sites, cells, modes, start, end,
                          horizon,
                          p0_prb, p0_thp, p0_users,
                          p1_prb, p1_thp, p1_users):
    if not n_clicks:
        raise PreventUpdate

    with _FC_LOCK:
        if _FC_STATUS["running"]:
            return _forecast_status_ui("⏳ A forecast is already running — please wait…",
                                       True, False), False

    cell_list = _cells_from_filter(regions, sites, cells, modes, start, end)
    if not cell_list:
        return _forecast_status_ui(
            "❌  No cells match the current filter. Adjust Region / Site / Cell filters.",
            False, True), True

    def _f(v, d): return float(v) if v is not None else d
    thresholds = {
        "p0_prb":   _f(p0_prb,   90),  "p0_thp":   _f(p0_thp,   2000),
        "p0_users": _f(p0_users, 8),
        "p1_prb":   _f(p1_prb,   80),  "p1_thp":   _f(p1_thp,   4000),
        "p1_users": _f(p1_users, 4),
    }
    h = int(horizon or 24)

    with _FC_LOCK:
        _FC_STATUS["running"]     = True
        _FC_STATUS["message"]     = f"⏳ Starting forecast for {len(cell_list)} cells ({h} h)…"
        _FC_STATUS["error"]       = False
        _FC_STATUS["progress"]    = 0
        _FC_STATUS["total"]       = len(cell_list)
        _FC_STATUS["mode"]        = "live"
        _FC_STATUS["live_result"] = None

    threading.Thread(
        target=_run_live_forecast_thread,
        args=(cell_list, h, thresholds),
        daemon=True,
    ).start()

    return _forecast_status_ui(
        f"⏳ Starting forecast for {len(cell_list)} cells ({h} h)…",
        True, False), False   # enable poll


# ── 5. Save Week Forecast (background thread) ──────────────────────────────────
@callback(
    Output("fc-week-store",       "data"),
    Output("fc-forecast-status",  "children", allow_duplicate=True),
    Output("fc-forecast-poll",    "disabled", allow_duplicate=True),
    Input("fc-save-week-btn",     "n_clicks"),
    State("region-filter",        "value"),
    State("site-filter",          "value"),
    State("cell-filter",          "value"),
    State("mode-filter",          "value"),
    State("date-filter",          "start_date"),
    State("date-filter",          "end_date"),
    State("fc-p0-prb",            "value"),
    State("fc-p0-thp",            "value"),
    State("fc-p0-users",          "value"),
    State("fc-p1-prb",            "value"),
    State("fc-p1-thp",            "value"),
    State("fc-p1-users",          "value"),
    prevent_initial_call=True,
)
def save_week_forecast(n_clicks,
                       regions, sites, cells, modes, start, end,
                       p0_prb, p0_thp, p0_users, p1_prb, p1_thp, p1_users):
    if not n_clicks:
        raise PreventUpdate

    with _FC_LOCK:
        if _FC_STATUS["running"]:
            return no_update, _forecast_status_ui(
                "⏳ A forecast is already running — please wait…", True, False), False

    cell_list = _cells_from_filter(regions, sites, cells, modes, start, end)
    if not cell_list:
        return no_update, _forecast_status_ui(
            "❌  No cells match the current filter. Adjust Region / Site / Cell filters.", False, True), True

    def _f(v, d): return float(v) if v is not None else d
    thresholds = {
        "p0_prb":   _f(p0_prb,   90),  "p0_thp":   _f(p0_thp,   2000),
        "p0_users": _f(p0_users, 8),
        "p1_prb":   _f(p1_prb,   80),  "p1_thp":   _f(p1_thp,   4000),
        "p1_users": _f(p1_users, 4),
    }

    with _FC_LOCK:
        _FC_STATUS["running"]  = True
        _FC_STATUS["message"]  = f"⏳ Starting week forecast for {len(cell_list)} cells…"
        _FC_STATUS["error"]    = False
        _FC_STATUS["progress"] = 0
        _FC_STATUS["total"]    = len(cell_list)
        _FC_STATUS["mode"]     = "week"
        _FC_STATUS["live_result"] = None

    threading.Thread(
        target=_run_week_forecast_thread,
        args=(cell_list, thresholds),
        daemon=True,
    ).start()

    status = _forecast_status_ui(
        f"⏳ Starting week forecast for {len(cell_list)} cells…", True, False)
    meta = {"running": True, "cell_count": len(cell_list)}
    return meta, status, False   # enable poll interval


# ── 6. Poll week-forecast progress ────────────────────────────────────────────
@callback(
    Output("fc-forecast-status",  "children",  allow_duplicate=True),
    Output("fc-week-store",        "data",      allow_duplicate=True),
    Output("fc-forecast-poll",    "disabled",  allow_duplicate=True),
    Output("fc-forecast-store",   "data",      allow_duplicate=True),
    Input("fc-forecast-poll",     "n_intervals"),
    prevent_initial_call=True,
)
def poll_forecast_status(n_intervals):
    with _FC_LOCK:
        running     = _FC_STATUS["running"]
        message     = _FC_STATUS["message"]
        error       = _FC_STATUS["error"]
        mode        = _FC_STATUS.get("mode")
        ts          = _FC_STATUS.get("saved_ts")
        n_cells     = _FC_STATUS.get("saved_cells", 0)
        live_result = _FC_STATUS.get("live_result")

    status_ui = _forecast_status_ui(message, running, error)
    stop_poll = not running

    # Week-save completed → update week-store metadata
    if not running and not error and ts and mode == "week":
        week_meta = {"saved": True, "saved_ts": ts, "cell_count": n_cells}
    else:
        week_meta = no_update

    # Live forecast completed → push result to fc-forecast-store
    if not running and live_result is not None and mode == "live":
        store_data = live_result
        with _FC_LOCK:
            _FC_STATUS["live_result"] = None   # consume it
    else:
        store_data = no_update

    return status_ui, week_meta, stop_poll, store_data


# ── 7. Auto-load from cache when filter changes ────────────────────────────────
@callback(
    Output("fc-forecast-store",  "data",     allow_duplicate=True),
    Output("fc-forecast-status", "children", allow_duplicate=True),
    Input("region-filter",   "value"),
    Input("site-filter",     "value"),
    Input("cell-filter",     "value"),
    Input("mode-filter",     "value"),
    State("fc-week-store",   "data"),
    State("fc-p0-prb",       "value"),
    State("fc-p0-thp",       "value"),
    State("fc-p0-users",     "value"),
    State("fc-p1-prb",       "value"),
    State("fc-p1-thp",       "value"),
    State("fc-p1-users",     "value"),
    State("date-filter",     "start_date"),
    State("date-filter",     "end_date"),
    prevent_initial_call=True,
)
def auto_load_from_cache(regions, sites, cells, modes,
                         week_meta,
                         p0_prb, p0_thp, p0_users, p1_prb, p1_thp, p1_users,
                         start, end):
    """When the filter changes and a week-cache file exists, reload the store."""
    if not week_meta or not week_meta.get("saved"):
        raise PreventUpdate
    if not os.path.exists(_CACHE_PATH):
        raise PreventUpdate

    cell_list = _cells_from_filter(regions, sites, cells, modes, start, end)
    if not cell_list:
        raise PreventUpdate

    try:
        cache_df = pd.read_csv(_CACHE_PATH)
        cache_df["Time"] = pd.to_datetime(cache_df["Time"])
    except Exception:
        raise PreventUpdate

    # Filter cached data to the cells currently in scope
    present = set(cache_df["Cell Name"].unique())
    matched = [c for c in cell_list if c in present]
    if not matched:
        status = _forecast_status_ui(
            f"⚠️  None of the {len(cell_list)} filtered cells appear in the saved cache. "
            "Run Save Week Forecast again with the current filter.", False, True)
        return no_update, status

    sub = cache_df[cache_df["Cell Name"].isin(matched)]

    # Pivot from long format (KPI column) → wide records
    wide = sub.pivot_table(index=["Time", "Cell Name", "Hour"],
                           columns="KPI", values="Value",
                           aggfunc="first").reset_index()
    wide.columns.name = None

    def _f(v, d): return float(v) if v is not None else d
    thresholds = {
        "p0_prb":   _f(p0_prb,   90),  "p0_thp":   _f(p0_thp,   2000),
        "p0_users": _f(p0_users, 8),
        "p1_prb":   _f(p1_prb,   80),  "p1_thp":   _f(p1_thp,   4000),
        "p1_users": _f(p1_users, 4),
    }
    t = thresholds

    from congestion_engine import DEFAULT_THRESHOLDS as _DT
    t = {**_DT, **t}

    records = []
    for _, row in wide.iterrows():
        prb  = float(row["DL PRB Load"])            if "DL PRB Load"            in row and pd.notna(row["DL PRB Load"])            else None
        usr  = float(row["Active Users"])           if "Active Users"           in row and pd.notna(row["Active Users"])           else None
        tput = float(row["LTE DL User Throughput"]) if "LTE DL User Throughput" in row and pd.notna(row["LTE DL User Throughput"]) else None

        if (prb is not None and prb > t["p0_prb"]
                and tput is not None and tput < t["p0_thp"]
                and usr  is not None and usr  > t["p0_users"]):
            risk = "P0"
        elif (prb is not None and prb > t["p1_prb"]
                and tput is not None and tput < t["p1_thp"]
                and usr  is not None and usr  > t["p1_users"]):
            risk = "P1"
        else:
            risk = "Not Congested"

        records.append({
            "Time":                   row["Time"].isoformat(),
            "Cell Name":              row["Cell Name"],
            "DL PRB Load":            prb,
            "Active Users":           usr,
            "LTE DL User Throughput": tput,
            "Hour":                   int(row["Hour"]),
            "Risk Level":             risk,
        })

    if not records:
        raise PreventUpdate

    fc_df    = pd.DataFrame(records)
    p0_cells = fc_df.loc[fc_df["Risk Level"] == "P0", "Cell Name"].nunique()
    p1_cells = fc_df.loc[fc_df["Risk Level"] == "P1", "Cell Name"].nunique()
    max_prb  = fc_df["DL PRB Load"].max()
    mean_prb = fc_df["DL PRB Load"].mean()

    store = {
        "records": records,
        "summary": {
            "n_cells":  len(matched),
            "p0_cells": int(p0_cells),
            "p1_cells": int(p1_cells),
            "max_prb":  round(float(max_prb),  2) if pd.notna(max_prb)  else None,
            "mean_prb": round(float(mean_prb), 2) if pd.notna(mean_prb) else None,
        },
        "method":          "model+lag" if _XGB_MODELS else "lag-only",
        "horizon":         168,
        "thresholds":      t,
        "prb_threshold":   t["p0_prb"],
        "users_threshold": t["p1_users"],
        "source":          "cache",
    }

    ts = week_meta.get("saved_ts", "")
    status = _forecast_status_ui(
        f"📂  Loaded from saved cache ({len(matched)} cells · saved {ts}). "
        "Charts reflect the current filter.", False, False)
    return store, status


# ── 8. Summary cards ──────────────────────────────────────────────────────────
@callback(
    Output("fc-summary-cards", "children"),
    Input("fc-forecast-store", "data"),
)
def update_summary_cards(store):
    if not store or "summary" not in store:
        return []
    s      = store["summary"]
    method = store.get("method", "lag-only")
    source = store.get("source", "live")

    metrics = [
        ("Cells Forecasted",  str(s.get("n_cells",  0)),  _ACCENT),
        ("P0 Risk Cells",     str(s.get("p0_cells", 0)),  _DANGER),
        ("P1 Risk Cells",     str(s.get("p1_cells", 0)),  _WARN),
        ("Max PRB Load (%)",  str(s.get("max_prb",  "—")), "#0369a1"),
        ("Avg PRB Load (%)",  str(s.get("mean_prb", "—")), "#0369a1"),
    ]
    cards = [
        html.Div([
            html.Div(label, style={"fontSize": "12px", "color": "#64748b", "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": 700, "color": color}),
        ], style=CARD_STYLE)
        for label, value, color in metrics
    ]
    method_text = "Neural Network + Lag Blend" if method == "model+lag" else "Lag Blend (no model)"
    source_text = " · from cache" if source == "cache" else ""
    method_badge = html.Div([
        html.Div("Forecast method", style={"fontSize": "12px", "color": "#64748b", "marginBottom": "4px"}),
        html.Div(method_text + source_text,
                 style={"fontSize": "13px", "fontWeight": 700,
                        "color": _SUCCESS if method == "model+lag" else "#64748b"}),
    ], style=CARD_STYLE)
    return cards + [method_badge]


# ── 9. Alert banner ────────────────────────────────────────────────────────────
@callback(
    Output("fc-alert-banner", "children"),
    Input("fc-forecast-store", "data"),
)
def update_alert_banner(store):
    if not store or "summary" not in store:
        return []
    s  = store["summary"]
    p0 = s.get("p0_cells", 0)
    p1 = s.get("p1_cells", 0)
    if p0 == 0 and p1 == 0:
        return html.Div("✅  No cells are predicted to be congested during this forecast window.",
                        style={"padding": "10px 16px", "background": "#dcfce7",
                               "border": "1px solid #86efac", "borderRadius": "10px",
                               "fontSize": "13px", "color": "#14532d", "fontWeight": 600})
    parts = []
    if p0:
        parts.append(html.Span(f"⚠️  {p0} cell(s) predicted P0 (critical congestion)",
                               style={"color": _DANGER, "fontWeight": 700}))
    if p1:
        if parts:
            parts.append(html.Span("  ·  ", style={"color": "#64748b"}))
        parts.append(html.Span(f"{p1} cell(s) predicted P1 (moderate congestion)",
                               style={"color": _WARN, "fontWeight": 700}))
    return html.Div(parts, style={
        "padding": "10px 16px",
        "background": "#fff7ed" if p0 == 0 else "#fee2e2",
        "border": f"1px solid {P1_COLOR if p0 == 0 else P0_COLOR}",
        "borderRadius": "10px", "fontSize": "13px",
    })


# ── 10. Main forecast chart ────────────────────────────────────────────────────
@callback(
    Output("fc-main-chart", "figure"),
    Input("fc-forecast-store", "data"),
    Input("fc-kpi-selector",   "value"),
)
def update_main_chart(store, kpi):
    if not store or "records" not in store:
        return _empty_fig()
    fc_df = _records_to_df(store)
    if fc_df.empty or kpi not in fc_df.columns:
        return _empty_fig()

    prb_thr = store.get("prb_threshold", DEFAULT_PRB_THRESHOLD)
    cells   = fc_df["Cell Name"].unique().tolist()
    colors  = px.colors.qualitative.Plotly
    fig     = go.Figure()

    # add is_forecast column if missing (backward compat with cache loads)
    if "is_forecast" not in fc_df.columns:
        fc_df["is_forecast"] = True

    for idx, cn in enumerate(cells):
        clr      = colors[idx % len(colors)]
        cell_df  = fc_df[fc_df["Cell Name"] == cn].sort_values("Time")
        real_df  = cell_df[cell_df["is_forecast"] == False]
        pred_df  = cell_df[cell_df["is_forecast"] == True]

        # ── Historical context: dotted, from DF, before the overlap window ──
        if not DF.empty and kpi in DF.columns and not real_df.empty:
            overlap_start = real_df["Time"].min()
            hist_start    = overlap_start - pd.Timedelta(hours=48)
            h = DF[(DF["Cell Name"] == cn) &
                   (DF["Time"] >= hist_start) &
                   (DF["Time"] < overlap_start)].sort_values("Time")
            if not h.empty:
                fig.add_trace(go.Scatter(
                    x=h["Time"], y=h[kpi], mode="lines",
                    name=f"{cn} (history)", legendgroup=cn,
                    line={"color": clr, "width": 1.5, "dash": "dot"},
                    opacity=0.45, showlegend=True,
                ))

        # ── Overlap: real values in forecast window — solid, slightly faded ─
        if not real_df.empty and kpi in real_df.columns:
            fig.add_trace(go.Scatter(
                x=real_df["Time"], y=real_df[kpi].astype(float),
                mode="lines", name=f"{cn} (actual)", legendgroup=cn,
                line={"color": clr, "width": 2},
                opacity=0.75, showlegend=False,
            ))

        # ── Prediction: dashed with markers, starts right after real data ──
        if not pred_df.empty and kpi in pred_df.columns:
            y_pred = pred_df[kpi].astype(float)

            # Stitch: prepend the last real point so there's no visual gap
            if not real_df.empty:
                stitch_t = real_df["Time"].iloc[-1:]
                stitch_y = real_df[kpi].astype(float).iloc[-1:]
                x_plot   = pd.concat([stitch_t, pred_df["Time"]])
                y_plot   = pd.concat([stitch_y, y_pred])
            else:
                x_plot, y_plot = pred_df["Time"], y_pred

            fig.add_trace(go.Scatter(
                x=x_plot, y=y_plot,
                mode="lines+markers", name=f"{cn} (forecast)", legendgroup=cn,
                line={"color": clr, "width": 2.5, "dash": "dash"},
                marker={"size": 4},
                showlegend=True,
            ))

            # P0 congestion markers
            p0_mask = pred_df["Risk Level"] == "P0"
            if p0_mask.any():
                fig.add_trace(go.Scatter(
                    x=pred_df.loc[p0_mask, "Time"],
                    y=pred_df.loc[p0_mask, kpi],
                    mode="markers", name=f"{cn} P0",
                    marker={"color": P0_COLOR, "size": 9, "symbol": "circle"},
                    legendgroup=cn, showlegend=False,
                ))

    # ── Threshold lines & forecast-start marker ────────────────────────────
    if kpi == "DL PRB Load":
        fig.add_hline(y=prb_thr,
                      line={"color": P0_COLOR, "dash": "dash", "width": 1.5},
                      annotation_text=f"P0 ({prb_thr}%)",
                      annotation_position="top right")
        fig.add_hline(y=prb_thr * 0.75,
                      line={"color": P1_COLOR, "dash": "dash", "width": 1},
                      annotation_text=f"P1 ({prb_thr * 0.75:.0f}%)",
                      annotation_position="bottom right")

    # Vertical line at last-real-data point
    pred_rows = fc_df[fc_df["is_forecast"] == True]
    if not pred_rows.empty:
        forecast_start = pred_rows["Time"].min()
        fig.add_vline(
            x=forecast_start.timestamp() * 1000,
            line={"color": "#94a3b8", "dash": "dot", "width": 1.5},
            annotation_text="Forecast start",
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"Forecast — {kpi}",
        template="plotly_white", height=400,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0},
        xaxis_title="Time", yaxis_title=kpi,
        margin={"t": 55, "b": 40, "l": 55, "r": 20},
    )
    return fig


# ── 11. Heatmap ────────────────────────────────────────────────────────────────
@callback(
    Output("fc-heatmap", "figure"),
    Input("fc-forecast-store", "data"),
)
def update_heatmap(store):
    if not store or "records" not in store:
        return _empty_fig()
    fc_df = _records_to_df(store)
    if fc_df.empty:
        return _empty_fig()

    # Use only predicted rows for the risk heatmap
    if "is_forecast" in fc_df.columns:
        fc_df = fc_df[fc_df["is_forecast"] == True]
    if fc_df.empty:
        return _empty_fig()

    cells     = sorted(fc_df["Cell Name"].unique())
    _risk_num = {"Not Congested": 0, "P1": 1, "P0": 2}
    z = np.zeros((len(cells), 24), dtype=float)
    for i, cn in enumerate(cells):
        for _, row in fc_df[fc_df["Cell Name"] == cn].iterrows():
            h = int(row["Hour"]) % 24
            z[i, h] = max(z[i, h], _risk_num.get(row["Risk Level"], 0))

    colorscale = [
        [0.0,  "#dcfce7"], [0.49, "#dcfce7"],
        [0.5,  "#fed7aa"], [0.74, "#fed7aa"],
        [0.75, "#fee2e2"], [1.0,  "#fee2e2"],
    ]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{h:02d}:00" for h in range(24)], y=cells,
        colorscale=colorscale, zmin=0, zmax=2,
        colorbar={"tickvals": [0, 1, 2], "ticktext": ["Clear", "P1", "P0"],
                  "thickness": 14, "len": 0.7},
    ))
    fig.update_layout(
        template="plotly_white",
        height=max(200, 28 * len(cells) + 80),
        margin={"t": 10, "b": 50, "l": 10, "r": 60},
        xaxis_title="Hour of Day", yaxis_title="", yaxis={"tickfont": {"size": 10}},
    )
    return fig


# ── 12. PRB box plot ───────────────────────────────────────────────────────────
@callback(
    Output("fc-prb-box", "figure"),
    Input("fc-forecast-store", "data"),
)
def update_prb_box(store):
    if not store or "records" not in store:
        return _empty_fig()
    fc_df   = _records_to_df(store)
    prb_thr = store.get("prb_threshold", DEFAULT_PRB_THRESHOLD)
    if fc_df.empty or "DL PRB Load" not in fc_df.columns:
        return _empty_fig()

    fig = go.Figure()
    for cn in sorted(fc_df["Cell Name"].unique()):
        vals  = fc_df.loc[fc_df["Cell Name"] == cn, "DL PRB Load"].dropna()
        max_v = vals.max() if not vals.empty else 0
        clr   = P0_COLOR if max_v >= prb_thr else (P1_COLOR if max_v >= prb_thr * 0.75 else _SUCCESS)
        fig.add_trace(go.Box(y=vals, name=cn, marker_color=clr,
                             boxpoints="outliers", line={"width": 1.5}))
    fig.add_hline(y=prb_thr, line={"color": P0_COLOR, "dash": "dash", "width": 1.5},
                  annotation_text=f"{prb_thr}% P0")
    fig.update_layout(template="plotly_white", showlegend=False, height=300,
                      yaxis_title="DL PRB Load (%)", xaxis_title="",
                      margin={"t": 10, "b": 40, "l": 50, "r": 10})
    return fig


# ── 13. Time-to-breach ────────────────────────────────────────────────────────
@callback(
    Output("fc-time-to-breach", "figure"),
    Input("fc-forecast-store",  "data"),
)
def update_time_to_breach(store):
    if not store or "records" not in store:
        return _empty_fig()
    fc_df   = _records_to_df(store)
    prb_thr = store.get("prb_threshold", DEFAULT_PRB_THRESHOLD)
    if fc_df.empty or "DL PRB Load" not in fc_df.columns:
        return _empty_fig()

    # Only future predicted rows
    if "is_forecast" in fc_df.columns:
        fc_df = fc_df[fc_df["is_forecast"] == True]

    rows = []
    for cn, grp in fc_df.groupby("Cell Name"):
        breach = grp.loc[grp["DL PRB Load"] >= prb_thr].sort_values("Time")
        if not breach.empty:
            hours_until = int((breach["Time"].min() - fc_df["Time"].min()).total_seconds() / 3600) + 1
            rows.append({"Cell Name": cn, "Hours Until Breach": hours_until})

    if not rows:
        return _empty_fig("No cells forecast to breach the P0 threshold 🎉")

    ttb = pd.DataFrame(rows).sort_values("Hours Until Breach")
    fig = px.bar(ttb, x="Hours Until Breach", y="Cell Name", orientation="h",
                 color="Hours Until Breach",
                 color_continuous_scale=["#ef4444", "#f97316", "#fde68a"],
                 labels={"Hours Until Breach": "Hours"})
    fig.update_layout(template="plotly_white", showlegend=False,
                      height=max(220, 32 * len(rows) + 80),
                      margin={"t": 10, "b": 40, "l": 10, "r": 10},
                      coloraxis_showscale=False, yaxis_title="")
    return fig


# ── 14. Busy hours ─────────────────────────────────────────────────────────────
@callback(
    Output("fc-busy-hours", "figure"),
    Input("fc-forecast-store", "data"),
)
def update_busy_hours(store):
    if not store or "records" not in store:
        return _empty_fig()
    fc_df = _records_to_df(store)
    if fc_df.empty or "DL PRB Load" not in fc_df.columns:
        return _empty_fig()

    # Only future predicted rows
    if "is_forecast" in fc_df.columns:
        fc_df = fc_df[fc_df["is_forecast"] == True]

    avg_by_hour = (fc_df.groupby("Hour")["DL PRB Load"]
                       .mean().reindex(range(24), fill_value=0).reset_index())
    avg_by_hour.columns = ["Hour", "Avg PRB Load"]
    avg_by_hour["Label"] = avg_by_hour["Hour"].apply(lambda h: f"{h:02d}:00")
    prb_thr = store.get("prb_threshold", DEFAULT_PRB_THRESHOLD)
    avg_by_hour["Color"] = avg_by_hour["Avg PRB Load"].apply(
        lambda v: P0_COLOR if v >= prb_thr else (P1_COLOR if v >= prb_thr * 0.75 else _SUCCESS))

    fig = go.Figure(go.Bar(x=avg_by_hour["Label"], y=avg_by_hour["Avg PRB Load"],
                           marker_color=avg_by_hour["Color"]))
    fig.add_hline(y=prb_thr, line={"color": P0_COLOR, "dash": "dash", "width": 1.5})
    fig.update_layout(template="plotly_white", height=300,
                      xaxis_title="Hour of Day", yaxis_title="Avg DL PRB Load (%)",
                      margin={"t": 10, "b": 40, "l": 50, "r": 10})
    return fig


# ── 15. Alert table ────────────────────────────────────────────────────────────
@callback(
    Output("fc-alert-table",   "data"),
    Output("fc-alert-table",   "columns"),
    Input("fc-forecast-store", "data"),
)
def update_alert_table(store):
    empty_cols = [{"name": c, "id": c} for c in
                  ["Cell Name", "Risk Level", "Max PRB Load (%)",
                   "Max Active Users", "P0 Hours", "P1 Hours", "First Breach"]]
    if not store or "records" not in store:
        return [], empty_cols
    fc_df = _records_to_df(store)
    if fc_df.empty:
        return [], empty_cols

    # Only future predicted rows for the alert table
    if "is_forecast" in fc_df.columns:
        fc_df = fc_df[fc_df["is_forecast"] == True]
    if fc_df.empty:
        return [], empty_cols

    rows = []
    for cn, grp in fc_df.groupby("Cell Name"):
        p0_h = int((grp["Risk Level"] == "P0").sum())
        p1_h = int((grp["Risk Level"] == "P1").sum())
        risk = "P0" if p0_h else ("P1" if p1_h else "Not Congested")
        if risk == "Not Congested":
            continue
        breach_rows  = grp[grp["Risk Level"].isin(["P0", "P1"])].sort_values("Time")
        first_breach = breach_rows["Time"].min().strftime("%Y-%m-%d %H:%M") if not breach_rows.empty else "—"
        rows.append({
            "Cell Name":        cn, "Risk Level": risk,
            "Max PRB Load (%)": round(float(grp["DL PRB Load"].max()), 1) if pd.notna(grp["DL PRB Load"].max()) else "—",
            "Max Active Users": round(float(grp["Active Users"].max()), 1) if pd.notna(grp["Active Users"].max()) else "—",
            "P0 Hours":  p0_h, "P1 Hours": p1_h,
            "First Breach": first_breach,
        })

    if not rows:
        return [], empty_cols
    rows.sort(key=lambda r: (0 if r["Risk Level"] == "P0" else 1, r["First Breach"]))
    return rows, [{"name": c, "id": c} for c in rows[0].keys()]


# ── 16. CSV export ─────────────────────────────────────────────────────────────
@callback(
    Output("fc-download",      "data"),
    Input("fc-export-btn",     "n_clicks"),
    State("fc-forecast-store", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, store):
    if not store or "records" not in store:
        raise PreventUpdate
    fc_df = _records_to_df(store)
    return dcc.send_data_frame(fc_df.to_csv, "forecast.csv", index=False)


# ── 17. Run-spinner sync ───────────────────────────────────────────────────────
@callback(
    Output("fc-run-spinner-target", "children"),
    Input("fc-forecast-store", "data"),
)
def _spinner_sync(store):
    return None


# ── 18. Weekly risk calendar ───────────────────────────────────────────────────
_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

@callback(
    Output("fc-weekly-calendar", "figure"),
    Input("fc-forecast-store", "data"),
)
def update_weekly_calendar(store: dict):
    if not store or "records" not in store:
        fig = go.Figure()
        fig.update_layout(title="Run a ≥ 168 h forecast to see the weekly calendar",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    df = pd.DataFrame(store["records"])
    if "Time" not in df.columns or "DL PRB Load" not in df.columns:
        return go.Figure()

    df["Time"]    = pd.to_datetime(df["Time"])
    df["Hour"]    = df["Time"].dt.hour
    df["DayOfWk"] = df["Time"].dt.dayofweek
    df["Day"]     = df["DayOfWk"].map(dict(enumerate(_DAY_NAMES)))

    pivot = (df.groupby(["Hour", "Day"])["DL PRB Load"]
               .mean().unstack("Day").reindex(columns=_DAY_NAMES).fillna(0))

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=_DAY_NAMES, y=[f"{h:02d}h" for h in pivot.index],
        colorscale=[[0.0, "#22c55e"], [0.5, "#fbbf24"], [0.75, "#f97316"], [1.0, "#ef4444"]],
        zmin=0, zmax=100,
        colorbar={"title": "Avg PRB %", "thickness": 14, "tickfont": {"size": 10}},
        hovertemplate="<b>%{x} %{y}</b><br>Avg PRB: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title={"text": "Weekly Risk Calendar — Avg Predicted DL PRB Load",
               "font": {"size": 14, "color": "#0f172a"}, "x": 0.02},
        xaxis={"title": "Day of Week"},
        yaxis={"title": "Hour of Day", "autorange": "reversed"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "size": 11},
        margin={"t": 48, "b": 40, "l": 60, "r": 20},
    )
    return fig


# ── 19. What-if simulator ──────────────────────────────────────────────────────
@callback(
    Output("fc-whatif-output", "children"),
    Output("fc-whatif-chart",  "figure"),
    Input("fc-growth-slider",  "value"),
    Input("fc-forecast-store", "data"),
)
def update_whatif(multiplier: float, store: dict):
    empty_fig = go.Figure()
    empty_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    if not store or "records" not in store:
        return "Run a forecast first.", empty_fig

    df = pd.DataFrame(store["records"])
    if "DL PRB Load" not in df.columns:
        return "No PRB data in forecast.", empty_fig

    mult          = float(multiplier or 1.0)
    prb_threshold = float(store.get("prb_threshold", 90))
    df["ScaledPRB"] = (df["DL PRB Load"].fillna(0) * mult).clip(0, 100)

    cell_summary = (df.groupby("Cell Name")
                      .agg(BaseMax=("DL PRB Load", "max"), ScaledMax=("ScaledPRB", "max"),
                           ScaledMean=("ScaledPRB", "mean"))
                      .reset_index())
    cell_summary["Risk"] = cell_summary["ScaledMax"].apply(
        lambda v: "P0 ⚠" if v >= prb_threshold else ("P1" if v >= prb_threshold * 0.75 else "✓ Clear"))

    n_p0 = (cell_summary["Risk"] == "P0 ⚠").sum()
    n_p1 = (cell_summary["Risk"] == "P1").sum()
    n_ok = (cell_summary["Risk"] == "✓ Clear").sum()
    tot  = len(cell_summary)

    summary_children = [
        html.Div(f"×{mult:.1f} traffic growth", style={"fontWeight": 700, "marginBottom": "6px", "color": "#0f172a"}),
        html.Div([
            html.Span(f"🔴 P0: {n_p0}/{tot}", style={"color": "#ef4444", "fontWeight": 600, "marginRight": "14px"}),
            html.Span(f"🟠 P1: {n_p1}/{tot}", style={"color": "#f97316", "fontWeight": 600, "marginRight": "14px"}),
            html.Span(f"✅ Clear: {n_ok}/{tot}", style={"color": "#22c55e", "fontWeight": 600}),
        ]),
    ]

    cell_summary = cell_summary.sort_values("ScaledMax", ascending=False).head(20)
    color_map    = {"P0 ⚠": "#ef4444", "P1": "#f97316", "✓ Clear": "#22c55e"}
    colors       = cell_summary["Risk"].map(color_map).fillna("#94a3b8").tolist()

    fig = go.Figure()
    fig.add_bar(x=cell_summary["ScaledMax"], y=cell_summary["Cell Name"],
                orientation="h", marker_color=colors,
                hovertemplate="%{y}<br>Scaled Max PRB: %{x:.1f}%<extra></extra>")
    fig.add_vline(x=prb_threshold, line={"color": "#ef4444", "dash": "dash", "width": 1.5},
                  annotation_text=f"P0 ({prb_threshold}%)", annotation_position="top right")
    fig.update_layout(template="plotly_white", showlegend=False,
                      height=max(200, 28 * len(cell_summary) + 80),
                      xaxis_title="Scaled Max PRB Load (%)", yaxis_title="",
                      margin={"t": 30, "b": 40, "l": 10, "r": 80},
                      yaxis={"autorange": "reversed"})
    return summary_children, fig


# ─────────────────────────────────────────────────────────────────────────────
# Retrain callbacks  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────

def _retrain_status_ui(message: str, running: bool, error: bool):
    if running:
        color, prefix = "#0369a1", "⏳ "
    elif error:
        color, prefix = "#b91c1c", ""
    else:
        color, prefix = "#15803d" if message.startswith("✅") else "#64748b", ""
    return html.Div(prefix + message, style={
        "fontSize": "12px", "color": color, "fontWeight": 500,
        "marginTop": "10px", "whiteSpace": "pre-wrap",
    })


@callback(
    Output("retrain-status",   "children"),
    Output("retrain-poll",     "disabled"),
    Input("retrain-models-btn", "n_clicks"),
    State("train-data-path",   "value"),
    prevent_initial_call=True,
)
def retrain_models(n_clicks, data_path):
    if not n_clicks:
        raise PreventUpdate
    with _TRAIN_LOCK:
        if _TRAIN_STATUS["running"]:
            return _retrain_status_ui("Training already in progress …", True, False), False

    path = (data_path or "").strip()
    if not path:
        return _retrain_status_ui("❌  No data file path specified.", False, True), True
    if not os.path.exists(path):
        return _retrain_status_ui(f"❌  File not found: {path}", False, True), True

    with _TRAIN_LOCK:
        _TRAIN_STATUS["running"] = True
        _TRAIN_STATUS["message"] = "Starting …"
        _TRAIN_STATUS["error"]   = False

    threading.Thread(target=_run_training_thread, args=(path,), daemon=True).start()
    return _retrain_status_ui("Starting …", True, False), False


@callback(
    Output("retrain-status", "children", allow_duplicate=True),
    Output("retrain-poll",   "disabled", allow_duplicate=True),
    Input("retrain-poll",    "n_intervals"),
    prevent_initial_call=True,
)
def poll_retrain_status(n_intervals):
    with _TRAIN_LOCK:
        running = _TRAIN_STATUS["running"]
        message = _TRAIN_STATUS["message"]
        error   = _TRAIN_STATUS["error"]
    return _retrain_status_ui(message, running, error), not running
