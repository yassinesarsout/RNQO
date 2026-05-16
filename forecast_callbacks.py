"""
Forecast tab callbacks — AI-powered forecasting for RAN KPIs.

Forecasting engine:
  1. Tries to load XGBoost models + sklearn scalers from  Models/
     (trained by Models/train_xgboost.py).
  2. If a model + scaler pair is found  → uses XGBoost with the full
     22-feature vector (identity codes, 9 lags, rolling mean, time features,
     FDD/TDD dummies) matching the training notebooks exactly.
  3. For any missing model             → falls back to lag-blending
     (average of T-24h, T-168h, T-336h from historical data).

Congestion detection (3-KPI AND logic, matches congestion_engine.py):
  P0 → PRB > p0_prb  AND  Throughput < p0_thp  AND  Users > p0_users
  P1 → PRB > p1_prb  AND  Throughput < p1_thp  AND  Users > p1_users
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
from precompute import DF

_ACCENT  = "#6366f1"
_SUCCESS = "#22c55e"
_WARN    = "#f97316"
_DANGER  = "#ef4444"

# ── XGBoost model directory ───────────────────────────────────────────────────
_XGB_DIR = os.path.join(os.path.dirname(__file__), "Models")

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

_XGB_MODELS:  dict = {}   # kpi → xgb.XGBRegressor
_XGB_SCALERS: dict = {}   # kpi → sklearn scaler
_XGB_META:    dict = {}   # loaded from meta.json

# Per-KPI feature lists (set after meta is loaded; fallback hardcoded below)
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
    """Load XGBoost models, scalers, and meta.json once at import time."""
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
# Model retraining — background thread + status
# ─────────────────────────────────────────────────────────────────────────────

_TRAIN_LOCK   = threading.Lock()
_TRAIN_STATUS = {"running": False, "message": "Idle", "error": False}


def _reload_xgb_models():
    """Refresh all in-process model dicts from disk without server restart."""
    _XGB_MODELS.clear()
    _XGB_SCALERS.clear()
    _XGB_META.clear()
    _XGB_FEATURES.clear()
    _try_load_xgb()


def _run_training_thread(data_path: str):
    """Runs in a daemon thread — trains models, reloads them, updates status."""
    def _progress(msg: str):
        with _TRAIN_LOCK:
            _TRAIN_STATUS["message"] = msg

    # Clear current models so dashboard falls back to lag-blend while training
    _XGB_MODELS.clear()

    try:
        # Import train_all from Models/train_xgboost.py
        if _XGB_DIR not in sys.path:
            sys.path.insert(0, _XGB_DIR)
        import importlib
        tx = importlib.import_module("train_xgboost")
        importlib.reload(tx)   # pick up any edits since last import

        result = tx.train_all(
            data_path=data_path,
            models_dir=_XGB_DIR,
            progress_cb=_progress,
        )
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
# Core forecast engine
# ─────────────────────────────────────────────────────────────────────────────

def _make_cell_index(df: pd.DataFrame) -> dict:
    """Return {cell_name: DataFrame indexed by Time} for all 3 KPIs."""
    kpis = ["DL PRB Load", "Active Users", "LTE DL User Throughput"]
    idx = {}
    for cn, grp in df.groupby("Cell Name"):
        g = grp.sort_values("Time").drop_duplicates("Time").set_index("Time")
        idx[str(cn)] = g[[k for k in kpis if k in g.columns]]
    return idx


def _lag_lookup(series: pd.Series, t: pd.Timestamp, h: int):
    """Safe lookup of a single lag value."""
    try:
        return series.at[t - pd.Timedelta(hours=h)]
    except KeyError:
        return np.nan


def _rolling24(series: pd.Series, t: pd.Timestamp) -> float:
    """Mean of the 24 real-history hours ending just before t."""
    lo = t - pd.Timedelta(hours=24)
    hi = t - pd.Timedelta(hours=1)
    window = series.loc[lo:hi]
    return float(window.mean()) if not window.empty else 0.0


def _cell_fdd(cell_name: str) -> str:
    if DF.empty or "Cell FDD TDD Indication" not in DF.columns:
        return ""
    rows = DF.loc[DF["Cell Name"] == cell_name, "Cell FDD TDD Indication"]
    return str(rows.iloc[0]) if not rows.empty else ""


def _build_xgb_rows(kpi: str, cell_name: str, forecast_times: list,
                    hist) -> list[dict]:
    """Build one feature-dict per forecast timestamp for the XGBoost model."""
    meta     = _XGB_META
    fdd_cols = meta.get("fdd_cols", [])

    cell_id   = meta.get("cell_codes",   {}).get(cell_name, 0)
    site_name = meta.get("cell_to_site", {}).get(cell_name, "")
    reg_name  = meta.get("cell_to_region", {}).get(cell_name, "")
    site_id   = meta.get("site_codes",   {}).get(site_name, 0)
    region_id = meta.get("region_codes", {}).get(reg_name,  0)

    fdd_val = _cell_fdd(cell_name)

    # Determine which column holds the target's rolling feature
    _rolling_col = {
        "DL PRB Load":            "prb_rolling_24h",
        "Active Users":           "users_rolling_24h",
        "LTE DL User Throughput": "throughput_rolling_24h",
    }[kpi]
    _rolling_kpi = {
        "prb_rolling_24h":         "DL PRB Load",
        "users_rolling_24h":       "Active Users",
        "throughput_rolling_24h":  "LTE DL User Throughput",
    }[_rolling_col]

    rows = []
    for ft in forecast_times:
        def lag(k, h):
            if hist is None or k not in hist.columns:
                return 0.0
            v = _lag_lookup(hist[k], ft, h)
            return float(v) if not np.isnan(v) else 0.0

        rolling_val = (
            _rolling24(hist[_rolling_kpi], ft)
            if hist is not None and _rolling_kpi in hist.columns
            else 0.0
        )

        row = {
            "cell_id":   cell_id,
            "site_id":   site_id,
            "region_id": region_id,
            "prb_lag_1d":        lag("DL PRB Load",            24),
            "throughput_lag_1d": lag("LTE DL User Throughput", 24),
            "users_lag_1d":      lag("Active Users",           24),
            "prb_lag_1w":        lag("DL PRB Load",            168),
            "throughput_lag_1w": lag("LTE DL User Throughput", 168),
            "users_lag_1w":      lag("Active Users",           168),
            "prb_lag_2w":        lag("DL PRB Load",            336),
            "throughput_lag_2w": lag("LTE DL User Throughput", 336),
            "users_lag_2w":      lag("Active Users",           336),
            _rolling_col:        rolling_val,
            "hour_sin":   np.sin(2 * np.pi * ft.hour      / 24),
            "hour_cos":   np.cos(2 * np.pi * ft.hour      / 24),
            "dow_sin":    np.sin(2 * np.pi * ft.dayofweek / 7),
            "dow_cos":    np.cos(2 * np.pi * ft.dayofweek / 7),
            "month_sin":  np.sin(2 * np.pi * ft.month     / 12),
            "month_cos":  np.cos(2 * np.pi * ft.month     / 12),
            "is_weekend": int(ft.dayofweek >= 5),
        }
        for col in fdd_cols:
            v = col.replace("Cell FDD TDD Indication_", "")
            row[col] = 1 if fdd_val == v else 0
        rows.append(row)
    return rows


def _forecast_one_cell_one_kpi(
    kpi: str,
    cell_name: str,
    forecast_times: list,
    cell_index: dict,
) -> np.ndarray:
    """
    Predict `kpi` for `cell_name` at each timestamp in `forecast_times`.

    If the XGBoost model is loaded: return its predictions directly.
    Fallback (no model): average of T-24h, T-168h, T-336h historical values.
    """
    hist = cell_index.get(cell_name)

    # ── XGBoost path (preferred) ──────────────────────────────────────────────
    model  = _XGB_MODELS.get(kpi)
    scaler = _XGB_SCALERS.get(kpi)
    feats  = _XGB_FEATURES.get(kpi, [])

    if model is not None and scaler is not None and feats:
        rows = _build_xgb_rows(kpi, cell_name, forecast_times, hist)
        X_df = pd.DataFrame(rows)
        for fn in feats:
            if fn not in X_df.columns:
                X_df[fn] = 0
        X_df = X_df[feats].fillna(0)
        try:
            return model.predict(scaler.transform(X_df.values))
        except Exception:
            pass  # fall through to lag fallback

    # ── Lag-blend fallback (no model available) ───────────────────────────────
    if hist is None or kpi not in hist.columns:
        return np.full(len(forecast_times), np.nan)

    s = hist[kpi]
    lags = np.array([
        [_lag_lookup(s, t, 24), _lag_lookup(s, t, 168), _lag_lookup(s, t, 336)]
        for t in forecast_times
    ])
    return np.nanmean(lags, axis=1)


def run_forecast(
    cells: list,
    horizon_hours: int,
    thresholds: dict | None = None,
) -> dict:
    """
    Main entry point.  Returns a dict that is JSON-serialisable and stored
    in the dcc.Store for the charts/table callbacks.

    Keys:
      forecast_df   : list of records  (Time, Cell Name, kpi columns, Risk)
      summary       : {n_cells, p0_cells, p1_cells, max_prb, mean_prb}
      method        : "model+lag" | "lag-only"
    """
    if not cells or DF.empty:
        return {}

    from congestion_engine import DEFAULT_THRESHOLDS as _DT
    t = {**_DT, **(thresholds or {})}

    cell_index = _make_cell_index(DF)
    last_time  = DF["Time"].max()
    forecast_times = [last_time + pd.Timedelta(hours=h + 1) for h in range(horizon_hours)]

    kpis_to_forecast = ["DL PRB Load", "Active Users", "LTE DL User Throughput"]

    records = []
    for cn in cells:
        kpi_preds = {}
        for kpi in kpis_to_forecast:
            if kpi in DF.columns:
                kpi_preds[kpi] = _forecast_one_cell_one_kpi(
                    kpi, cn, forecast_times, cell_index)
            else:
                kpi_preds[kpi] = np.full(len(forecast_times), np.nan)

        for i, ft in enumerate(forecast_times):
            prb  = float(kpi_preds["DL PRB Load"][i])              if not np.isnan(kpi_preds["DL PRB Load"][i])              else None
            usr  = float(kpi_preds["Active Users"][i])             if not np.isnan(kpi_preds["Active Users"][i])             else None
            tput = float(kpi_preds["LTE DL User Throughput"][i])   if not np.isnan(kpi_preds["LTE DL User Throughput"][i])   else None

            # ── 3-KPI AND logic — identical to congestion_engine.py ─────────
            if (prb  is not None and prb  > t["p0_prb"]
                    and tput is not None and tput < t["p0_thp"]
                    and usr  is not None and usr  > t["p0_users"]):
                risk = "P0"
            elif (prb  is not None and prb  > t["p1_prb"]
                    and tput is not None and tput < t["p1_thp"]
                    and usr  is not None and usr  > t["p1_users"]):
                risk = "P1"
            else:
                risk = "Not Congested"

            records.append({
                "Time":                   ft.isoformat(),
                "Cell Name":              cn,
                "DL PRB Load":            prb,
                "Active Users":           usr,
                "LTE DL User Throughput": tput,
                "Hour":                   ft.hour,
                "Risk Level":             risk,
            })

    if not records:
        return {}

    fc_df    = pd.DataFrame(records)
    p0_cells = fc_df.loc[fc_df["Risk Level"] == "P0", "Cell Name"].nunique()
    p1_cells = fc_df.loc[fc_df["Risk Level"] == "P1", "Cell Name"].nunique()
    max_prb  = fc_df["DL PRB Load"].max()
    mean_prb = fc_df["DL PRB Load"].mean()

    return {
        "records": records,
        "summary": {
            "n_cells":  len(cells),
            "p0_cells": int(p0_cells),
            "p1_cells": int(p1_cells),
            "max_prb":  round(float(max_prb),  2) if pd.notna(max_prb)  else None,
            "mean_prb": round(float(mean_prb), 2) if pd.notna(mean_prb) else None,
        },
        "method":       "model+lag" if _XGB_MODELS else "lag-only",
        "horizon":      horizon_hours,
        "thresholds":   t,
        # keep these for backwards-compat with existing chart callbacks
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
        margin={"t": 10, "b": 10, "l": 10, "r": 10},
        height=300,
    )
    return fig


def _badge(label, color, bg=None):
    bg = bg or (color + "18")
    return html.Span(label, style={
        "fontSize": "11px", "fontWeight": 700, "color": color,
        "background": bg, "border": f"1px solid {color}",
        "borderRadius": "6px", "padding": "2px 9px",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Model status bar (runs on page load) ──────────────────────────────────
@callback(
    Output("fc-model-status-bar", "children"),
    Input("fc-model-status-bar", "id"),          # fires once on mount
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
        if loaded:
            badges.append(_badge(f"✓ {label} (XGBoost)", _SUCCESS))
        else:
            badges.append(_badge(f"◌ {label} (lag fallback)", "#94a3b8"))
    return badges


# ── 2. Auto-select top-10 most congested cells ───────────────────────────────
@callback(
    Output("fc-cell-selector", "value"),
    Input("fc-auto-select-btn", "n_clicks"),
    Input("fc-clear-btn",       "n_clicks"),
    State("fc-cell-selector",   "value"),
    prevent_initial_call=True,
)
def auto_select_cells(auto_n, clear_n, current):
    triggered = ctx.triggered_id
    if triggered == "fc-clear-btn":
        return []
    # Top 10 by P0 congestion count
    if DF.empty or "Cell Name" not in DF.columns:
        raise PreventUpdate
    if "P0_Congestion" in DF.columns:
        top = (DF[DF["P0_Congestion"] == True]
               .groupby("Cell Name").size()
               .sort_values(ascending=False)
               .head(10).index.tolist())
    elif "DL PRB Load" in DF.columns:
        top = (DF.groupby("Cell Name")["DL PRB Load"]
               .mean()
               .sort_values(ascending=False)
               .head(10).index.tolist())
    else:
        raise PreventUpdate
    return top


# ── 3. Run forecast → store ──────────────────────────────────────────────────
@callback(
    Output("fc-forecast-store", "data"),
    Input("fc-run-btn",        "n_clicks"),
    State("fc-cell-selector",  "value"),
    State("fc-horizon",        "value"),
    State("fc-p0-prb",         "value"),
    State("fc-p0-thp",         "value"),
    State("fc-p0-users",       "value"),
    State("fc-p1-prb",         "value"),
    State("fc-p1-thp",         "value"),
    State("fc-p1-users",       "value"),
    prevent_initial_call=True,
)
def run_forecast_callback(n_clicks, cells, horizon,
                          p0_prb, p0_thp, p0_users,
                          p1_prb, p1_thp, p1_users):
    if not cells:
        return {}
    def _f(v, d): return float(v) if v is not None else d
    thresholds = {
        "p0_prb":   _f(p0_prb,   90),   "p0_thp":   _f(p0_thp,   2000),
        "p0_users": _f(p0_users, 8),
        "p1_prb":   _f(p1_prb,   80),   "p1_thp":   _f(p1_thp,   4000),
        "p1_users": _f(p1_users, 4),
    }
    return run_forecast(cells, int(horizon or 24), thresholds)


# ── 4. Summary cards ──────────────────────────────────────────────────────────
@callback(
    Output("fc-summary-cards", "children"),
    Input("fc-forecast-store", "data"),
)
def update_summary_cards(store):
    if not store or "summary" not in store:
        return []

    s      = store["summary"]
    method = store.get("method", "lag-only")

    metrics = [
        ("Cells Forecasted",  str(s.get("n_cells",  0)),  _ACCENT),
        ("P0 Risk Cells",     str(s.get("p0_cells", 0)),  _DANGER),
        ("P1 Risk Cells",     str(s.get("p1_cells", 0)),  _WARN),
        ("Max PRB Load (%)",  str(s.get("max_prb",  "—")), "#0369a1"),
        ("Avg PRB Load (%)",  str(s.get("mean_prb", "—")), "#0369a1"),
    ]

    cards = [
        html.Div([
            html.Div(label, style={"fontSize": "12px", "color": "#64748b",
                                   "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "26px", "fontWeight": 700,
                                   "color": color}),
        ], style=CARD_STYLE)
        for label, value, color in metrics
    ]

    method_badge = html.Div([
        html.Div("Forecast method", style={"fontSize": "12px", "color": "#64748b",
                                            "marginBottom": "4px"}),
        html.Div(
            "Neural Network + Lag Blend" if method == "model+lag" else "Lag Blend (no model)",
            style={"fontSize": "13px", "fontWeight": 700,
                   "color": _SUCCESS if method == "model+lag" else "#64748b"},
        ),
    ], style=CARD_STYLE)

    return cards + [method_badge]


# ── 5. Alert banner ────────────────────────────────────────────────────────────
@callback(
    Output("fc-alert-banner", "children"),
    Input("fc-forecast-store", "data"),
)
def update_alert_banner(store):
    if not store or "summary" not in store:
        return []
    s = store["summary"]
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


# ── 6. Main forecast chart ────────────────────────────────────────────────────
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

    kpi_label = kpi
    prb_thr   = store.get("prb_threshold", DEFAULT_PRB_THRESHOLD)

    # Pull historical tail (last 48 h) for context
    hist_hours = 48
    last_time  = DF["Time"].max()
    hist_start = last_time - pd.Timedelta(hours=hist_hours)
    cells      = fc_df["Cell Name"].unique().tolist()

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, cn in enumerate(cells):
        clr = colors[idx % len(colors)]

        # Historical line
        if not DF.empty and kpi in DF.columns:
            h = DF[(DF["Cell Name"] == cn) & (DF["Time"] >= hist_start)] \
                    .sort_values("Time")
            if not h.empty:
                fig.add_trace(go.Scatter(
                    x=h["Time"], y=h[kpi],
                    mode="lines", name=f"{cn} (historical)",
                    line={"color": clr, "width": 1.5, "dash": "dot"},
                    opacity=0.55, legendgroup=cn,
                ))

        # Forecast line
        cell_fc = fc_df[fc_df["Cell Name"] == cn].sort_values("Time")
        y_vals  = cell_fc[kpi].values.astype(float)

        fig.add_trace(go.Scatter(
            x=cell_fc["Time"], y=y_vals,
            mode="lines+markers",
            name=f"{cn} (forecast)",
            line={"color": clr, "width": 2.5},
            marker={"size": 4},
            legendgroup=cn,
        ))

        # Colour congested segments
        p0_mask = cell_fc["Risk Level"] == "P0"
        if p0_mask.any():
            fig.add_trace(go.Scatter(
                x=cell_fc.loc[p0_mask, "Time"],
                y=cell_fc.loc[p0_mask, kpi],
                mode="markers", name=f"{cn} P0",
                marker={"color": P0_COLOR, "size": 8, "symbol": "circle"},
                legendgroup=cn, showlegend=False,
            ))

    # Threshold line for PRB
    if kpi == "DL PRB Load":
        all_times = fc_df["Time"].sort_values()
        fig.add_hline(
            y=prb_thr,
            line={"color": P0_COLOR, "dash": "dash", "width": 1.5},
            annotation_text=f"P0 threshold ({prb_thr}%)",
            annotation_position="top right",
        )
        fig.add_hline(
            y=prb_thr * 0.75,
            line={"color": P1_COLOR, "dash": "dash", "width": 1},
            annotation_text=f"P1 threshold ({prb_thr * 0.75:.0f}%)",
            annotation_position="bottom right",
        )

    # Divider between historical and forecast
    if not DF.empty and "Time" in DF.columns:
        fig.add_vline(
            x=DF["Time"].max().timestamp() * 1000,
            line={"color": "#94a3b8", "dash": "dot", "width": 1.5},
            annotation_text="Forecast start",
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"Forecast — {kpi_label}",
        template="plotly_white",
        height=380,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "left", "x": 0},
        xaxis_title="Time",
        yaxis_title=kpi_label,
        margin={"t": 50, "b": 40, "l": 50, "r": 20},
    )
    return fig


# ── 7. Congestion heatmap ─────────────────────────────────────────────────────
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

    cells = sorted(fc_df["Cell Name"].unique())
    hours = list(range(24))

    _risk_num = {"Not Congested": 0, "P1": 1, "P0": 2}

    z = np.zeros((len(cells), 24), dtype=float)
    for i, cn in enumerate(cells):
        sub = fc_df[fc_df["Cell Name"] == cn]
        for _, row in sub.iterrows():
            h = int(row["Hour"]) % 24
            z[i, h] = max(z[i, h], _risk_num.get(row["Risk Level"], 0))

    colorscale = [
        [0.0,  "#dcfce7"],   # green — Not Congested
        [0.49, "#dcfce7"],
        [0.5,  "#fed7aa"],   # orange — P1
        [0.74, "#fed7aa"],
        [0.75, "#fee2e2"],   # red — P0
        [1.0,  "#fee2e2"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{h:02d}:00" for h in hours], y=cells,
        colorscale=colorscale, zmin=0, zmax=2,
        colorbar={
            "tickvals": [0, 1, 2],
            "ticktext": ["Clear", "P1", "P0"],
            "thickness": 14, "len": 0.7,
        },
        hovertemplate="Cell: %{y}<br>Hour: %{x}<br>Risk: %{z}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        height=max(200, 28 * len(cells) + 80),
        margin={"t": 10, "b": 50, "l": 10, "r": 60},
        xaxis_title="Hour of Day",
        yaxis_title="",
        yaxis={"tickfont": {"size": 10}},
    )
    return fig


# ── 8. PRB box plot ────────────────────────────────────────────────────────────
@callback(
    Output("fc-prb-box", "figure"),
    Input("fc-forecast-store", "data"),
)
def update_prb_box(store):
    if not store or "records" not in store:
        return _empty_fig()

    fc_df = _records_to_df(store)
    if fc_df.empty or "DL PRB Load" not in fc_df.columns:
        return _empty_fig()

    prb_thr = store.get("prb_threshold", DEFAULT_PRB_THRESHOLD)

    fig = go.Figure()
    cells = sorted(fc_df["Cell Name"].unique())
    for cn in cells:
        vals = fc_df.loc[fc_df["Cell Name"] == cn, "DL PRB Load"].dropna()
        max_v = vals.max() if not vals.empty else 0
        clr   = P0_COLOR if max_v >= prb_thr else (P1_COLOR if max_v >= prb_thr * 0.75 else _SUCCESS)
        fig.add_trace(go.Box(
            y=vals, name=cn,
            marker_color=clr,
            boxpoints="outliers",
            line={"width": 1.5},
        ))

    fig.add_hline(y=prb_thr, line={"color": P0_COLOR, "dash": "dash", "width": 1.5},
                  annotation_text=f"{prb_thr}% P0")
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=300,
        yaxis_title="DL PRB Load (%)",
        xaxis_title="",
        margin={"t": 10, "b": 40, "l": 50, "r": 10},
    )
    return fig


# ── 9. Time-to-breach bar chart ───────────────────────────────────────────────
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

    rows = []
    for cn, grp in fc_df.groupby("Cell Name"):
        grp = grp.sort_values("Time")
        breach = grp.loc[grp["DL PRB Load"] >= prb_thr]
        if not breach.empty:
            hours_until = int((breach["Time"].min() - fc_df["Time"].min()).total_seconds() / 3600) + 1
            rows.append({"Cell Name": cn, "Hours Until Breach": hours_until})

    if not rows:
        return _empty_fig("No cells forecast to breach the P0 threshold 🎉")

    ttb = pd.DataFrame(rows).sort_values("Hours Until Breach")
    fig = px.bar(
        ttb, x="Hours Until Breach", y="Cell Name",
        orientation="h",
        color="Hours Until Breach",
        color_continuous_scale=["#ef4444", "#f97316", "#fde68a"],
        labels={"Hours Until Breach": "Hours"},
    )
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        height=max(220, 32 * len(rows) + 80),
        margin={"t": 10, "b": 40, "l": 10, "r": 10},
        coloraxis_showscale=False,
        yaxis_title="",
    )
    return fig


# ── 10. Busy-hour bar chart ───────────────────────────────────────────────────
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

    avg_by_hour = (
        fc_df.groupby("Hour")["DL PRB Load"]
             .mean()
             .reindex(range(24), fill_value=0)
             .reset_index()
    )
    avg_by_hour.columns = ["Hour", "Avg PRB Load"]
    avg_by_hour["Label"] = avg_by_hour["Hour"].apply(lambda h: f"{h:02d}:00")

    prb_thr = store.get("prb_threshold", DEFAULT_PRB_THRESHOLD)
    avg_by_hour["Color"] = avg_by_hour["Avg PRB Load"].apply(
        lambda v: P0_COLOR if v >= prb_thr else (P1_COLOR if v >= prb_thr * 0.75 else _SUCCESS)
    )

    fig = go.Figure(go.Bar(
        x=avg_by_hour["Label"],
        y=avg_by_hour["Avg PRB Load"],
        marker_color=avg_by_hour["Color"],
        hovertemplate="%{x}<br>Avg PRB: %{y:.1f}%<extra></extra>",
    ))
    fig.add_hline(y=prb_thr, line={"color": P0_COLOR, "dash": "dash", "width": 1.5})
    fig.update_layout(
        template="plotly_white",
        height=300,
        xaxis_title="Hour of Day",
        yaxis_title="Avg DL PRB Load (%)",
        margin={"t": 10, "b": 40, "l": 50, "r": 10},
    )
    return fig


# ── 11. Alert table ───────────────────────────────────────────────────────────
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

    rows = []
    for cn, grp in fc_df.groupby("Cell Name"):
        p0_h   = int((grp["Risk Level"] == "P0").sum())
        p1_h   = int((grp["Risk Level"] == "P1").sum())
        max_prb = grp["DL PRB Load"].max()
        max_usr = grp["Active Users"].max()
        risk    = "P0" if p0_h else ("P1" if p1_h else "Not Congested")
        if risk == "Not Congested":
            continue
        breach_rows = grp[grp["Risk Level"].isin(["P0", "P1"])].sort_values("Time")
        first_breach = breach_rows["Time"].min().strftime("%Y-%m-%d %H:%M") \
                       if not breach_rows.empty else "—"
        rows.append({
            "Cell Name":         cn,
            "Risk Level":        risk,
            "Max PRB Load (%)":  round(float(max_prb), 1) if pd.notna(max_prb) else "—",
            "Max Active Users":  round(float(max_usr), 1) if pd.notna(max_usr) else "—",
            "P0 Hours":          p0_h,
            "P1 Hours":          p1_h,
            "First Breach":      first_breach,
        })

    if not rows:
        return [], empty_cols

    rows.sort(key=lambda r: (0 if r["Risk Level"] == "P0" else 1, r["First Breach"]))
    cols = [{"name": c, "id": c} for c in rows[0].keys()]
    return rows, cols


# ── 12. CSV export ─────────────────────────────────────────────────────────────
@callback(
    Output("fc-download",      "data"),
    Input("fc-export-btn",     "n_clicks"),
    State("fc-forecast-store", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, store):
    if not store or "records" not in store:
        raise PreventUpdate
    from dash import dcc as _dcc
    fc_df = _records_to_df(store)
    return _dcc.send_data_frame(fc_df.to_csv, "forecast.csv", index=False)


# ── 12b. Spinner target — fires while run_forecast_callback runs ──────────────
@callback(
    Output("fc-run-spinner-target", "children"),
    Input("fc-forecast-store", "data"),
)
def _spinner_sync(store):
    """Keeps fc-run-loading spinner active until the store is updated."""
    return None


# ── 13. Refresh cache button ──────────────────────────────────────────────────
@callback(
    Output("fc-cache-status",       "children"),
    Input("fc-refresh-cache-btn",   "n_clicks"),
    State("fc-prb-threshold",       "value"),
    State("fc-users-threshold",     "value"),
    prevent_initial_call=True,
)
def refresh_cache(n_clicks, prb_thr, usr_thr):
    from config import DEFAULT_PRB_THRESHOLD, DEFAULT_ACTIVE_USER_THRESHOLD
    prb_thr = float(prb_thr or DEFAULT_PRB_THRESHOLD)
    usr_thr = float(usr_thr or DEFAULT_ACTIVE_USER_THRESHOLD)

    if DF.empty:
        return "No data available."

    import os
    _CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "forecast_cache.csv")

    # Build full 168-h forecast for every cell
    all_cells      = DF["Cell Name"].dropna().unique().tolist()
    last_time      = DF["Time"].max()
    forecast_times = [last_time + pd.Timedelta(hours=h + 1) for h in range(168)]

    # Reuse the same per-cell loop used by run_forecast
    kpis = ["DL PRB Load", "Active Users", "LTE DL User Throughput"]
    cell_index = {}
    for cn, grp in DF.groupby("Cell Name"):
        g = grp.sort_values("Time").drop_duplicates("Time").set_index("Time")
        cell_index[str(cn)] = g[[k for k in kpis if k in g.columns]]

    records = []
    for cn in all_cells:
        hist = cell_index.get(cn)
        for kpi in kpis:
            if kpi not in DF.columns:
                continue

            # Use _forecast_one_cell_one_kpi for consistency: XGBoost if available, lag fallback otherwise
            preds = _forecast_one_cell_one_kpi(kpi, cn, forecast_times, cell_index)

            for i, t in enumerate(forecast_times):
                records.append({
                    "Time": t.isoformat(),
                    "Cell Name": cn,
                    "KPI": kpi,
                    "Value": float(preds[i]) if not np.isnan(preds[i]) else None,
                    "Hour": t.hour,
                })

    if not records:
        return "Forecast returned no data."

    fc_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    fc_df.to_csv(_CACHE_PATH, index=False)
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    return f"✓ Cached {len(all_cells)} cells · {ts}"


# ─────────────────────────────────────────────────────────────────────────────
# Retrain callbacks
# ─────────────────────────────────────────────────────────────────────────────

def _retrain_status_ui(message: str, running: bool, error: bool):
    """Build the status div shown in the Settings tab."""
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
    """Validate input, then kick off a background training thread."""
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
    return _retrain_status_ui("Starting …", True, False), False   # enable polling


@callback(
    Output("retrain-status", "children", allow_duplicate=True),
    Output("retrain-poll",   "disabled", allow_duplicate=True),
    Input("retrain-poll",    "n_intervals"),
    prevent_initial_call=True,
)
def poll_retrain_status(n_intervals):
    """Read _TRAIN_STATUS every 2s; stop polling when training is done."""
    with _TRAIN_LOCK:
        running = _TRAIN_STATUS["running"]
        message = _TRAIN_STATUS["message"]
        error   = _TRAIN_STATUS["error"]

    ui = _retrain_status_ui(message, running, error)
    stop_poll = not running   # disable interval once finished
    return ui, stop_poll


# ─────────────────────────────────────────────────────────────────────────────
# Weekly Risk Calendar
# ─────────────────────────────────────────────────────────────────────────────
_DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

@callback(
    Output("fc-weekly-calendar", "figure"),
    Input("fc-forecast-store", "data"),
)
def update_weekly_calendar(store: dict):
    if not store or "records" not in store:
        fig = go.Figure()
        fig.update_layout(
            title="Run a ≥ 168 h forecast to see the weekly calendar",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, sans-serif"},
        )
        return fig

    df = pd.DataFrame(store["records"])
    if "Time" not in df.columns or "DL PRB Load" not in df.columns:
        return go.Figure()

    df["Time"]     = pd.to_datetime(df["Time"])
    df["Hour"]     = df["Time"].dt.hour
    df["DayOfWk"]  = df["Time"].dt.dayofweek   # 0=Mon … 6=Sun
    df["Day"]      = df["DayOfWk"].map(dict(enumerate(_DAY_NAMES)))

    pivot = (
        df.groupby(["Hour", "Day"])["DL PRB Load"]
        .mean()
        .unstack("Day")
        .reindex(columns=_DAY_NAMES)
        .fillna(0)
    )

    prb_threshold = store.get("prb_threshold", 90)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=_DAY_NAMES,
        y=[f"{h:02d}h" for h in pivot.index],
        colorscale=[
            [0.0,  "#22c55e"],
            [0.5,  "#fbbf24"],
            [0.75, "#f97316"],
            [1.0,  "#ef4444"],
        ],
        zmin=0,
        zmax=100,
        colorbar={"title": "Avg PRB %", "thickness": 14,
                  "tickfont": {"size": 10}},
        hovertemplate="<b>%{x} %{y}</b><br>Avg PRB: %{z:.1f}%<extra></extra>",
    ))

    # Threshold contour line annotation
    fig.update_layout(
        title={"text": "Weekly Risk Calendar — Avg Predicted DL PRB Load",
               "font": {"size": 14, "color": "#0f172a", "family": "Inter, sans-serif"},
               "x": 0.02},
        xaxis={"title": "Day of Week"},
        yaxis={"title": "Hour of Day", "autorange": "reversed"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "size": 11},
        margin={"t": 48, "b": 40, "l": 60, "r": 20},
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# What-if Traffic Growth Simulator
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("fc-whatif-output", "children"),
    Output("fc-whatif-chart",  "figure"),
    Input("fc-growth-slider",  "value"),
    Input("fc-forecast-store", "data"),
)
def update_whatif(multiplier: float, store: dict):
    empty_fig = go.Figure()
    empty_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={"family": "Inter, sans-serif"})

    if not store or "records" not in store:
        return "Run a forecast first.", empty_fig

    df = pd.DataFrame(store["records"])
    if "DL PRB Load" not in df.columns:
        return "No PRB data in forecast.", empty_fig

    mult          = float(multiplier or 1.0)
    prb_threshold = float(store.get("prb_threshold", 90))
    df["ScaledPRB"] = (df["DL PRB Load"].fillna(0) * mult).clip(0, 100)

    # Per-cell summary
    cell_summary = (
        df.groupby("Cell Name")
        .agg(
            BaseMax=("DL PRB Load",  "max"),
            ScaledMax=("ScaledPRB",  "max"),
            ScaledMean=("ScaledPRB", "mean"),
        )
        .reset_index()
    )
    cell_summary["Risk"] = cell_summary["ScaledMax"].apply(
        lambda v: "P0 ⚠" if v >= prb_threshold
                  else ("P1" if v >= prb_threshold * 0.75 else "✓ Clear")
    )

    n_p0  = (cell_summary["Risk"] == "P0 ⚠").sum()
    n_p1  = (cell_summary["Risk"] == "P1").sum()
    n_ok  = (cell_summary["Risk"] == "✓ Clear").sum()
    total = len(cell_summary)

    summary_children = [
        html.Div(f"×{mult:.1f} traffic growth", style={"fontWeight": 700,
                                                         "marginBottom": "6px",
                                                         "color": "#0f172a"}),
        html.Div([
            html.Span(f"🔴 P0: {n_p0}/{total}",
                      style={"color": "#ef4444", "fontWeight": 600,
                             "marginRight": "14px"}),
            html.Span(f"🟠 P1: {n_p1}/{total}",
                      style={"color": "#f97316", "fontWeight": 600,
                             "marginRight": "14px"}),
            html.Span(f"✅ Clear: {n_ok}/{total}",
                      style={"color": "#22c55e", "fontWeight": 600}),
        ]),
    ]

    # Bar chart — scaled max PRB per cell with threshold line
    cell_summary = cell_summary.sort_values("ScaledMax", ascending=False).head(20)
    color_map    = {"P0 ⚠": "#ef4444", "P1": "#f97316", "✓ Clear": "#22c55e"}
    colors       = cell_summary["Risk"].map(color_map).fillna("#94a3b8").tolist()

    fig = go.Figure()
    fig.add_bar(
        y=cell_summary["Cell Name"],
        x=cell_summary["ScaledMax"],
        orientation="h",
        marker_color=colors,
        name=f"PRB @ ×{mult:.1f}",
        hovertemplate="<b>%{y}</b><br>Scaled max PRB: %{x:.1f}%<extra></extra>",
    )
    if mult != 1.0:
        fig.add_bar(
            y=cell_summary["Cell Name"],
            x=cell_summary["BaseMax"],
            orientation="h",
            marker_color="rgba(99,102,241,0.3)",
            name="Base (×1)",
            hovertemplate="<b>%{y}</b><br>Base max PRB: %{x:.1f}%<extra></extra>",
        )
    fig.add_vline(x=prb_threshold, line_dash="dot", line_color="#ef4444",
                  line_width=1.5,
                  annotation_text=f"P0 threshold {prb_threshold}%",
                  annotation_font_color="#ef4444")
    fig.update_layout(
        title={"text": f"Top 20 Cells — Predicted Max PRB at ×{mult:.1f} Traffic",
               "font": {"size": 13, "color": "#0f172a", "family": "Inter, sans-serif"},
               "x": 0.01},
        barmode="overlay",
        xaxis={"title": "DL PRB Load (%)", "range": [0, 105]},
        yaxis={"categoryorder": "total ascending", "tickfont": {"size": 10}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "size": 11},
        legend={"orientation": "h", "y": -0.15},
        margin={"t": 44, "b": 40, "l": 10, "r": 10},
    )
    return summary_children, fig
