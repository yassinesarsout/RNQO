"""
Forecast tab callbacks — AI-powered forecasting for RAN KPIs.

Forecasting engine:
  1. Tries to load Keras models + sklearn scalers from  ../my training code/
  2. If a model + scaler pair is found  → uses the neural network prediction.
  3. For any missing model             → falls back to lag-blending
     (average of T-24h, T-168h, T-336h from historical data).
  4. Final value = (model_pred + lag_24h + lag_1w + lag_2w) / N_available
     — same blending strategy used during training evaluation.

Congestion detection:
  P0 → forecasted DL PRB Load  ≥  prb_threshold  (default 90 %)
  P1 → forecasted Active Users ≥  users_threshold  OR
        forecasted DL PRB Load  ≥  prb_threshold × 0.75
"""

import os
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import (
    Input, Output, State, callback,
    no_update, html, ctx,
)
from dash.exceptions import PreventUpdate

from config import (
    SECTION_STYLE, CARD_STYLE, P0_COLOR, P1_COLOR,
    DEFAULT_PRB_THRESHOLD, DEFAULT_ACTIVE_USER_THRESHOLD,
)
from precompute import DF

# ── paths ───────────────────────────────────────────────────────────────────
_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "my training code")

_MODELS: dict   = {}   # kpi → keras model  (loaded once)
_SCALERS: dict  = {}   # kpi → sklearn scaler

_KPI_FILES = {
    "DL PRB Load":           ("prb_model",          "prb_scaler"),
    "Active Users":          ("active_users_model",  "active_users_scaler"),
    "LTE DL User Throughput":("throughput_model",    "throughput_scaler"),
}

_FEATURES = {
    "DL PRB Load": [
        "cell_id", "Active Users", "LTE DL User Throughput",
        "prb_lag", "throughput_lag", "users_lag",
    ],
    "Active Users": [
        "cell_id", "LTE DL User Throughput", "DL PRB Load",
        "users_lag", "throughput_lag", "prb_lag",
    ],
    "LTE DL User Throughput": [
        "cell_id", "DL PRB Load", "Active Users",
        "throughput_lag", "prb_lag", "users_lag",
    ],
}

_LAG_H = 24 * 14   # 336 h

_ACCENT  = "#6366f1"
_SUCCESS = "#22c55e"
_WARN    = "#f97316"
_DANGER  = "#ef4444"

# ── cell_id lookup (mirrors training encoding) ───────────────────────────────
_CELL_IDS: dict = {}
if not DF.empty and "Cell Name" in DF.columns:
    _cats = pd.Categorical(DF["Cell Name"])
    _CELL_IDS = dict(zip(_cats.categories, range(len(_cats.categories))))

# ── FDD/TDD dummy column names ────────────────────────────────────────────────
_FDD_COLS: list = []
if not DF.empty and "Cell FDD TDD Indication" in DF.columns:
    _FDD_COLS = [
        f"Cell FDD TDD Indication_{v}"
        for v in sorted(DF["Cell FDD TDD Indication"].dropna().unique())
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (attempted once at import)
# ─────────────────────────────────────────────────────────────────────────────

def _try_load():
    for kpi, (mname, sname) in _KPI_FILES.items():
        # Try .keras then .h5
        for ext in [".keras", ".h5"]:
            mp = os.path.join(_MODEL_DIR, mname + ext)
            if os.path.exists(mp):
                try:
                    from keras.models import load_model as _lm
                    _MODELS[kpi] = _lm(mp, compile=False)
                except Exception:
                    pass
                break

        # Scaler (.pkl)
        sp = os.path.join(_MODEL_DIR, sname + ".pkl")
        if os.path.exists(sp):
            try:
                import joblib
                _SCALERS[kpi] = joblib.load(sp)
            except Exception:
                pass

_try_load()


# ─────────────────────────────────────────────────────────────────────────────
# Core forecast engine
# ─────────────────────────────────────────────────────────────────────────────

def _make_cell_index(df: pd.DataFrame) -> dict:
    """Return {cell_name: series indexed by Time} for all 3 KPIs."""
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


def _forecast_one_cell_one_kpi(
    kpi: str,
    cell_name: str,
    forecast_times: list,
    cell_index: dict,
) -> np.ndarray:
    """
    Predict `kpi` for `cell_name` at each timestamp in `forecast_times`.

    Strategy:
      1. Build lag features for 24h / 168h / 336h.
      2. If model + scaler are available, run the neural network and blend.
      3. Otherwise, average available lags (lag-blend fallback).
    """
    hist = cell_index.get(cell_name)

    lags_24  = np.full(len(forecast_times), np.nan)
    lags_168 = np.full(len(forecast_times), np.nan)
    lags_336 = np.full(len(forecast_times), np.nan)

    if hist is not None and kpi in hist.columns:
        s = hist[kpi]
        for i, t in enumerate(forecast_times):
            lags_24[i]  = _lag_lookup(s, t, 24)
            lags_168[i] = _lag_lookup(s, t, 168)
            lags_336[i] = _lag_lookup(s, t, 336)

    # ── Neural net path ──────────────────────────────────────────────────────
    model  = _MODELS.get(kpi)
    scaler = _SCALERS.get(kpi)

    model_preds = None
    if model is not None and scaler is not None and hist is not None:
        feat_names = _FEATURES[kpi] + _FDD_COLS
        rows = []
        for i, t in enumerate(forecast_times):
            # Use same-lag values as proxies for "current" correlated KPIs
            row = {
                "cell_id":        _CELL_IDS.get(cell_name, 0),
                "prb_lag":        _lag_lookup(hist["DL PRB Load"],            t, _LAG_H) if "DL PRB Load"            in hist.columns else np.nan,
                "users_lag":      _lag_lookup(hist["Active Users"],            t, _LAG_H) if "Active Users"            in hist.columns else np.nan,
                "throughput_lag": _lag_lookup(hist["LTE DL User Throughput"],  t, _LAG_H) if "LTE DL User Throughput"  in hist.columns else np.nan,
            }
            # Correlated KPI proxies — use same-hour 1-week ago value
            for other_kpi in ["DL PRB Load", "Active Users", "LTE DL User Throughput"]:
                if other_kpi != kpi and hist is not None and other_kpi in hist.columns:
                    row[other_kpi] = _lag_lookup(hist[other_kpi], t, 168)

            for fc in _FDD_COLS:
                col_src = fc.replace("Cell FDD TDD Indication_", "")
                row[fc] = 1 if _cell_fdd(cell_name) == col_src else 0

            rows.append(row)

        X_df = pd.DataFrame(rows)
        for fn in feat_names:
            if fn not in X_df.columns:
                X_df[fn] = 0
        X_df = X_df[feat_names].fillna(0)

        try:
            X_scaled    = scaler.transform(X_df.values)
            model_preds = model.predict(X_scaled, verbose=0).flatten()
        except Exception:
            model_preds = None

    # ── Blend ────────────────────────────────────────────────────────────────
    sources = [s for s in [lags_24, lags_168, lags_336] if not np.all(np.isnan(s))]
    if model_preds is not None:
        sources.insert(0, model_preds)

    if not sources:
        return np.full(len(forecast_times), np.nan)

    stack = np.vstack(sources)                          # (N_sources, T)
    return np.nanmean(stack, axis=0)                    # (T,)


def _cell_fdd(cell_name: str) -> str:
    """Return the FDD/TDD indication for a cell (from DF)."""
    if DF.empty or "Cell FDD TDD Indication" not in DF.columns:
        return ""
    rows = DF.loc[DF["Cell Name"] == cell_name, "Cell FDD TDD Indication"]
    return str(rows.iloc[0]) if not rows.empty else ""


def run_forecast(
    cells: list,
    horizon_hours: int,
    prb_threshold: float,
    users_threshold: float,
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

    cell_index = _make_cell_index(DF)
    last_time  = DF["Time"].max()
    forecast_times = [last_time + pd.Timedelta(hours=h + 1) for h in range(horizon_hours)]

    kpis_to_forecast = [
        "DL PRB Load",
        "Active Users",
        "LTE DL User Throughput",
    ]

    records = []
    for cn in cells:
        kpi_preds = {}
        for kpi in kpis_to_forecast:
            if kpi in (DF.columns.tolist()):
                kpi_preds[kpi] = _forecast_one_cell_one_kpi(
                    kpi, cn, forecast_times, cell_index
                )
            else:
                kpi_preds[kpi] = np.full(len(forecast_times), np.nan)

        for i, t in enumerate(forecast_times):
            prb  = float(kpi_preds["DL PRB Load"][i])   if not np.isnan(kpi_preds["DL PRB Load"][i])   else None
            usr  = float(kpi_preds["Active Users"][i])  if not np.isnan(kpi_preds["Active Users"][i])  else None
            tput = float(kpi_preds["LTE DL User Throughput"][i]) if not np.isnan(kpi_preds["LTE DL User Throughput"][i]) else None

            # Congestion
            if prb is not None and prb >= prb_threshold:
                risk = "P0"
            elif (prb is not None and prb >= prb_threshold * 0.75) or \
                 (usr is not None and usr >= users_threshold):
                risk = "P1"
            else:
                risk = "Not Congested"

            records.append({
                "Time":                    t.isoformat(),
                "Cell Name":               cn,
                "DL PRB Load":             prb,
                "Active Users":            usr,
                "LTE DL User Throughput":  tput,
                "Hour":                    t.hour,
                "Risk Level":              risk,
            })

    if not records:
        return {}

    fc_df = pd.DataFrame(records)

    p0_cells  = fc_df.loc[fc_df["Risk Level"] == "P0",  "Cell Name"].nunique()
    p1_cells  = fc_df.loc[fc_df["Risk Level"] == "P1",  "Cell Name"].nunique()
    max_prb   = fc_df["DL PRB Load"].max()
    mean_prb  = fc_df["DL PRB Load"].mean()

    method = "model+lag" if _MODELS else "lag-only"

    return {
        "records":  records,
        "summary": {
            "n_cells":  len(cells),
            "p0_cells": int(p0_cells),
            "p1_cells": int(p1_cells),
            "max_prb":  round(float(max_prb),  2) if pd.notna(max_prb)  else None,
            "mean_prb": round(float(mean_prb), 2) if pd.notna(mean_prb) else None,
        },
        "method": method,
        "horizon": horizon_hours,
        "prb_threshold":   prb_threshold,
        "users_threshold": users_threshold,
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
    badges = []
    for kpi, (mname, _) in _KPI_FILES.items():
        loaded = kpi in _MODELS
        label  = mname.replace("_", " ").title()
        if loaded:
            badges.append(_badge(f"✓ {label}", _SUCCESS))
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
    State("fc-prb-threshold",  "value"),
    State("fc-users-threshold","value"),
    prevent_initial_call=True,
)
def run_forecast_callback(n_clicks, cells, horizon, prb_thr, usr_thr):
    if not cells:
        return {}
    horizon   = int(horizon  or 24)
    prb_thr   = float(prb_thr  or DEFAULT_PRB_THRESHOLD)
    usr_thr   = float(usr_thr  or DEFAULT_ACTIVE_USER_THRESHOLD)
    return run_forecast(cells, horizon, prb_thr, usr_thr)


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

            lags_168 = np.full(len(forecast_times), np.nan)
            lags_336 = np.full(len(forecast_times), np.nan)
            if hist is not None and kpi in hist.columns:
                s = hist[kpi]
                for i, t in enumerate(forecast_times):
                    try:
                        lags_168[i] = s.at[t - pd.Timedelta(hours=168)]
                    except KeyError:
                        pass
                    try:
                        lags_336[i] = s.at[t - pd.Timedelta(hours=336)]
                    except KeyError:
                        pass

            model_preds = None
            model  = _MODELS.get(kpi)
            scaler = _SCALERS.get(kpi)
            if model and scaler and hist is not None:
                feat_names = _FEATURES[kpi] + _FDD_COLS
                rows = []
                for i, t in enumerate(forecast_times):
                    row = {
                        "cell_id":        _CELL_IDS.get(cn, 0),
                        "prb_lag":        hist["DL PRB Load"].get(t - pd.Timedelta(hours=_LAG_H), np.nan)        if "DL PRB Load"            in hist.columns else np.nan,
                        "users_lag":      hist["Active Users"].get(t - pd.Timedelta(hours=_LAG_H), np.nan)       if "Active Users"            in hist.columns else np.nan,
                        "throughput_lag": hist["LTE DL User Throughput"].get(t - pd.Timedelta(hours=_LAG_H), np.nan) if "LTE DL User Throughput" in hist.columns else np.nan,
                    }
                    for other in [k for k in kpis if k != kpi]:
                        if hist is not None and other in hist.columns:
                            row[other] = hist[other].get(t - pd.Timedelta(hours=168), np.nan)
                    for fc in _FDD_COLS:
                        row[fc] = 1 if _cell_fdd(cn) == fc.replace("Cell FDD TDD Indication_", "") else 0
                    rows.append(row)
                X_df = pd.DataFrame(rows)
                for fn in feat_names:
                    if fn not in X_df.columns:
                        X_df[fn] = 0
                X_df = X_df[feat_names].fillna(0)
                try:
                    model_preds = model.predict(scaler.transform(X_df.values), verbose=0).flatten()
                except Exception:
                    model_preds = None

            # Blend: mean(model, lag_1w, lag_2w)
            sources = []
            if model_preds is not None:
                sources.append(model_preds)
            if not np.all(np.isnan(lags_168)):
                sources.append(lags_168)
            if not np.all(np.isnan(lags_336)):
                sources.append(lags_336)
            preds = np.nanmean(np.vstack(sources), axis=0) if sources else np.full(len(forecast_times), np.nan)

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
