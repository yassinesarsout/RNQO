"""
XGBoost training script for RAN KPI forecasting.

Trains three models (one per KPI) at a 24-hour direct forecast horizon,
exactly replicating the feature engineering from the comparison notebooks.

Outputs (all written to models_dir, default = this directory — Models/):
  xgb_prb.json          XGBoost model for DL PRB Load
  xgb_users.json         XGBoost model for Active Users
  xgb_throughput.json    XGBoost model for LTE DL User Throughput
  scaler_prb.pkl         StandardScaler for PRB model
  scaler_users.pkl       StandardScaler for Users model
  scaler_throughput.pkl  StandardScaler for Throughput model
  meta.json              Categorical encodings + FDD column names

Usage (CLI):
  cd "c:/Users/yassi/PFE dashboard"
  python Models/train_xgboost.py

Usage (programmatic — called from dashboard):
  from Models.train_xgboost import train_all
  status = train_all(data_path="datasets/final_kpi_data.csv",
                     progress_cb=lambda msg: print(msg))
"""

import os, json, warnings
import numpy  as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

# ── Defaults ──────────────────────────────────────────────────────────────────
_HERE           = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA   = os.path.join(_HERE, "..", "datasets", "final_kpi_data.csv")
HORIZON         = 24   # hours ahead to predict

_TARGETS = {
    "DL PRB Load":            ("xgb_prb",         "scaler_prb",        "prb_rolling_24h"),
    "Active Users":           ("xgb_users",        "scaler_users",      "users_rolling_24h"),
    "LTE DL User Throughput": ("xgb_throughput",   "scaler_throughput", "throughput_rolling_24h"),
}

_REQUIRED_COLS = [
    "Time", "Cell Name", "Radio Site", "Region",
    "Active Users", "LTE DL User Throughput", "DL PRB Load",
    "Cell FDD TDD Indication",
]

_LAG_2W, _LAG_1W, _LAG_1D = 24 * 14, 24 * 7, 24


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    df = raw[_REQUIRED_COLS].copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["Cell Name", "Time"])

    cell_cat   = df["Cell Name"].astype("category")
    site_cat   = df["Radio Site"].astype("category")
    region_cat = df["Region"].astype("category")

    df["cell_id"]   = cell_cat.cat.codes
    df["site_id"]   = site_cat.cat.codes
    df["region_id"] = region_cat.cat.codes

    meta = {
        "cell_codes":     dict(zip(cell_cat.cat.categories,   range(len(cell_cat.cat.categories)))),
        "site_codes":     dict(zip(site_cat.cat.categories,   range(len(site_cat.cat.categories)))),
        "region_codes":   dict(zip(region_cat.cat.categories, range(len(region_cat.cat.categories)))),
        "cell_to_site":   df.drop_duplicates("Cell Name").set_index("Cell Name")["Radio Site"].to_dict(),
        "cell_to_region": df.drop_duplicates("Cell Name").set_index("Cell Name")["Region"].to_dict(),
    }

    df = pd.get_dummies(df, columns=["Cell FDD TDD Indication"])
    fdd_cols = sorted([c for c in df.columns if "Cell FDD TDD Indication_" in c])

    for kpi, prefix in [
        ("DL PRB Load",            "prb"),
        ("LTE DL User Throughput", "throughput"),
        ("Active Users",           "users"),
    ]:
        grp = df.groupby("Cell Name")[kpi]
        df[f"{prefix}_lag_2w"] = grp.shift(_LAG_2W)
        df[f"{prefix}_lag_1w"] = grp.shift(_LAG_1W)
        df[f"{prefix}_lag_1d"] = grp.shift(_LAG_1D)

    for kpi, col in [
        ("DL PRB Load",            "prb_rolling_24h"),
        ("Active Users",           "users_rolling_24h"),
        ("LTE DL User Throughput", "throughput_rolling_24h"),
    ]:
        df[col] = (
            df.groupby("Cell Name")[kpi]
            .transform(lambda x: x.shift(1).rolling(24, min_periods=1).mean())
        )

    df["hour_sin"]   = np.sin(2 * np.pi * df["Time"].dt.hour      / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["Time"].dt.hour      / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["Time"].dt.dayofweek / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["Time"].dt.dayofweek / 7)
    df["month_sin"]  = np.sin(2 * np.pi * df["Time"].dt.month     / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df["Time"].dt.month     / 12)
    df["is_weekend"] = (df["Time"].dt.dayofweek >= 5).astype(int)

    df = df.ffill().bfill()
    meta["fdd_cols"] = fdd_cols
    return df, fdd_cols, meta


def feature_list(rolling_col: str, fdd_cols: list) -> list:
    return [
        "cell_id", "site_id", "region_id",
        "prb_lag_1d", "throughput_lag_1d", "users_lag_1d",
        "prb_lag_1w", "throughput_lag_1w", "users_lag_1w",
        "prb_lag_2w", "throughput_lag_2w", "users_lag_2w",
        rolling_col,
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_weekend",
    ] + fdd_cols


def evaluate(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else float("nan")
    return round(mae, 4), round(rmse, 4), round(mape, 4)


# ── Public API ────────────────────────────────────────────────────────────────
def train_all(
    data_path: str | None = None,
    models_dir: str | None = None,
    progress_cb=None,
) -> str:
    """
    Train all three XGBoost models and save them to models_dir.

    Parameters
    ----------
    data_path   : path to raw KPI CSV; defaults to datasets/final_kpi_data.csv
    models_dir  : directory to save models/scalers/meta; defaults to Models/
    progress_cb : optional callable(str) invoked at each milestone for live UI updates

    Returns
    -------
    str — human-readable status: starts with "✅" on success or "❌" on failure
    """
    def _log(msg: str):
        if progress_cb:
            progress_cb(msg)

    data_path  = data_path  or _DEFAULT_DATA
    models_dir = models_dir or _HERE

    # ── Validate input ────────────────────────────────────────────────────────
    if not os.path.exists(data_path):
        return f"❌  File not found: {data_path}"

    missing_cols = []
    try:
        _log("Checking columns …")
        header = pd.read_csv(data_path, nrows=0)
        missing_cols = [c for c in _REQUIRED_COLS if c not in header.columns]
    except Exception as exc:
        return f"❌  Could not read file: {exc}"

    if missing_cols:
        return f"❌  Missing required columns: {missing_cols}"

    try:
        # ── Load ──────────────────────────────────────────────────────────────
        _log("Loading data …")
        raw = pd.read_csv(data_path, low_memory=False)
        _log(f"Loaded {len(raw):,} rows × {raw.shape[1]} columns")

        # ── Feature engineering ───────────────────────────────────────────────
        _log("Building features (lags + rolling means + time encoding) …")
        df, fdd_cols, meta = build_features(raw)
        split_date = df["Time"].quantile(0.8)
        _log(f"Train/test split at {str(split_date)[:10]}  "
             f"({len(df[df['Time'] <= split_date]):,} train / "
             f"{len(df[df['Time'] > split_date]):,} test rows)")

        os.makedirs(models_dir, exist_ok=True)
        metrics_log = []

        # ── Train each KPI ────────────────────────────────────────────────────
        for idx, (target, (model_file, scaler_file, rolling_col)) in enumerate(_TARGETS.items(), 1):
            _log(f"Training {target} model ({idx}/3) …")

            d = df.copy()
            d["target"] = d.groupby("Cell Name")[target].shift(-HORIZON)
            d = d.dropna(subset=["target"])

            feats = feature_list(rolling_col, fdd_cols)
            train = d[d["Time"] <= split_date]
            test  = d[d["Time"] >  split_date]

            X_tr, y_tr = train[feats].values, train["target"].values
            X_te, y_te = test[feats].values,  test["target"].values

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = xgb.XGBRegressor(
                n_estimators=100, random_state=42,
                eval_metric="rmse", verbosity=0,
            )
            model.fit(X_tr_s, y_tr)
            preds = model.predict(X_te_s)

            mae, rmse, mape = evaluate(y_te, preds)
            metrics_log.append(f"{target}: MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%")
            _log(f"  ✓ {target}  MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.1f}%  → saving …")

            model.save_model(os.path.join(models_dir, model_file  + ".json"))
            joblib.dump(scaler,  os.path.join(models_dir, scaler_file + ".pkl"))

        # ── Save meta ─────────────────────────────────────────────────────────
        _log("Saving meta.json …")
        meta["feature_lists"] = {
            "DL PRB Load":            feature_list("prb_rolling_24h",        fdd_cols),
            "Active Users":           feature_list("users_rolling_24h",       fdd_cols),
            "LTE DL User Throughput": feature_list("throughput_rolling_24h",  fdd_cols),
        }
        with open(os.path.join(models_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        summary = "  |  ".join(metrics_log)
        return f"✅  Training complete — 3 models saved  ({summary})"

    except Exception as exc:
        return f"❌  Training failed: {exc}"


# ── CLI entry point ───────────────────────────────────────────────────────────
def main():
    result = train_all(progress_cb=print)
    print(f"\n{result}")


if __name__ == "__main__":
    main()
