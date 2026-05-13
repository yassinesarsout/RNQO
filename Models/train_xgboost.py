"""
XGBoost training script for RAN KPI forecasting.

Trains three models (one per KPI) at a 24-hour direct forecast horizon,
exactly replicating the feature engineering from the comparison notebooks.

Outputs (all written to this directory — Models/):
  xgb_prb.json          XGBoost model for DL PRB Load
  xgb_users.json         XGBoost model for Active Users
  xgb_throughput.json    XGBoost model for LTE DL User Throughput
  scaler_prb.pkl         StandardScaler for PRB model
  scaler_users.pkl       StandardScaler for Users model
  scaler_throughput.pkl  StandardScaler for Throughput model
  meta.json              Categorical encodings + FDD column names

Usage:
  cd "c:/Users/yassi/PFE dashboard"
  python Models/train_xgboost.py
"""

import os, json, warnings
import numpy  as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
_DATA_PATH = os.path.join(_HERE, "..", "datasets", "final_kpi_data.csv")
HORIZON    = 24   # hours ahead to predict

_TARGETS = {
    "DL PRB Load":            ("xgb_prb",         "scaler_prb",        "prb_rolling_24h"),
    "Active Users":           ("xgb_users",        "scaler_users",      "users_rolling_24h"),
    "LTE DL User Throughput": ("xgb_throughput",   "scaler_throughput", "throughput_rolling_24h"),
}

_LAG_2W, _LAG_1W, _LAG_1D = 24 * 14, 24 * 7, 24


# ── Feature engineering (mirrors all three notebooks) ────────────────────────
def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """
    Applies all feature engineering and returns:
      (df_with_features, fdd_col_names, meta_encodings)
    """
    df = raw[["Time", "Cell Name", "Radio Site", "Region",
              "Active Users", "LTE DL User Throughput", "DL PRB Load",
              "Cell FDD TDD Indication"]].copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["Cell Name", "Time"])

    # Integer identity encodings
    cell_cat   = df["Cell Name"].astype("category")
    site_cat   = df["Radio Site"].astype("category")
    region_cat = df["Region"].astype("category")

    df["cell_id"]   = cell_cat.cat.codes
    df["site_id"]   = site_cat.cat.codes
    df["region_id"] = region_cat.cat.codes

    meta = {
        "cell_codes":    dict(zip(cell_cat.cat.categories,   range(len(cell_cat.cat.categories)))),
        "site_codes":    dict(zip(site_cat.cat.categories,   range(len(site_cat.cat.categories)))),
        "region_codes":  dict(zip(region_cat.cat.categories, range(len(region_cat.cat.categories)))),
        "cell_to_site":   df.drop_duplicates("Cell Name").set_index("Cell Name")["Radio Site"].to_dict(),
        "cell_to_region": df.drop_duplicates("Cell Name").set_index("Cell Name")["Region"].to_dict(),
    }

    # FDD/TDD one-hot
    df = pd.get_dummies(df, columns=["Cell FDD TDD Indication"])
    fdd_cols = sorted([c for c in df.columns if "Cell FDD TDD Indication_" in c])

    # Lag features (2-week, 1-week, 24-hour for all three KPIs)
    for kpi, prefix in [
        ("DL PRB Load",            "prb"),
        ("LTE DL User Throughput", "throughput"),
        ("Active Users",           "users"),
    ]:
        grp = df.groupby("Cell Name")[kpi]
        df[f"{prefix}_lag_2w"] = grp.shift(_LAG_2W)
        df[f"{prefix}_lag_1w"] = grp.shift(_LAG_1W)
        df[f"{prefix}_lag_1d"] = grp.shift(_LAG_1D)

    # Rolling 24h mean per target KPI (shifted 1 step to avoid leakage)
    for kpi, col in [
        ("DL PRB Load",            "prb_rolling_24h"),
        ("Active Users",           "users_rolling_24h"),
        ("LTE DL User Throughput", "throughput_rolling_24h"),
    ]:
        df[col] = (
            df.groupby("Cell Name")[kpi]
            .transform(lambda x: x.shift(1).rolling(24, min_periods=1).mean())
        )

    # Cyclical time features
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
    """Return the exact feature order used during training."""
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading data from {_DATA_PATH} …")
    raw = pd.read_csv(_DATA_PATH, low_memory=False)
    print(f"  {len(raw):,} rows  ×  {raw.shape[1]} columns")

    print("Building features …")
    df, fdd_cols, meta = build_features(raw)

    split_date = df["Time"].quantile(0.8)
    print(f"Train/test split at {split_date}")

    for target, (model_file, scaler_file, rolling_col) in _TARGETS.items():
        print(f"\n{'='*60}\n  Training XGBoost → {target}  (H+{HORIZON})")

        # Direct forecast target
        d = df.copy()
        d["target"] = d.groupby("Cell Name")[target].shift(-HORIZON)
        d = d.dropna(subset=["target"])

        feats     = feature_list(rolling_col, fdd_cols)
        train     = d[d["Time"] <= split_date]
        test      = d[d["Time"] >  split_date]
        X_tr, y_tr = train[feats].values, train["target"].values
        X_te, y_te = test[feats].values,  test["target"].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = xgb.XGBRegressor(
            n_estimators=100,
            random_state=42,
            eval_metric="rmse",
            verbosity=0,
        )
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)

        mae, rmse, mape = evaluate(y_te, preds)
        print(f"  Test  MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

        model_path  = os.path.join(_HERE, model_file  + ".json")
        scaler_path = os.path.join(_HERE, scaler_file + ".pkl")
        model.save_model(model_path)
        joblib.dump(scaler, scaler_path)
        print(f"  Saved → {model_path}")
        print(f"  Saved → {scaler_path}")

    meta["feature_lists"] = {
        "DL PRB Load":            feature_list("prb_rolling_24h",         fdd_cols),
        "Active Users":           feature_list("users_rolling_24h",        fdd_cols),
        "LTE DL User Throughput": feature_list("throughput_rolling_24h",   fdd_cols),
    }
    meta_path = os.path.join(_HERE, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta saved → {meta_path}")
    print("\nDone. All three XGBoost models trained and saved.")


if __name__ == "__main__":
    main()
