"""
LTE Network Data Cleaning Pipeline
====================================
Cleans and prepares two LTE datasets:
  1. KPI time-series data  (kpi_with_congestion.csv → final_kpi_data.csv)
  2. Cell configuration/info data (LTE_cell_Plan_HU_MasterFile.csv → final_info_data.csv)

Usage
-----
Run directly:
    python lte_data_pipeline.py

Or import and call from another module:
    from lte_data_pipeline import run_pipeline
    kpi_df, info_df = run_pipeline(
        kpi_path="path/to/kpi.csv",
        conf_path="path/to/masterfile.csv",
        kpi_out="path/to/final_kpi_data.csv",
        info_out="path/to/final_info_data.csv",
    )
"""

import re
import sys
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONF_KEEP_COLS = [
    "CellName",
    "Longitude_Sector",
    "Latitude_Sector",
    "CoverageType",
    "Azimuth",
    "LAC",
    "expected_cell_range",
]

KPI_KEEP_COLS = [
    "Time",
    "Cell Name",
    "Cell FDD TDD Indication",
    "L.Traffic.ActiveUser.DL.Avg",
    "FT_AVE 4G/LTE DL USER THRPUT without Last TTI(ALL) (KBPS)(kbit/s)",
    "FT_PHYSICAL RESOURCE BLOCKS LOAD DL(%)",
    "FT_4G/LTE AVERAGE REPORTED CQI",
    "Average TA",
]

RENAME_DICT = {
    "FT_PHYSICAL RESOURCE BLOCKS LOAD DL(%)": "DL PRB Load",
    "FT_PHYSICAL RESOURCE BLOCKS LOAD UL": "UL PRB Load",
    "FT_AVERAGE NB OF USERS (UEs RRC CONNECTED)": "Connected Users",
    "L.Traffic.User.Avg": "Traffic Users",
    "L.Traffic.ActiveUser.DL.Avg": "Active Users",
    "L.Traffic.ActiveUser.UL.Avg": "UL Active Users",
    "FT_AVERAGE NUMBER OF UE QUEUED DL": "DL Queued Users",
    "FT_AVERAGE NUMBER OF UE QUEUED UL": "UL Queued Users",
    "FT_AVE NSA LTE/NR Ul User Thrput (ALL) (KBPS)(kbit/s)": "NSA UL User Throughput",
    "FT_AVE NSA LTE/NR Dl User Thrput (ALL) (KBPS)(kbit/s)": "NSA DL User Throughput",
    "FT_NSA LTE/NR UL TRAFFIC VOLUME (GBYTES)(GB/s)": "NSA UL Traffic Volume",
    "FT_NSA LTE/NR DL TRAFFIC VOLUME (GBYTES)(GB/s)": "NSA DL Traffic Volume",
    "FT_4G/LTE DL TRAFFIC VOLUME (GBYTES)": "LTE DL Traffic Volume",
    "FT_UL.Interference": "UL Interference",
    "DL Average Throughput (Kbps)": "DL Avg Throughput",
    "FT_AVE 4G/LTE DL USER THRPUT without Last TTI(ALL) (KBPS)(kbit/s)": "LTE DL User Throughput",
    "Downlink Resource Block Utilizing Rate (NSA DC)(%)": "NSA DC DL RB Utilization",
    "Average TA": "Average Timing Advance",
    "FT_4G/LTE DROP CALL RATE": "Drop Call Rate",
    "FT_4G/LTE AVERAGE REPORTED CQI": "Average CQI",
    "FT_RRC SUCCESS RATE": "RRC Success Rate",
    "FT_4G/LTE CALL SETUP SUCCESS RATE": "Call Setup Success Rate",
    "L.E-RAB.AbnormRel.Cong": "Congestion E-RAB Drops",
    "FT_4G/LTE RADIO DOWNLINK DELAY(ms)": "DL Radio Delay",
}

CELL_COL = "Cell Name"
DATETIME_COL = "Time"
KPI_COLS = [
    "L.Traffic.ActiveUser.DL.Avg",
    "FT_AVE 4G/LTE DL USER THRPUT without Last TTI(ALL) (KBPS)(kbit/s)",
    "FT_PHYSICAL RESOURCE BLOCKS LOAD DL(%)",
    "FT_4G/LTE AVERAGE REPORTED CQI",
    "Average TA",
]

MISSING_THRESHOLD_PCT = 25  # cells with >25% missing in any KPI are dropped


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _clean_numeric_col(series: pd.Series) -> pd.Series:
    """Strip formatting characters and coerce to float."""
    return pd.to_numeric(
        series.astype(str)
              .str.strip()
              .str.replace(",", "", regex=False)
              .str.replace("%", "", regex=False)
              .replace({"nan": np.nan, "None": np.nan, "": np.nan, "NIL": np.nan}),
        errors="coerce",
    )


def _extract_region_and_site(cell_name_series: pd.Series):
    """Return (region, radio_site) Series derived from Cell Name."""
    extracted = (
        cell_name_series.astype(str)
        .str.strip()
        .str.extract(r"^([A-Za-z]{3})_(\d+)_([A-Za-z0-9]+)$")
    )
    region = (
        extracted[0]
        .fillna(cell_name_series.astype(str).str.extract(r"^([A-Za-z]{3})")[0])
        .str.upper()
    )
    site_num = extracted[1].fillna(
        cell_name_series.astype(str).str.extract(r"^[A-Za-z]{3}_(\d+)")[0]
    )
    radio_site = region + "_" + site_num
    return region, radio_site


def _get_frequency(cell_id: str) -> str:
    """Map the last cell-ID segment to a frequency band label."""
    cell_id = re.sub(r"^\d+", "", str(cell_id).lower())
    match = re.match(r"^([a-z])", cell_id)
    if not match:
        return "Unknown"
    frequency_map = {
        "h": "1800",
        "l": "2100",
        "f": "800",
        "j": "800",
        "g": "TDD",
        "k": "TDD",
    }
    return frequency_map.get(match.group(1), "Unknown")


# ---------------------------------------------------------------------------
# Stage 1 — Load raw data
# ---------------------------------------------------------------------------

def load_kpi_data(path: str) -> pd.DataFrame:
    """Load the raw KPI CSV (skipping the 6-line header and last footer row)."""
    print(f"[1/6] Loading KPI data from: {path}")
    df = pd.read_csv(path, skiprows=6, skipfooter=1, engine="python")
    print(f"      → {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def load_conf_data(path: str) -> pd.DataFrame:
    """Load the cell configuration master file (semicolon-delimited)."""
    print(f"[1/6] Loading config data from: {path}")
    df = pd.read_csv(path, sep=";")
    print(f"      → {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Stage 2 — Clean configuration (cell info) data
# ---------------------------------------------------------------------------

def clean_conf_data(conf_data: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant columns, rename, coerce numeric fields,
    fill missing Azimuth, and deduplicate per Cell Name.
    """
    print("[2/6] Cleaning configuration data …")

    # Keep only needed columns
    conf_data = conf_data[CONF_KEEP_COLS].copy()
    conf_data = conf_data.rename(columns={"CellName": "Cell Name"})

    # --- Azimuth ---
    conf_data["Azimuth"] = _clean_numeric_col(conf_data["Azimuth"])
    invalid_az = conf_data[conf_data["Azimuth"].isna()]["Cell Name"].unique()
    if len(invalid_az):
        print(f"      Filling {len(invalid_az)} cells with missing Azimuth → 20")
    conf_data["Azimuth"] = conf_data["Azimuth"].fillna(20)

    # --- expected_cell_range ---
    conf_data["expected_cell_range"] = _clean_numeric_col(conf_data["expected_cell_range"])

    # --- Deduplicate: average coordinates, keep first value for other cols ---
    agg_dict = {"Longitude_Sector": "mean", "Latitude_Sector": "mean"}
    for col in conf_data.columns:
        if col not in ("Cell Name", "Longitude_Sector", "Latitude_Sector"):
            agg_dict[col] = "first"
    conf_data = conf_data.groupby("Cell Name", as_index=False).agg(agg_dict)

    print(f"      → {conf_data.shape[0]:,} unique cells after deduplication")
    return conf_data


# ---------------------------------------------------------------------------
# Stage 3 — Clean KPI data
# ---------------------------------------------------------------------------

def clean_kpi_data(kpi_data: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant KPI columns, fix the Active Users column,
    drop cells with >25% missing data, and interpolate the rest.
    """
    print("[3/6] Cleaning KPI data …")

    # Keep only needed columns
    kpi_data = kpi_data[KPI_KEEP_COLS].copy()

    # Fix 'L.Traffic.ActiveUser.DL.Avg' which may contain strings
    kpi_data["L.Traffic.ActiveUser.DL.Avg"] = _clean_numeric_col(
        kpi_data["L.Traffic.ActiveUser.DL.Avg"]
    )

    # Sort by cell and time before interpolation
    kpi_data = kpi_data.sort_values([CELL_COL, DATETIME_COL]).reset_index(drop=True)

    # Drop cells where any KPI exceeds the missing threshold
    cells_to_remove = []
    for cell, group in kpi_data.groupby(CELL_COL):
        for col in KPI_COLS:
            pct_missing = group[col].isna().mean() * 100
            if pct_missing > MISSING_THRESHOLD_PCT:
                cells_to_remove.append(cell)
                print(
                    f"      Removing cell '{cell}' — "
                    f"'{col[:50]}' has {pct_missing:.1f}% missing"
                )
                break

    kpi_data = kpi_data[~kpi_data[CELL_COL].isin(cells_to_remove)].reset_index(drop=True)
    print(f"      Removed {len(cells_to_remove)} cells, {len(kpi_data):,} rows remaining")

    # Linear interpolation per cell (handles leading/trailing NaNs too)
    kpi_data[KPI_COLS] = kpi_data.groupby(CELL_COL, group_keys=False)[KPI_COLS].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    remaining_missing = {
        col: kpi_data[col].isna().sum()
        for col in KPI_COLS
        if kpi_data[col].isna().sum() > 0
    }
    if remaining_missing:
        print(f"      Remaining NaNs after interpolation: {remaining_missing}")
    else:
        print("      No NaNs remain after interpolation")

    return kpi_data


# ---------------------------------------------------------------------------
# Stage 4 — Align both datasets to common cells
# ---------------------------------------------------------------------------

def align_to_common_cells(
    kpi_data: pd.DataFrame, conf_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only cells present in both datasets."""
    print("[4/6] Aligning to common cells …")
    common = set(kpi_data[CELL_COL]).intersection(set(conf_data[CELL_COL]))
    kpi_data = kpi_data[kpi_data[CELL_COL].isin(common)].copy()
    conf_data = conf_data[conf_data[CELL_COL].isin(common)].copy()
    print(f"      Common cells: {len(common):,}")
    return kpi_data, conf_data


# ---------------------------------------------------------------------------
# Stage 5 — Enrich / finalise KPI data
# ---------------------------------------------------------------------------

def prepare_kpi_dataframe(kpi_data: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to human-friendly names, parse Time, and
    add Region / Radio Site derived columns.
    """
    print("[5/6] Preparing final KPI dataframe …")

    kpi_data = kpi_data.copy()
    kpi_data.columns = [str(c).strip() for c in kpi_data.columns]
    kpi_data = kpi_data.rename(columns=RENAME_DICT)

    # Parse datetime
    if "Time" in kpi_data.columns:
        kpi_data["Time"] = pd.to_datetime(kpi_data["Time"], errors="coerce", dayfirst=True)
        kpi_data["Date"] = kpi_data["Time"].dt.date.astype("string")
        kpi_data["Hour"] = kpi_data["Time"].dt.hour
    else:
        kpi_data["Date"] = pd.Series(dtype="string")
        kpi_data["Hour"] = pd.Series(dtype="float")

    # Region & Radio Site
    if CELL_COL in kpi_data.columns:
        kpi_data["Region"], kpi_data["Radio Site"] = _extract_region_and_site(
            kpi_data[CELL_COL]
        )
    else:
        kpi_data["Region"] = pd.Series(dtype="string")
        kpi_data["Radio Site"] = pd.Series(dtype="string")

    # Sort
    sort_cols = [c for c in ["Region", "Radio Site", CELL_COL, "Time"] if c in kpi_data.columns]
    kpi_data = kpi_data.sort_values(sort_cols).reset_index(drop=True)

    print(f"      → {kpi_data.shape[0]:,} rows, {kpi_data.shape[1]} columns")
    return kpi_data


# ---------------------------------------------------------------------------
# Stage 5b — Enrich cell info data
# ---------------------------------------------------------------------------

def prepare_info_dataframe(conf_data: pd.DataFrame) -> pd.DataFrame:
    """Add Region, Radio Site, and frequency band columns to the config data."""
    print("[5/6] Preparing final cell info dataframe …")

    conf_data = conf_data.copy()

    # Region & Radio Site
    conf_data["Region"], conf_data["Radio Site"] = _extract_region_and_site(
        conf_data[CELL_COL]
    )

    # Frequency band derived from the last segment of the Cell Name
    conf_data["cell_id"] = conf_data[CELL_COL].str.extract(r"_([^_]+)$")
    conf_data["frequency"] = conf_data["cell_id"].apply(_get_frequency)
    conf_data.drop(columns=["cell_id"], inplace=True)

    print(f"      → {conf_data.shape[0]:,} rows, {conf_data.shape[1]} columns")
    print("      Frequency distribution:")
    for freq, count in conf_data["frequency"].value_counts().items():
        print(f"        {freq}: {count}")
    return conf_data


# ---------------------------------------------------------------------------
# Stage 6 — Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    kpi_data: pd.DataFrame,
    conf_data: pd.DataFrame,
    kpi_out: str,
    info_out: str,
) -> None:
    print("[6/6] Saving outputs …")
    kpi_data.to_csv(kpi_out, index=False)
    conf_data.to_csv(info_out, index=False)
    print(f"      KPI data  → {kpi_out}")
    print(f"      Cell info → {info_out}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    kpi_path: str = "Congestion_4G_Query_Result_20260413110442414(Subreport 1).csv",
    conf_path: str = "LTE_cell_Plan_HU_MasterFile.csv",
    kpi_out: str = "datasets/final_kpi_data.csv",
    info_out: str = "datasets/final_info_data.csv",
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full cleaning pipeline and return (kpi_df, info_df).

    Parameters
    ----------
    kpi_path  : Path to the raw KPI CSV file.
    conf_path : Path to the cell configuration master CSV file.
    kpi_out   : Where to write the cleaned KPI CSV.
    info_out  : Where to write the cleaned cell info CSV.
    save      : Set to False to skip writing output files (useful when importing).

    Returns
    -------
    kpi_df  : Cleaned, enriched KPI DataFrame.
    info_df : Cleaned, enriched cell info DataFrame.
    """
    print("=" * 60)
    print("  LTE Data Cleaning Pipeline")
    print("=" * 60)

    kpi_raw  = load_kpi_data(kpi_path)
    conf_raw = load_conf_data(conf_path)

    conf_data = clean_conf_data(conf_raw)
    kpi_data  = clean_kpi_data(kpi_raw)

    kpi_data, conf_data = align_to_common_cells(kpi_data, conf_data)

    kpi_data  = prepare_kpi_dataframe(kpi_data)
    conf_data = prepare_info_dataframe(conf_data)

    if save:
        save_outputs(kpi_data, conf_data, kpi_out, info_out)

    print("=" * 60)
    print("  Pipeline complete.")
    print("=" * 60)
    return kpi_data, conf_data


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LTE Network Data Cleaning Pipeline")
    parser.add_argument(
        "--kpi",
        default="datasets\Congestion_4G_Query_Result_20260413110442414(Subreport 1).csv",
        help="Path to raw KPI CSV",
    )
    parser.add_argument(
        "--conf",
        default="datasets\LTE_cell_Plan_HU_MasterFile.csv",
        help="Path to cell config master CSV",
    )
    parser.add_argument(
        "--kpi-out",
        default="datasets/final_kpi_data.csv",
        help="Output path for cleaned KPI CSV",
    )
    parser.add_argument(
        "--info-out",
        default="datasets/final_info_data.csv",
        help="Output path for cleaned cell info CSV",
    )
    args = parser.parse_args()

    run_pipeline(
        kpi_path=args.kpi,
        conf_path=args.conf,
        kpi_out=args.kpi_out,
        info_out=args.info_out,
    )
