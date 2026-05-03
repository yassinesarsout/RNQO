"""
Module-level data loading and precomputation.

Everything expensive that doesn't depend on user interaction is computed
exactly once here at import time:
  • DF              — raw KPI dataframe
  • NUMERIC_COLS    — numeric column names
  • KPI_OPTIONS     — dropdown options list
  • DATE_MIN/MAX    — date range for the date picker
  • LRU filter cache (get_filtered_df)
"""

from collections import OrderedDict

import pandas as pd

from config import DATA_PATH, _FILTER_CACHE_MAX
from geo_utils import normalize_level
from data_utils import load_dataframe, numeric_columns, _filter_dataframe_impl


# ============================================================
# Load raw KPI data
# ============================================================
try:
    DF = load_dataframe(DATA_PATH)
    LOAD_ERROR = None
except Exception as exc:
    DF = pd.DataFrame()
    LOAD_ERROR = str(exc)


# ============================================================
# Post-process KPI data
# ============================================================
if not DF.empty and "Time" in DF.columns:
    DF["Time"] = pd.to_datetime(DF["Time"], errors="coerce", dayfirst=False)
if not DF.empty and "CongestionLevel" in DF.columns:
    DF["CongestionLevel"] = DF["CongestionLevel"].apply(normalize_level)

NUMERIC_COLS = numeric_columns(DF) if not DF.empty else []
KPI_OPTIONS  = [{"label": c, "value": c} for c in NUMERIC_COLS]

DATE_MIN = (DF["Time"].min().isoformat()
            if not DF.empty and "Time" in DF.columns and DF["Time"].notna().any() else None)
DATE_MAX = (DF["Time"].max().isoformat()
            if not DF.empty and "Time" in DF.columns and DF["Time"].notna().any() else None)


# ============================================================
# LRU filter cache
# ============================================================
_df_filter_cache: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()


def _filter_key(regions, sites, cells, modes, start, end) -> tuple:
    return (
        tuple(sorted(regions or [])),
        tuple(sorted(sites   or [])),
        tuple(sorted(cells   or [])),
        tuple(sorted(modes   or [])),
        str(start) if start else "",
        str(end)   if end   else "",
    )


def get_filtered_df(regions, sites, cells, modes, start, end) -> pd.DataFrame:
    if DF.empty:
        return DF
    key = _filter_key(regions, sites, cells, modes, start, end)
    if key in _df_filter_cache:
        _df_filter_cache.move_to_end(key)
        return _df_filter_cache[key]
    result = _filter_dataframe_impl(DF, regions, sites, cells, modes, start, end)
    _df_filter_cache[key] = result
    if len(_df_filter_cache) > _FILTER_CACHE_MAX:
        _df_filter_cache.popitem(last=False)
    return result
