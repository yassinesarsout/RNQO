"""
Module-level data loading and precomputation (runs once at import time).

KPI side:
  DF, LOAD_ERROR, NUMERIC_COLS, KPI_OPTIONS, DATE_MIN, DATE_MAX
  get_filtered_df()

Geo side:
  GEO_DF, GEO_LOAD_ERROR, GEO_COVERAGE_TYPES
  CELL_POLYGONS  — {cell_name: [[lon,lat], ...]}  (closed GeoJSON ring)
  CELL_GEO_META  — {cell_name: {lat, lon, azimuth, range_m, site_key, region, site, coverage}}
  SITE_SUMMARY   — DataFrame with one row per site
  get_geo_visible_cells()
  get_cell_congestion_lookup()
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

from config import (
    DATA_PATH, GEO_DATA_PATH, _FILTER_CACHE_MAX,
    COVERAGE_RANGES, DEFAULT_CONE_RANGE, CELL_RANGE_INCREMENT,
)
from geo_utils import _edges_vectorized, make_site_key, normalize_level
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

if not DF.empty and "Time" in DF.columns:
    DF["Time"] = pd.to_datetime(DF["Time"], errors="coerce", dayfirst=False)
if not DF.empty and "CongestionLevel" in DF.columns:
    DF["CongestionLevel"] = DF["CongestionLevel"].apply(normalize_level)

NUMERIC_COLS = numeric_columns(DF) if not DF.empty else []
KPI_OPTIONS  = [{"label": c, "value": c} for c in NUMERIC_COLS]

from datetime import timedelta as _td
_has_time = not DF.empty and "Time" in DF.columns and DF["Time"].notna().any()
DATE_MIN    = DF["Time"].min().date().isoformat()                    if _has_time else None
DATE_MAX    = DF["Time"].max().date().isoformat()                    if _has_time else None
DATE_SECOND = (DF["Time"].min().date() + _td(days=1)).isoformat()   if _has_time else None


# ============================================================
# Load geo data
# ============================================================
try:
    GEO_DF = load_dataframe(GEO_DATA_PATH)
    GEO_LOAD_ERROR = None
    for col in ["Longitude_Sector", "Latitude_Sector", "Azimuth", "expected_cell_range"]:
        if col in GEO_DF.columns:
            GEO_DF[col] = pd.to_numeric(GEO_DF[col], errors="coerce")
    GEO_DF = GEO_DF.dropna(subset=["Longitude_Sector", "Latitude_Sector"])
except Exception as exc:
    GEO_DF = pd.DataFrame()
    GEO_LOAD_ERROR = str(exc)

GEO_COVERAGE_TYPES = (
    sorted(GEO_DF["CoverageType"].dropna().unique().tolist())
    if not GEO_DF.empty and "CoverageType" in GEO_DF.columns else []
)


# ============================================================
# LRU filter cache (KPI data)
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


# ============================================================
# Geo precomputation — polygons and metadata
# ============================================================
CELL_POLYGONS: dict = {}   # {cell_name: [[lon, lat], ...]}  GeoJSON closed ring
CELL_GEO_META: dict = {}   # {cell_name: {lat, lon, azimuth, range_m, site_key, ...}}

if not GEO_DF.empty:
    _geo = GEO_DF.copy()

    # Attach Region / Radio Site from KPI data when available
    if not DF.empty and "Cell Name" in DF.columns and "Radio Site" in DF.columns:
        _map_cols = ["Cell Name", "Radio Site"]
        if "Region" in DF.columns:
            _map_cols.append("Region")
        _site_lookup = DF[_map_cols].drop_duplicates(subset=["Cell Name"])
        _geo = _geo.merge(_site_lookup, on="Cell Name", how="left")

    if "Radio Site" not in _geo.columns:
        _geo["Radio Site"] = _geo["Cell Name"].astype(str).str.rsplit("_", n=1).str[0]
    if "Region" not in _geo.columns:
        _geo["Region"] = "Unknown"
    _geo["Radio Site"] = _geo["Radio Site"].fillna("Unknown")
    _geo["Region"]     = _geo["Region"].fillna("Unknown")
    _geo = _geo.drop_duplicates(subset=["Cell Name"]).reset_index(drop=True)

    _lats       = _geo["Latitude_Sector"].values.astype(float)
    _lons       = _geo["Longitude_Sector"].values.astype(float)
    _azs        = _geo["Azimuth"].fillna(0).values.astype(float)
    _cell_names = _geo["Cell Name"].astype(str).values
    _sites      = _geo["Radio Site"].astype(str).values
    _regions    = _geo["Region"].astype(str).values
    _coverages  = (_geo["CoverageType"].astype(str).values
                   if "CoverageType" in _geo.columns
                   else np.array([""] * len(_geo)))

    # ---- Assign per-cell range based on coverage type + index within site ----
    # Group cells by site, sort by azimuth, assign increasing range per index
    _sk_list = [make_site_key(_regions[i], _sites[i]) for i in range(len(_geo))]
    _site_az_idx: dict = {}  # site_key → [(azimuth, geo_row_index)]
    for i in range(len(_geo)):
        _site_az_idx.setdefault(_sk_list[i], []).append((float(_azs[i]), i))

    _cell_ranges = np.zeros(len(_geo), dtype=float)
    for _sk, _pairs in _site_az_idx.items():
        _sorted = sorted(_pairs)  # sort by azimuth → stable cell index
        for _cell_idx, (_, _geo_idx) in enumerate(_sorted):
            _cov       = str(_coverages[_geo_idx])
            _base      = COVERAGE_RANGES.get(_cov, DEFAULT_CONE_RANGE)
            _cell_ranges[_geo_idx] = _base + _cell_idx * CELL_RANGE_INCREMENT

    # ---- Vectorised cone computation ----
    if len(_geo) > 0:
        _edge_lats, _edge_lons = _edges_vectorized(
            _lats, _lons, _azs, _cell_ranges
        )
        for _i in range(len(_geo)):
            _cn  = _cell_names[_i]
            _lat = float(_lats[_i])
            _lon = float(_lons[_i])

            # GeoJSON polygon ring: center → arc points → center (closed)
            _coords = [[_lon, _lat]]
            for _j in range(_edge_lons.shape[1]):
                _coords.append([float(_edge_lons[_i, _j]), float(_edge_lats[_i, _j])])
            _coords.append([_lon, _lat])

            CELL_POLYGONS[_cn] = _coords
            CELL_GEO_META[_cn] = {
                "lat":      _lat,
                "lon":      _lon,
                "azimuth":  float(_azs[_i]),
                "range_m":  float(_cell_ranges[_i]),
                "site_key": _sk_list[_i],
                "region":   _regions[_i],
                "site":     _sites[_i],
                "coverage": _coverages[_i],
            }


# ============================================================
# Site summary
# ============================================================
SITE_SUMMARY: pd.DataFrame = pd.DataFrame()

if CELL_GEO_META:
    _groups: dict = {}
    for _cn, _m in CELL_GEO_META.items():
        _sk = _m["site_key"]
        if _sk not in _groups:
            _groups[_sk] = {
                "SiteKey": _sk, "region": _m["region"], "site_name": _m["site"],
                "lats": [], "lons": [], "cells": [],
            }
        _groups[_sk]["lats"].append(_m["lat"])
        _groups[_sk]["lons"].append(_m["lon"])
        _groups[_sk]["cells"].append(_cn)

    SITE_SUMMARY = pd.DataFrame([{
        "SiteKey":    g["SiteKey"],
        "region":     g["region"],
        "site_name":  g["site_name"],
        "lat":        float(np.median(g["lats"])),
        "lon":        float(np.median(g["lons"])),
        "cell_count": len(g["cells"]),
        "cells":      g["cells"],
    } for g in _groups.values()])


# ============================================================
# Dynamic geo helpers (called per-render)
# ============================================================

def get_geo_visible_cells(regions, sites, cells) -> frozenset:
    """Return cell names to display on the map based on dropdown filters only."""
    if not CELL_GEO_META:
        return frozenset()
    if not regions and not sites and not cells:
        return frozenset(CELL_GEO_META.keys())
    region_set = set(regions or [])
    site_set   = set(sites   or [])
    cell_set   = set(cells   or [])
    result = set()
    for cn, meta in CELL_GEO_META.items():
        if region_set and meta["region"] not in region_set:
            continue
        if site_set   and meta["site"]   not in site_set:
            continue
        if cell_set   and cn             not in cell_set:
            continue
        result.add(cn)
    return frozenset(result)


def _build_cong_lookup(df: pd.DataFrame) -> dict:
    """Build {cell_name: 'P0'|'P1'|'Not Congested'} from a filtered DataFrame."""
    if df.empty or "Cell Name" not in df.columns:
        return {}
    has_p0 = "P0_Congestion" in df.columns
    has_p1 = "P1_Congestion" in df.columns
    if not has_p0 and not has_p1:
        return {}
    agg = {}
    if has_p0: agg["P0_Congestion"] = "any"
    if has_p1: agg["P1_Congestion"] = "any"
    cell_agg = (df[["Cell Name"] + list(agg.keys())]
                .dropna(subset=["Cell Name"])
                .groupby("Cell Name").agg(agg))
    result = {}
    for cn, row in cell_agg.iterrows():
        p0 = bool(row.get("P0_Congestion", False)) if has_p0 else False
        p1 = bool(row.get("P1_Congestion", False)) if has_p1 else False
        result[str(cn)] = "P0" if p0 else ("P1" if p1 else "Not Congested")
    return result


def get_cell_congestion_lookup(regions, sites, cells, modes,
                               start, end, animation_hour=None) -> dict:
    """
    Return {cell_name: level} for cells that appear in the KPI data matching
    the current filters.  Cells in geo data but absent here → colored gray
    ('No Data') by the caller.
    """
    filtered = get_filtered_df(regions, sites, cells, modes, start, end)
    if filtered.empty:
        return {}

    if animation_hour is not None:
        try:
            h = int(animation_hour)
            if "Hour" in filtered.columns:
                sub = filtered[filtered["Hour"] == h]
                if not sub.empty:
                    filtered = sub
        except (TypeError, ValueError):
            pass

    return _build_cong_lookup(filtered)
