"""
Dash-Leaflet map GeoJSON builders and JavaScript style function strings.
"""

import numpy as np

from config import (
    CONGESTION_COLORS, NO_DATA_COLOR, LEVEL_ORDER,
    SECTOR_ZOOM_THRESHOLD,
    FREQ_CONE_PARAMS, FREQ_CONE_DEFAULT,
)
from geo_utils import _edges_vectorized
from precompute import CELL_POLYGONS, CELL_GEO_META, SITE_SUMMARY

# Sector radius is halved for every zoom level above SECTOR_ZOOM_THRESHOLD
# so cones shrink proportionally as you zoom in and stop overlapping.
_SECTOR_MIN_M = 15   # floor (metres) — keeps tiny sectors clickable


def _freq_lookup(freq_str: str) -> dict:
    """Return the FREQ_CONE_PARAMS entry for a raw frequency string."""
    s = str(freq_str).strip().lower()
    if s in ("tdd", "td", "2600", "tdd2600"):
        return FREQ_CONE_PARAMS["tdd"]
    try:
        return FREQ_CONE_PARAMS.get(int(float(s)), FREQ_CONE_DEFAULT)
    except (ValueError, TypeError):
        return FREQ_CONE_DEFAULT


# ── JavaScript function references for Dash Leaflet 1.x ────────────────────
#
# DL 1.x resolves {"variable": "path"} by looking up window["path"] using
# dot-notation split.  Raw function strings are returned unchanged (not eval'd)
# so they would throw "c is not a function" at runtime.
# Functions must be defined in assets/map_functions.js which Dash auto-serves.

SITE_POINT_TO_LAYER = {"variable": "mapFunctions.sitePointToLayer"}
SECTOR_STYLE        = {"variable": "mapFunctions.sectorStyle"}


# ── Site/sector colour helpers ──────────────────────────────────────────────

def _site_color(site_cells: list, cong_lookup: dict) -> tuple[str, str]:
    levels = [cong_lookup.get(c) for c in site_cells if c in cong_lookup]
    levels = [l for l in levels if l is not None]
    if not levels:
        return "No Data", NO_DATA_COLOR
    best = max(levels, key=lambda lv: LEVEL_ORDER.get(lv, 0))
    return best, CONGESTION_COLORS.get(best, NO_DATA_COLOR)


# ── GeoJSON builders ────────────────────────────────────────────────────────

def build_sites_geojson(
    visible_cells: frozenset,
    cong_lookup: dict,
    selected_site_keys: list | None = None,
    selected_cell_names: list | None = None,
) -> dict:
    if SITE_SUMMARY.empty:
        return {"type": "FeatureCollection", "features": []}

    sel_sites = set(selected_site_keys  or [])
    features  = []

    for _, row in SITE_SUMMARY.iterrows():
        sk         = row["SiteKey"]
        site_cells = [c for c in row["cells"] if c in visible_cells]
        if not site_cells:
            continue
        congestion, color = _site_color(site_cells, cong_lookup)
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["lon"]), float(row["lat"])],
            },
            "properties": {
                "site_key":   sk,
                "site_name":  row["site_name"],
                "region":     row["region"],
                "cell_count": len(site_cells),
                "congestion": congestion,
                "color":      color,
                "selected":   sk in sel_sites,
            },
        })

    return {"type": "FeatureCollection", "features": features}


def build_sectors_geojson(
    visible_cells: frozenset,
    cong_lookup: dict,
    selected_site_keys: list | None = None,
    selected_cell_names: list | None = None,
    zoom: int = SECTOR_ZOOM_THRESHOLD,
    approach: str = "ta",
) -> dict:
    """
    approach = "ta"   → TA-derived radius, fixed half-angle (original)
    approach = "freq" → frequency-derived radius + per-cell half-angle;
                        800 MHz = long+narrow, 2100 MHz = short+wide, TDD = small
    In both cases the radius is halved for every zoom level above
    SECTOR_ZOOM_THRESHOLD so cones stay proportional on screen.
    Smaller cones are rendered last (on top in SVG) for easier selection.
    """
    sel_sites = set(selected_site_keys  or [])
    sel_cells = set(selected_cell_names or [])

    # ── Collect per-cell metadata ───────────────────────────────────────────
    cell_names, lats, lons, azs = [], [], [], []
    base_ranges, half_angles    = [], []

    for cn in visible_cells:
        if cn not in CELL_GEO_META:
            continue
        m = CELL_GEO_META[cn]
        cell_names.append(cn)
        lats.append(m["lat"])
        lons.append(m["lon"])
        azs.append(m["azimuth"])

        if approach == "freq":
            p = _freq_lookup(m.get("frequency", ""))
            base_ranges.append(p["range_m"])
            half_angles.append(p["half_angle"])
        else:
            base_ranges.append(m["range_m"])
            half_angles.append(None)   # uniform angle handled below

    if not cell_names:
        return {"type": "FeatureCollection", "features": []}

    # ── Zoom-proportional scaling ───────────────────────────────────────────
    # Cap effective_zoom at threshold so selected-site sectors shown at low
    # zoom don't blow up in size.
    effective_zoom = max(SECTOR_ZOOM_THRESHOLD, int(zoom or SECTOR_ZOOM_THRESHOLD))
    scale = 2.0 ** (SECTOR_ZOOM_THRESHOLD - effective_zoom)
    scaled_ranges = np.maximum(
        np.array(base_ranges, dtype=float) * scale,
        _SECTOR_MIN_M,
    )

    lats_a = np.array(lats, dtype=float)
    lons_a = np.array(lons, dtype=float)
    azs_a  = np.array(azs,  dtype=float)

    # ── Compute arc edge points ─────────────────────────────────────────────
    if approach == "freq":
        ha_arr = np.array(half_angles, dtype=float)
        edge_lats, edge_lons = _edges_vectorized(
            lats_a, lons_a, azs_a, scaled_ranges, half_angle=ha_arr
        )
    else:
        edge_lats, edge_lons = _edges_vectorized(lats_a, lons_a, azs_a, scaled_ranges)

    # ── Build feature list ──────────────────────────────────────────────────
    items = []   # (scaled_range_m, feature_dict)
    for i, cn in enumerate(cell_names):
        meta  = CELL_GEO_META[cn]
        level = cong_lookup.get(cn)
        color = CONGESTION_COLORS.get(level, NO_DATA_COLOR) if level else NO_DATA_COLOR
        sr    = float(scaled_ranges[i])

        center = [lons_a[i], lats_a[i]]
        arc    = [[float(edge_lons[i, j]), float(edge_lats[i, j])]
                  for j in range(edge_lons.shape[1])]
        coords = [center] + arc + [center]

        items.append((sr, {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "cell_name": cn,
                "site_key":  meta["site_key"],
                "site_name": meta["site"],
                "region":    meta["region"],
                "azimuth":   round(meta["azimuth"], 1),
                "range_m":   int(meta["range_m"]),
                "frequency": meta.get("frequency", ""),
                "coverage":  meta["coverage"],
                "congestion": level or "No Data",
                "color":      color,
                "selected":  (cn in sel_cells or meta["site_key"] in sel_sites),
            },
        }))

    # Largest cones first → smallest rendered last → smallest sits on top in SVG
    items.sort(key=lambda x: x[0], reverse=True)
    return {"type": "FeatureCollection", "features": [f for _, f in items]}


# ── Click-resolution helpers (latlng → feature) ─────────────────────────────

def find_nearest_site(
    lat: float, lon: float,
    visible_cells: frozenset,
    zoom: int = 10,
) -> str | None:
    """
    Return the SiteKey of the visible site closest to (lat, lon).
    Tolerance scales with zoom so that cluster-centre clicks at low zoom
    don't accidentally select random sites.
    """
    if SITE_SUMMARY.empty:
        return None
    # tolerance in degrees: ~10 px at the given zoom level
    tolerance = max(0.0015, 0.05 / max(1, 2 ** (zoom - 8)))

    min_dist   = float("inf")
    best_key   = None
    for _, row in SITE_SUMMARY.iterrows():
        if not any(c in visible_cells for c in row["cells"]):
            continue
        dist = ((row["lat"] - lat) ** 2 + (row["lon"] - lon) ** 2) ** 0.5
        if dist < tolerance and dist < min_dist:
            min_dist = dist
            best_key = row["SiteKey"]
    return best_key


def find_sector_at_point(
    lat: float, lon: float,
    visible_cells: frozenset,
) -> tuple[str | None, str | None]:
    """
    Return (cell_name, site_key) of the first sector polygon that contains
    the point (lat, lon).  Returns (None, None) if none matches.
    """
    from geo_utils import point_in_polygon
    for cn in visible_cells:
        if cn not in CELL_POLYGONS or cn not in CELL_GEO_META:
            continue
        if point_in_polygon(lat, lon, CELL_POLYGONS[cn]):
            return cn, CELL_GEO_META[cn]["site_key"]
    return None, None
