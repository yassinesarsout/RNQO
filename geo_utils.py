"""
Geometry helpers — cone edge computation, site-key helpers, normalize level,
and a pure-Python point-in-polygon test (no shapely needed here).
"""

import numpy as np
import pandas as pd

from config import ALLOWED_LEVELS, FIXED_HALF_ANGLE, CONE_POINTS, SITE_KEY_SEP


def normalize_level(level) -> str:
    if pd.isna(level):
        return "Not Congested"
    raw   = str(level).strip()
    upper = raw.upper()
    if upper == "P0": return "P0"
    if upper == "P1": return "P1"
    title = raw.title()
    if title in {"High", "Critical", "Severe"}: return "P0"
    if title in {"Medium"}:                     return "P1"
    if title in {"Low", "Not Congested", "Unknown"}: return "Not Congested"
    if raw in ALLOWED_LEVELS:                   return raw
    return "Not Congested"


def _edges_vectorized(lats, lons, azimuths, ranges_m,
                      half_angle=FIXED_HALF_ANGLE, n_points=CONE_POINTS):
    """
    Vectorized sector-edge computation with per-cell ranges and optional
    per-cell half-angles.
    Returns (edge_lats, edge_lons) each shaped (n_cells, n_points+1).
    half_angle may be a scalar (same for all cells) or a 1-D array of
    length n_cells for per-cell angles (used by the frequency approach).
    """
    earth_r  = 6_371_000.0
    lats     = np.asarray(lats,     dtype=float)
    lons     = np.asarray(lons,     dtype=float)
    azimuths = np.asarray(azimuths, dtype=float)
    ranges_m = np.asarray(ranges_m, dtype=float)

    ha = np.asarray(half_angle, dtype=float)
    if ha.ndim == 0:
        # scalar — uniform angle across all cells (original behaviour)
        offsets  = np.linspace(-float(ha), float(ha), n_points + 1)
        bearings = np.radians(azimuths[:, None] + offsets[None, :])
    else:
        # per-cell array — interpolate from -ha[i] to +ha[i] for each cell
        t        = np.linspace(0.0, 1.0, n_points + 1)                   # (n_pts+1,)
        offsets  = -ha[:, None] + 2.0 * ha[:, None] * t[None, :]         # (n_cells, n_pts+1)
        bearings = np.radians(azimuths[:, None] + offsets)                # (n_cells, n_pts+1)
    lat_rad  = np.radians(lats)[:, None]
    lon_rad  = np.radians(lons)[:, None]
    d_over_r = (ranges_m / earth_r)[:, None]

    p_lat = np.arcsin(
        np.sin(lat_rad) * np.cos(d_over_r)
        + np.cos(lat_rad) * np.sin(d_over_r) * np.cos(bearings)
    )
    p_lon = lon_rad + np.arctan2(
        np.sin(bearings) * np.sin(d_over_r) * np.cos(lat_rad),
        np.cos(d_over_r) - np.sin(lat_rad) * np.sin(p_lat),
    )
    return np.degrees(p_lat), np.degrees(p_lon)


def make_site_key(region: str, site: str) -> str:
    return f"{region}{SITE_KEY_SEP}{site}"


def split_site_key(key: str) -> tuple[str, str]:
    parts = str(key).split(SITE_KEY_SEP, 1)
    return (parts[0], parts[1]) if len(parts) == 2 else ("", str(key))


def point_in_polygon(lat: float, lon: float, coords: list) -> bool:
    """
    Ray-casting point-in-polygon test.
    coords: GeoJSON ring in [[lon, lat], ...] order (closed — first == last).
    """
    inside = False
    n = len(coords)
    j = n - 1
    for i in range(n):
        xi, yi = coords[i][0], coords[i][1]   # lon, lat
        xj, yj = coords[j][0], coords[j][1]
        if ((yi > lat) != (yj > lat)) and \
           (lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi):
            inside = not inside
        j = i
    return inside
