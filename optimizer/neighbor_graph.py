"""
neighbor_graph.py — geometric neighbor-overlap matrix builder.

The optimizer needs an η_{i,j} matrix describing the fraction of cell i's
footprint that overlaps cell j. The dashboard's existing precompute module
already produces sector polygons for every cell as part of the map rendering.
This module reuses that data and computes pairwise overlap matrices on demand.

Usage:

    from optimizer.neighbor_graph import build_eta_matrix

    eta = build_eta_matrix(["ARI_0001_C01", "ARI_0001_C02", ...])
    # eta is a (N, N) numpy array where eta[i, j] is the overlap fraction

The function is cached: identical cell lists return the same matrix without
recomputation. Useful when the engineer optimizes the same cluster multiple
times with different algorithms or modes.

Requirements
------------
- Reads `precompute.CELL_POLYGONS` and `precompute.CELL_GEO_META` from the
  parent dashboard package. If those are unavailable (e.g. when this module is
  used standalone), pass `geo_meta` and `polygons` explicitly to `build_eta_matrix`.
- Uses shapely for polygon intersection. Already a transitive dependency of
  most geo-aware Python stacks; if missing, install with: pip install shapely
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import numpy as np

try:
    from shapely.geometry import Polygon
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False


def _polygon_from_ring(ring: list[list[float]]) -> "Polygon | None":
    """Build a shapely Polygon from a [[lon, lat], ...] GeoJSON ring."""
    if not _HAS_SHAPELY or not ring or len(ring) < 3:
        return None
    try:
        poly = Polygon(ring)
        if not poly.is_valid or poly.is_empty or poly.area <= 0:
            return None
        return poly
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell polygon cache
# Computing the shapely Polygon for every cell at module load is wasteful when
# only a subset gets used in any single optimization run, so we lazy-cache by
# cell name.
# ─────────────────────────────────────────────────────────────────────────────
_polygon_cache: dict[str, "Polygon | None"] = {}


def _get_polygon(cell_name: str, polygons_source: dict) -> "Polygon | None":
    if cell_name in _polygon_cache:
        return _polygon_cache[cell_name]
    ring = polygons_source.get(cell_name)
    poly = _polygon_from_ring(ring) if ring else None
    _polygon_cache[cell_name] = poly
    return poly


# ─────────────────────────────────────────────────────────────────────────────
# Eta-matrix cache — keyed by the sorted tuple of cell names
# ─────────────────────────────────────────────────────────────────────────────
_eta_cache: dict[tuple, np.ndarray] = {}
_ETA_CACHE_MAX = 16


def build_eta_matrix(
    cell_names: list[str],
    polygons_source: dict | None = None,
) -> np.ndarray:
    """
    Build an (N, N) overlap-fraction matrix for the given cell names.

    eta[i, j] = area(cell_i ∩ cell_j) / area(cell_i)
              = fraction of cell i's footprint that overlaps cell j

    Note: this is not symmetric in general — overlap fraction depends on
    which cell's area is in the denominator.

    Parameters
    ----------
    cell_names : list of str
        Cell names in a stable order; rows/columns of the returned matrix
        follow this order.
    polygons_source : dict, optional
        Mapping of {cell_name: [[lon, lat], ...]} GeoJSON rings.
        If None, imports from the dashboard's precompute module.

    Returns
    -------
    np.ndarray of shape (N, N), dtype float64, values in [0, 1].
    """
    cell_names = list(cell_names)
    cache_key = tuple(cell_names)
    if cache_key in _eta_cache:
        return _eta_cache[cache_key]

    if polygons_source is None:
        try:
            from precompute import CELL_POLYGONS as polygons_source
        except ImportError as e:
            raise RuntimeError(
                "Could not import CELL_POLYGONS from precompute. "
                "Pass polygons_source explicitly when running standalone."
            ) from e

    N = len(cell_names)
    eta = np.zeros((N, N), dtype=float)

    if not _HAS_SHAPELY:
        # Graceful degradation — return all-zeros matrix and let the
        # optimizer treat the cluster as having no neighbors. The recommendation
        # quality will be poor but the dashboard won't crash.
        _eta_cache[cache_key] = eta
        return eta

    # Resolve polygons once
    polygons = [_get_polygon(cn, polygons_source) for cn in cell_names]
    areas = np.array([p.area if p else 0.0 for p in polygons])

    # Skip self-pairs and pairs where either polygon is missing or has zero area
    for i in range(N):
        if polygons[i] is None or areas[i] <= 0:
            continue
        for j in range(N):
            if i == j or polygons[j] is None:
                continue
            # Fast bounding-box reject before expensive intersection
            if not polygons[i].intersects(polygons[j]):
                continue
            try:
                intersection = polygons[i].intersection(polygons[j])
                if intersection.is_empty:
                    continue
                eta[i, j] = intersection.area / areas[i]
            except Exception:
                # Shapely occasionally throws on degenerate geometries — skip them
                continue

    # Cache, evict oldest if over limit
    _eta_cache[cache_key] = eta
    if len(_eta_cache) > _ETA_CACHE_MAX:
        _eta_cache.pop(next(iter(_eta_cache)))
    return eta


def neighbor_summary(eta: np.ndarray) -> dict:
    """Diagnostic stats about an overlap matrix — useful for the UI status bar."""
    N = eta.shape[0]
    if N == 0:
        return {"n_cells": 0, "n_pairs": 0, "avg_neighbors": 0.0, "avg_overlap_pct": 0.0}
    nonzero_mask = eta > 0
    n_pairs = int(nonzero_mask.sum())
    nonzero_vals = eta[nonzero_mask]
    return {
        "n_cells":         N,
        "n_pairs":         n_pairs,
        "avg_neighbors":   round(n_pairs / N, 2),
        "avg_overlap_pct": round(100.0 * float(nonzero_vals.mean()), 2) if len(nonzero_vals) else 0.0,
        "max_overlap_pct": round(100.0 * float(nonzero_vals.max()), 2) if len(nonzero_vals) else 0.0,
    }


def clear_caches() -> None:
    """For test cleanup."""
    _polygon_cache.clear()
    _eta_cache.clear()
