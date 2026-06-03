"""
RF Cluster Analysis Engine
==========================
Analyses a set of filtered cells and produces ranked RF recommendations
following the Orange tiered action ladder:

  Priority  Action type                   Est. recovery
  ────────  ──────────────────────────    ─────────────
  SHORT     freq_balance                  40 %
  SHORT     tilt_down (TA too high)       15 %
  SHORT     tilt_up   (TA too low)        15 %
  SHORT     intra_site_neighbor_tilt      35 %
  SHORT     inter_site_neighbor_tilt      35 %
  MEDIUM    add_band                      50 %
  LONG      add_sector                    85 %
  LONG      new_site                      90 %

Public API
----------
analyse_cluster(cell_list, df, cell_geo_meta, cell_polygons,
                neighbor_radius_m=2000) -> dict
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Thresholds ────────────────────────────────────────────────────────────────
_PRB_CONGESTED   = 80.0   # % — cell is considered congested
_PRB_P0          = 90.0   # % — severe
_PRB_HEALTHY     = 65.0   # % — safe to accept offload
_PRB_IMBALANCE   = 15.0   # pp — min gap to flag freq imbalance

# TA thresholds (metres)
_TA_LOW          = 60.0   # m  — users very close → recommend uptilt
_TA_HIGH         = 180.0  # m  — users far away   → recommend downtilt

# Known frequency bands (MHz or string key)
_ALL_BANDS = ["800", "1800", "2100", "tdd"]

# Azimuth tolerance to group cells into the same "sector direction"
_SECTOR_AZ_TOL   = 30.0   # degrees
# Azimuth range considered "opposite"
_OPPOSITE_MIN    = 150.0
_OPPOSITE_MAX    = 210.0  # 180 ± 30
# Adjacent sector range
_ADJACENT_MIN    = 50.0
_ADJACENT_MAX    = 130.0

# ── Geometry helpers ──────────────────────────────────────────────────────────

def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres."""
    R = 6_371_000.0
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p)
         * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing from point 1 → point 2, in degrees [0, 360)."""
    p = math.pi / 180
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    x = math.sin(dlon) * math.cos(lat2 * p)
    y = (math.cos(lat1 * p) * math.sin(lat2 * p)
         - math.sin(lat1 * p) * math.cos(lat2 * p) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _az_diff(a1, a2) -> float:
    """Smallest absolute angular difference between two azimuths [0, 180]."""
    d = abs((a1 - a2 + 360) % 360)
    return d if d <= 180 else 360 - d


# ── Step 1: Build cluster topology ───────────────────────────────────────────

def _build_topology(cell_list: list[str], cell_geo_meta: dict) -> dict:
    """
    Returns
    -------
    topology : {
        site_key: {
            "site": str,
            "region": str,
            "lat": float, "lon": float,
            "sectors": {
                sector_id: {
                    "azimuth": float,           # representative azimuth
                    "cells": {
                        cell_name: {
                            "frequency": str,
                            "azimuth": float,
                            "lat": float, "lon": float,
                            "range_m": float, "ta_m": float|None,
                        }
                    }
                }
            }
        }
    }
    """
    # Group cells by site
    by_site: dict[str, list[str]] = defaultdict(list)
    for cn in cell_list:
        if cn in cell_geo_meta:
            by_site[cell_geo_meta[cn]["site_key"]].append(cn)

    topology = {}
    for sk, cells in by_site.items():
        meta0 = cell_geo_meta[cells[0]]
        site_entry = {
            "site":    meta0["site"],
            "region":  meta0["region"],
            "lat":     float(np.mean([cell_geo_meta[c]["lat"] for c in cells])),
            "lon":     float(np.mean([cell_geo_meta[c]["lon"] for c in cells])),
            "sectors": {},
        }

        # Bucket cells into sectors by azimuth similarity
        assigned: dict[str, int] = {}   # cell_name → sector_id
        sector_azimuths: list[float] = []

        for cn in sorted(cells):
            az = cell_geo_meta[cn]["azimuth"]
            matched = None
            for sid, saz in enumerate(sector_azimuths):
                if _az_diff(az, saz) <= _SECTOR_AZ_TOL:
                    matched = sid
                    break
            if matched is None:
                matched = len(sector_azimuths)
                sector_azimuths.append(az)
            assigned[cn] = matched

        # Build sector dicts
        for cn, sid in assigned.items():
            m = cell_geo_meta[cn]
            if sid not in site_entry["sectors"]:
                site_entry["sectors"][sid] = {
                    "azimuth": sector_azimuths[sid],
                    "cells":   {},
                }
            site_entry["sectors"][sid]["cells"][cn] = {
                "frequency": str(m.get("frequency", "")).strip(),
                "azimuth":   float(m["azimuth"]),
                "lat":       float(m["lat"]),
                "lon":       float(m["lon"]),
                "range_m":   float(m.get("range_m", 100)),
                "ta_m":      m.get("ta_m"),
            }

        topology[sk] = site_entry

    return topology


# ── Step 2: Compute per-cell KPI stats from DF ────────────────────────────────

def _cell_stats(cell_list: list[str], df: pd.DataFrame) -> dict:
    """
    Returns {cell_name: {avg_prb, max_prb, avg_users, avg_ta, avg_tput,
                          congestion_level, p0_hours, p1_hours}}
    """
    if df.empty or "Cell Name" not in df.columns:
        return {}

    kpi_cols = {
        "DL PRB Load":            "avg_prb",
        "Active Users":           "avg_users",
        "LTE DL User Throughput": "avg_tput",
        "Average Timing Advance": "avg_ta",
    }
    present = {k: v for k, v in kpi_cols.items() if k in df.columns}

    sub = df[df["Cell Name"].isin(cell_list)].copy()
    if sub.empty:
        return {}

    agg_map = {k: "mean" for k in present}
    if "DL PRB Load" in df.columns:
        agg_map["DL PRB Load"] = ["mean", "max"]

    result = {}
    for cn, grp in sub.groupby("Cell Name"):
        entry = {}
        for col, alias in present.items():
            entry[alias] = float(grp[col].dropna().mean()) if not grp[col].dropna().empty else None

        if "DL PRB Load" in grp.columns:
            entry["max_prb"] = float(grp["DL PRB Load"].dropna().max()) \
                               if not grp["DL PRB Load"].dropna().empty else None
        else:
            entry["max_prb"] = None

        # Congestion level
        avg_prb = entry.get("avg_prb")
        if avg_prb is None:
            entry["congestion_level"] = "no_data"
        elif avg_prb >= _PRB_P0:
            entry["congestion_level"] = "P0"
        elif avg_prb >= _PRB_CONGESTED:
            entry["congestion_level"] = "P1"
        else:
            entry["congestion_level"] = "healthy"

        # Hour counts
        if "P0_Congestion" in grp.columns:
            entry["p0_hours"] = int(grp["P0_Congestion"].fillna(False).sum())
        else:
            entry["p0_hours"] = 0
        if "P1_Congestion" in grp.columns:
            entry["p1_hours"] = int(grp["P1_Congestion"].fillna(False).sum())
        else:
            entry["p1_hours"] = 0

        result[str(cn)] = entry

    return result


# ── Analysis engines ──────────────────────────────────────────────────────────

def _engine_freq_balance(topology, stats) -> list[dict]:
    """Engine A — frequency-band imbalance within a sector."""
    recs = []
    for sk, site in topology.items():
        for sid, sector in site["sectors"].items():
            cells = sector["cells"]
            if len(cells) < 2:
                continue

            # Get avg PRB for each frequency present
            freq_prb: dict[str, tuple[str, float]] = {}   # freq → (cell_name, avg_prb)
            for cn, info in cells.items():
                s = stats.get(cn, {})
                ap = s.get("avg_prb")
                if ap is not None:
                    freq = info["frequency"] or "?"
                    freq_prb[freq] = (cn, ap)

            if len(freq_prb) < 2:
                continue

            items = sorted(freq_prb.items(), key=lambda x: x[1][1], reverse=True)
            hi_freq, (hi_cn, hi_prb) = items[0]
            lo_freq, (lo_cn, lo_prb) = items[-1]

            if hi_prb < _PRB_CONGESTED:
                continue
            if hi_prb - lo_prb < _PRB_IMBALANCE:
                continue

            est_after = hi_prb - (hi_prb - lo_prb) * 0.40
            recs.append({
                "priority":       "short",
                "action_type":    "freq_balance",
                "site":           site["site"],
                "sector_az":      round(sector["azimuth"], 1),
                "affected_cells": [hi_cn],
                "helper_cells":   [lo_cn],
                "rationale": (
                    f"Sector {sector['azimuth']:.0f}° on {site['site']}: "
                    f"band {hi_freq} is at {hi_prb:.1f}% PRB while {lo_freq} "
                    f"sits at {lo_prb:.1f}%. Balancing load between carriers "
                    f"by adjusting A3 offset / reselection priority could cut "
                    f"{hi_freq} PRB to ~{est_after:.0f}%."
                ),
                "evidence": {
                    "high_band": hi_freq, "high_prb": round(hi_prb, 1),
                    "low_band":  lo_freq, "low_prb":  round(lo_prb, 1),
                },
                "recovery_pct":        40,
                "estimated_prb_after": round(est_after, 1),
                "suggested_delta":     "Adjust HO/reselection A3 offset between bands",
            })
    return recs


def _engine_ta_tilt(topology, stats) -> list[dict]:
    """Engine B — TA-based downtilt / uptilt recommendations."""
    recs = []
    for sk, site in topology.items():
        for sid, sector in site["sectors"].items():
            for cn, info in sector["cells"].items():
                s = stats.get(cn, {})
                ap  = s.get("avg_prb")
                ta  = s.get("avg_ta")
                if ap is None or ap < _PRB_CONGESTED:
                    continue

                if ta is None:
                    continue   # no TA data — skip

                if ta <= _TA_LOW:
                    # Users are bunched close → uptilt to shrink footprint,
                    # push edge users to neighbours
                    est_after = ap * 0.85
                    recs.append({
                        "priority":       "short",
                        "action_type":    "tilt_up",
                        "site":           site["site"],
                        "sector_az":      round(sector["azimuth"], 1),
                        "affected_cells": [cn],
                        "helper_cells":   [],
                        "rationale": (
                            f"{cn} ({site['site']}, {sector['azimuth']:.0f}°): "
                            f"PRB={ap:.1f}% but avg TA only {ta:.0f} m — users "
                            f"are concentrated very close to the antenna. "
                            f"Uptilting by 2–4° will tighten the beam, reduce "
                            f"the near-field footprint and push edge users to "
                            f"adjacent/neighbouring sectors."
                        ),
                        "evidence": {
                            "avg_prb": round(ap, 1),
                            "avg_ta_m": round(ta, 1),
                            "ta_threshold_m": _TA_LOW,
                        },
                        "recovery_pct":        15,
                        "estimated_prb_after": round(est_after, 1),
                        "suggested_delta":     "Uptilt 2–4° (verify with RF planning tool)",
                    })

                elif ta >= _TA_HIGH:
                    # Cell reaching far → downtilt to reduce range
                    est_after = ap * 0.85
                    recs.append({
                        "priority":       "short",
                        "action_type":    "tilt_down",
                        "site":           site["site"],
                        "sector_az":      round(sector["azimuth"], 1),
                        "affected_cells": [cn],
                        "helper_cells":   [],
                        "rationale": (
                            f"{cn} ({site['site']}, {sector['azimuth']:.0f}°): "
                            f"PRB={ap:.1f}% and avg TA is {ta:.0f} m — the cell "
                            f"is pulling users from far away, competing with "
                            f"neighbouring cells. Downtilting by 2–5° will shrink "
                            f"the coverage footprint and reduce inter-site "
                            f"interference."
                        ),
                        "evidence": {
                            "avg_prb": round(ap, 1),
                            "avg_ta_m": round(ta, 1),
                            "ta_threshold_m": _TA_HIGH,
                        },
                        "recovery_pct":        15,
                        "estimated_prb_after": round(est_after, 1),
                        "suggested_delta":     "Downtilt 2–5° (verify with RF planning tool)",
                    })
    return recs


def _engine_intra_site(topology, stats) -> list[dict]:
    """Engine C — intra-site opposite/adjacent sector relief."""
    recs = []
    for sk, site in topology.items():
        sector_ids = list(site["sectors"].keys())
        if len(sector_ids) < 2:
            continue

        for sid in sector_ids:
            sector = site["sectors"][sid]
            # Check if this sector is congested
            cong_cells = [
                cn for cn in sector["cells"]
                if stats.get(cn, {}).get("congestion_level") in ("P0", "P1")
            ]
            if not cong_cells:
                continue

            # Average PRB of the congested sector
            cong_prb = float(np.mean([
                stats[cn]["avg_prb"] for cn in cong_cells
                if stats.get(cn, {}).get("avg_prb") is not None
            ])) if cong_cells else 0.0

            for other_sid in sector_ids:
                if other_sid == sid:
                    continue
                other_sector = site["sectors"][other_sid]
                diff = _az_diff(sector["azimuth"], other_sector["azimuth"])

                # Classify relationship
                if _OPPOSITE_MIN <= diff <= _OPPOSITE_MAX:
                    relationship = "opposite"
                    tilt_suggestion = "Uptilt other sector 2–4° to widen its footprint"
                elif _ADJACENT_MIN <= diff <= _ADJACENT_MAX:
                    relationship = "adjacent"
                    tilt_suggestion = "Uptilt adjacent sector 1–3° to absorb overflow"
                else:
                    continue

                # Check if helper sector is healthy enough
                helper_cells = list(other_sector["cells"].keys())
                healthy_helpers = [
                    cn for cn in helper_cells
                    if stats.get(cn, {}).get("avg_prb") is not None
                    and stats[cn]["avg_prb"] < _PRB_HEALTHY
                ]
                if not healthy_helpers:
                    continue

                helper_prb = float(np.mean([
                    stats[cn]["avg_prb"] for cn in healthy_helpers
                    if stats.get(cn, {}).get("avg_prb") is not None
                ]))

                # Expected absorption headroom
                headroom = _PRB_HEALTHY - helper_prb
                est_after = max(cong_prb - headroom * 0.35, cong_prb * 0.80)

                recs.append({
                    "priority":       "short",
                    "action_type":    "intra_site_neighbor_tilt",
                    "site":           site["site"],
                    "sector_az":      round(sector["azimuth"], 1),
                    "affected_cells": cong_cells,
                    "helper_cells":   healthy_helpers,
                    "rationale": (
                        f"{site['site']}: {relationship} sector at "
                        f"{other_sector['azimuth']:.0f}° is healthy "
                        f"({helper_prb:.1f}% PRB, {headroom:.0f} pp headroom). "
                        f"Adjusting its tilt can absorb overflow from the "
                        f"congested {sector['azimuth']:.0f}° sector "
                        f"({cong_prb:.1f}% PRB)."
                    ),
                    "evidence": {
                        "congested_sector_az": round(sector["azimuth"], 1),
                        "congested_prb":       round(cong_prb, 1),
                        "helper_sector_az":    round(other_sector["azimuth"], 1),
                        "helper_prb":          round(helper_prb, 1),
                        "headroom_pp":         round(headroom, 1),
                        "relationship":        relationship,
                    },
                    "recovery_pct":        35,
                    "estimated_prb_after": round(est_after, 1),
                    "suggested_delta":     tilt_suggestion,
                })
    return recs


def _engine_inter_site(topology, stats, cell_geo_meta, neighbor_radius_m) -> list[dict]:
    """Engine D — inter-site neighbor sector relief."""
    # Build a flat list of all sectors in the cluster
    all_sectors = []   # (sk, sid, az, lat, lon, [cell_names])
    for sk, site in topology.items():
        for sid, sector in site["sectors"].items():
            cell_names = list(sector["cells"].keys())
            lats = [sector["cells"][cn]["lat"] for cn in cell_names]
            lons = [sector["cells"][cn]["lon"] for cn in cell_names]
            all_sectors.append({
                "sk": sk, "sid": sid,
                "site": site["site"],
                "az":   sector["azimuth"],
                "lat":  float(np.mean(lats)),
                "lon":  float(np.mean(lons)),
                "cells": cell_names,
            })

    recs = []
    for cong_sec in all_sectors:
        # Congested cells in this sector
        cong_cells = [
            cn for cn in cong_sec["cells"]
            if stats.get(cn, {}).get("congestion_level") in ("P0", "P1")
        ]
        if not cong_cells:
            continue

        cong_prb = float(np.mean([
            stats[cn]["avg_prb"] for cn in cong_cells
            if stats.get(cn, {}).get("avg_prb") is not None
        ]))

        for helper_sec in all_sectors:
            if helper_sec["sk"] == cong_sec["sk"]:
                continue   # same site handled by engine C

            # Distance check
            dist = _haversine_m(
                cong_sec["lat"], cong_sec["lon"],
                helper_sec["lat"], helper_sec["lon"],
            )
            if dist > neighbor_radius_m:
                continue

            # Bearing from helper → congested sector
            bearing = _bearing_deg(
                helper_sec["lat"], helper_sec["lon"],
                cong_sec["lat"], cong_sec["lon"],
            )
            # Does the helper sector point toward the congested sector?
            az_to_target = _az_diff(helper_sec["az"], bearing)
            if az_to_target > _SECTOR_AZ_TOL * 2:
                continue

            # Check if helper sector has capacity
            healthy_helpers = [
                cn for cn in helper_sec["cells"]
                if stats.get(cn, {}).get("avg_prb") is not None
                and stats[cn]["avg_prb"] < _PRB_HEALTHY
            ]
            if not healthy_helpers:
                continue

            helper_prb = float(np.mean([
                stats[cn]["avg_prb"] for cn in healthy_helpers
            ]))
            headroom = _PRB_HEALTHY - helper_prb
            est_after = max(cong_prb - headroom * 0.30, cong_prb * 0.80)

            recs.append({
                "priority":       "short",
                "action_type":    "inter_site_neighbor_tilt",
                "site":           cong_sec["site"],
                "sector_az":      round(cong_sec["az"], 1),
                "affected_cells": cong_cells,
                "helper_cells":   healthy_helpers,
                "rationale": (
                    f"Neighbor site {helper_sec['site']} (sector "
                    f"{helper_sec['az']:.0f}°) faces the congested sector on "
                    f"{cong_sec['site']} ({cong_sec['az']:.0f}°) at "
                    f"{dist:.0f} m. That sector has {helper_prb:.1f}% PRB "
                    f"({headroom:.0f} pp headroom). Uptilting it 2–4° will "
                    f"widen its footprint to absorb overflow users."
                ),
                "evidence": {
                    "congested_site":  cong_sec["site"],
                    "congested_az":    round(cong_sec["az"], 1),
                    "congested_prb":   round(cong_prb, 1),
                    "helper_site":     helper_sec["site"],
                    "helper_az":       round(helper_sec["az"], 1),
                    "helper_prb":      round(helper_prb, 1),
                    "distance_m":      round(dist, 0),
                    "az_alignment_deg": round(az_to_target, 1),
                },
                "recovery_pct":        35,
                "estimated_prb_after": round(est_after, 1),
                "suggested_delta":     "Uptilt neighboring sector 2–4°",
            })

    return recs


def _engine_add_band(topology, stats) -> list[dict]:
    """Engine E — missing frequency band."""
    recs = []
    for sk, site in topology.items():
        for sid, sector in site["sectors"].items():
            cells = sector["cells"]
            present_bands = {info["frequency"].lower() for info in cells.values()
                             if info["frequency"]}
            # Congested?
            any_cong = any(
                stats.get(cn, {}).get("congestion_level") in ("P0", "P1")
                for cn in cells
            )
            if not any_cong:
                continue

            # What is missing?
            for band in _ALL_BANDS:
                if band not in present_bands:
                    recs.append({
                        "priority":       "medium",
                        "action_type":    "add_band",
                        "site":           site["site"],
                        "sector_az":      round(sector["azimuth"], 1),
                        "affected_cells": list(cells.keys()),
                        "helper_cells":   [],
                        "rationale": (
                            f"Sector {sector['azimuth']:.0f}° on {site['site']} "
                            f"is congested and currently carries "
                            f"{', '.join(sorted(present_bands)) or 'unknown bands'}. "
                            f"Adding the {band.upper()} carrier would provide "
                            f"~50% additional capacity. "
                            f"{'Lower frequencies (800 MHz) also improve indoor coverage.' if band == '800' else ''}"
                            f"{'1800 MHz is a good capacity layer with balanced range.' if band == '1800' else ''}"
                            f"{'2100 MHz adds high-capacity short-range coverage.' if band == '2100' else ''}"
                            f"{'TDD (2600 MHz) adds dense urban capacity.' if band == 'tdd' else ''}"
                        ),
                        "evidence": {
                            "present_bands": sorted(present_bands),
                            "missing_band":  band,
                        },
                        "recovery_pct":        50,
                        "estimated_prb_after": None,
                        "suggested_delta":     f"Deploy {band.upper()} carrier on this sector",
                    })
    return recs


def _engine_add_sector(topology, stats) -> list[dict]:
    """Engine F — add a 4th sector where the site has 3 fully-congested sectors."""
    recs = []
    for sk, site in topology.items():
        sectors = site["sectors"]
        if len(sectors) >= 4:
            continue   # already has 4+ sectors

        # How many sectors are congested?
        cong_sector_ids = []
        for sid, sector in sectors.items():
            cong_cells = [
                cn for cn in sector["cells"]
                if stats.get(cn, {}).get("congestion_level") in ("P0", "P1")
            ]
            if cong_cells:
                cong_sector_ids.append(sid)

        if len(cong_sector_ids) < len(sectors):
            continue   # not all sectors congested → short-term actions first

        # Find largest azimuth gap to recommend direction for new sector
        azimuths = sorted([sectors[sid]["azimuth"] for sid in sectors])
        gaps = []
        for i in range(len(azimuths)):
            next_az = azimuths[(i + 1) % len(azimuths)]
            gap = (next_az - azimuths[i] + 360) % 360
            mid = (azimuths[i] + gap / 2) % 360
            gaps.append((gap, mid))
        gaps.sort(reverse=True)
        best_gap, best_az = gaps[0]

        all_cong_cells = [
            cn for sid in cong_sector_ids
            for cn in sectors[sid]["cells"]
        ]
        recs.append({
            "priority":       "long",
            "action_type":    "add_sector",
            "site":           site["site"],
            "sector_az":      round(best_az, 1),
            "affected_cells": all_cong_cells,
            "helper_cells":   [],
            "rationale": (
                f"All {len(sectors)} sectors on {site['site']} are congested "
                f"and short-term actions are insufficient. Adding a "
                f"{len(sectors)+1}th sector at ≈{best_az:.0f}° would fill the "
                f"largest azimuth gap ({best_gap:.0f}°) and provide ~85% "
                f"congestion relief. Requires new antenna installation."
            ),
            "evidence": {
                "n_sectors":          len(sectors),
                "congested_sectors":  len(cong_sector_ids),
                "recommended_az_deg": round(best_az, 1),
                "largest_gap_deg":    round(best_gap, 1),
            },
            "recovery_pct":        85,
            "estimated_prb_after": None,
            "suggested_delta":     f"Install new sector antenna at ≈{best_az:.0f}°",
        })
    return recs


def _engine_new_site(topology, stats, cell_geo_meta) -> list[dict]:
    """Engine G — suggest a new site when no other remedy exists."""
    recs = []
    for sk, site in topology.items():
        sectors = site["sectors"]

        # All sectors and frequencies congested?
        all_cong = all(
            stats.get(cn, {}).get("congestion_level") in ("P0", "P1")
            for sector in sectors.values()
            for cn in sector["cells"]
            if stats.get(cn, {}).get("congestion_level") != "no_data"
        )
        if not all_cong:
            continue

        # Suggest location: offset from site centroid toward highest TA direction
        # (where the most distant users are)
        best_az   = 0.0
        best_ta   = -1.0
        for sector in sectors.values():
            for cn in sector["cells"]:
                ta = stats.get(cn, {}).get("avg_ta")
                if ta and ta > best_ta:
                    best_ta = ta
                    best_az = sector["azimuth"]

        # Offset ~500 m in the direction of heaviest distant traffic
        R = 6_371_000.0
        offset_m   = 500.0
        az_rad     = math.radians(best_az)
        dlat       = (offset_m / R) * math.cos(az_rad) * (180 / math.pi)
        dlon       = (offset_m / (R * math.cos(math.radians(site["lat"])))) \
                     * math.sin(az_rad) * (180 / math.pi)
        sug_lat    = round(site["lat"] + dlat, 6)
        sug_lon    = round(site["lon"] + dlon, 6)

        all_cong_cells = [
            cn for sector in sectors.values()
            for cn in sector["cells"]
        ]
        recs.append({
            "priority":       "long",
            "action_type":    "new_site",
            "site":           site["site"],
            "sector_az":      None,
            "affected_cells": all_cong_cells,
            "helper_cells":   [],
            "rationale": (
                f"Every sector and frequency carrier on {site['site']} is "
                f"congested and no neighbouring sector has sufficient headroom "
                f"for offload. A new capacitive site is required. "
                f"Suggested location: ≈{sug_lat:.5f}, {sug_lon:.5f} "
                f"(~500 m at {best_az:.0f}° from {site['site']}, toward the "
                f"area of highest average TA = {best_ta:.0f} m). "
                f"Coordinate full site acquisition and planning process."
            ),
            "evidence": {
                "site_lat":           site["lat"],
                "site_lon":           site["lon"],
                "suggested_lat":      sug_lat,
                "suggested_lon":      sug_lon,
                "offset_direction_deg": round(best_az, 1),
                "max_ta_m":           round(best_ta, 1) if best_ta > 0 else None,
            },
            "recovery_pct":        90,
            "estimated_prb_after": None,
            "suggested_delta":     "New site deployment (full planning process required)",
        })
    return recs


# ── Cluster-level summary ─────────────────────────────────────────────────────

def _cluster_summary(topology, stats) -> dict:
    n_sites = len(topology)
    n_sectors = sum(len(s["sectors"]) for s in topology.values())
    n_cells = sum(
        len(sec["cells"])
        for s in topology.values()
        for sec in s["sectors"].values()
    )
    freq_counts: dict[str, int] = defaultdict(int)
    for site in topology.values():
        for sector in site["sectors"].values():
            for cn, info in sector["cells"].items():
                f = info["frequency"].lower() if info["frequency"] else "unknown"
                freq_counts[f] += 1

    level_counts = {"P0": 0, "P1": 0, "healthy": 0, "no_data": 0}
    for s in stats.values():
        lv = s.get("congestion_level", "no_data")
        level_counts[lv] = level_counts.get(lv, 0) + 1

    return {
        "n_sites":    n_sites,
        "n_sectors":  n_sectors,
        "n_cells":    n_cells,
        "freq_counts": dict(freq_counts),
        "p0_cells":   level_counts["P0"],
        "p1_cells":   level_counts["P1"],
        "healthy_cells": level_counts["healthy"],
        "no_data_cells": level_counts["no_data"],
    }


# ── Deduplication ─────────────────────────────────────────────────────────────

def _deduplicate(recs: list[dict]) -> list[dict]:
    """
    Remove redundant recommendations:
    - For the same (site, sector_az, action_type), keep only the strongest.
    - For inter_site_neighbor_tilt, a helper cell should appear at most once
      as the recommended adjustment.
    """
    seen_keys: set[tuple] = set()
    seen_helpers: set[str] = set()
    out = []
    for r in recs:
        key = (r["action_type"], r["site"], r.get("sector_az"))
        if key in seen_keys:
            continue
        # Avoid recommending the same helper cell in multiple inter-site recs
        if r["action_type"] == "inter_site_neighbor_tilt":
            helpers = tuple(sorted(r["helper_cells"]))
            if helpers and helpers in seen_helpers:
                continue
            seen_helpers.add(helpers)
        seen_keys.add(key)
        out.append(r)
    return out


# ── Public entry point ────────────────────────────────────────────────────────

_PRIORITY_ORDER = {"short": 0, "medium": 1, "long": 2}
_TYPE_ORDER = {
    "freq_balance":              0,
    "tilt_down":                 1,
    "tilt_up":                   2,
    "intra_site_neighbor_tilt":  3,
    "inter_site_neighbor_tilt":  4,
    "add_band":                  5,
    "add_sector":                6,
    "new_site":                  7,
}


def analyse_cluster(
    cell_list:        list[str],
    df:               pd.DataFrame,
    cell_geo_meta:    dict,
    cell_polygons:    dict,
    neighbor_radius_m: float = 2000.0,
) -> dict:
    """
    Run all analysis engines and return a result dict:
    {
        "summary":       dict,
        "topology":      dict,
        "stats":         dict,
        "recommendations": [list of rec dicts, ranked],
        "n_recs":        int,
        "warning":       str | None,
    }
    """
    warning = None

    # Filter to cells that actually have geo data
    valid = [cn for cn in cell_list if cn in cell_geo_meta]
    if not valid:
        return {
            "summary": {}, "topology": {}, "stats": {},
            "recommendations": [], "n_recs": 0,
            "warning": "No geo data available for any filtered cell.",
        }

    # Cap for performance
    if len(valid) > 50000:
        warning = (
            f"Filter contains {len(valid)} cells — analysis capped at the "
            f"top 500 by congestion level. Refine the filter for full detail."
        )
        # Sort by PRB desc using DF stats preview
        if not df.empty and "DL PRB Load" in df.columns and "Cell Name" in df.columns:
            prb_max = (
                df[df["Cell Name"].isin(valid)]
                .groupby("Cell Name")["DL PRB Load"].max()
                .reindex(valid).fillna(0)
                .sort_values(ascending=False)
            )
            valid = prb_max.index.tolist()[:500]
        else:
            valid = valid[:500]

    topology = _build_topology(valid, cell_geo_meta)
    stats    = _cell_stats(valid, df)

    # Run all engines
    recs: list[dict] = []
    recs += _engine_freq_balance(topology, stats)
    recs += _engine_ta_tilt(topology, stats)
    recs += _engine_intra_site(topology, stats)
    recs += _engine_inter_site(topology, stats, cell_geo_meta, neighbor_radius_m)
    recs += _engine_add_band(topology, stats)
    recs += _engine_add_sector(topology, stats)
    recs += _engine_new_site(topology, stats, cell_geo_meta)

    recs = _deduplicate(recs)

    # Sort: priority → action_type_order → recovery_pct desc
    recs.sort(key=lambda r: (
        _PRIORITY_ORDER.get(r["priority"], 9),
        _TYPE_ORDER.get(r["action_type"], 9),
        -r["recovery_pct"],
    ))

    summary = _cluster_summary(topology, stats)

    return {
        "summary":         summary,
        "topology":        topology,
        "stats":           stats,
        "recommendations": recs,
        "n_recs":          len(recs),
        "warning":         warning,
    }
