"""
Cell adjacency graph builder
============================

Reads the cell-configuration CSV (Cell Name, Longitude_Sector, Latitude_Sector,
Azimuth, expected_cell_range, frequency, ...) and produces:

  1. cell_neighbors.csv         one row per (cell A, cell B) pair within
                                MAX_SEARCH_DIST_M, with geometric features
                                + a categorical 'relation' label.
  2. cell_neighbor_summary.csv  one row per cell with counts of each
                                relation type and number of distinct
                                neighbor sites.

These two files are the spatial backbone for the recommendation engine.

Tunable parameters at the top:
  MAX_SEARCH_DIST_M  — how far (in meters) to look for neighbors.
  BEAMWIDTH_DEG      — assumed antenna 3 dB beamwidth (typical sector: 65°).

Notes on edge cases:
  * Co-located sectors (same Radio Site) share lat/lon, so the bearing
    columns are meaningless for those rows — read 'relation' instead
    (it picks them up via the Radio Site key).
  * The script auto-derives 'Radio Site' from Cell Name if not present
    (first two underscore-tokens, e.g. ARI_0003).
  * If 'frequency' column is missing, it falls back to suffix-letter
    parsing (h→1800, l→2100, f/j→800, g→TDD).

Usage:
    python build_cell_neighbors.py \
        --in_path final_info_data.csv \
        --out_pairs cell_neighbors.csv \
        --out_summary cell_neighbor_summary.csv \
        --max_dist_m 5000 \
        --beamwidth_deg 65
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_R_M = 6_371_000.0

# --- relation taxonomy used in the output ---
# co_site_same_freq   same Radio Site, same frequency  → different sectors of the same band (most useful for tilt/LB inside the site)
# co_site_diff_freq   same Radio Site, different bands → different layers on the (likely) same sector position
# facing_neighbor     different sites, both antennas point at each other (overlap + interference candidate)
# one_way_facing      different sites, only one points at the other (asymmetric overlap)
# overlap_no_facing   different sites, ranges overlap but neither faces the other
# nearby_unrelated    within search radius but no overlap and no facing (weakest tie)
RELATION_ORDER = [
    "co_site_same_freq",
    "co_site_diff_freq",
    "facing_neighbor",
    "one_way_facing",
    "overlap_no_facing",
    "nearby_unrelated",
]


def derive_radio_site(cell_name_series: pd.Series) -> pd.Series:
    return cell_name_series.str.split("_").str[:2].str.join("_")


def derive_frequency(cell_name_series: pd.Series) -> pd.Series:
    suffix = cell_name_series.str.extract(r"_([^_]+)$")[0].str.lower()
    first_letter = (
        suffix.str.replace(r"^\d+", "", regex=True)
              .str.extract(r"^([a-z])")[0]
    )
    freq_map = {"h": "1800", "l": "2100", "f": "800", "j": "800", "g": "TDD"}
    return first_letter.map(freq_map).fillna("Unknown")


def angular_diff_deg(u, v):
    """Smallest unsigned angle between two compass directions (degrees)."""
    d = (u - v) % 360
    return np.where(d > 180, 360 - d, d)


def build_neighbors(in_path, out_pairs, out_summary,
                    max_dist_m=5000.0, beamwidth_deg=65.0):

    cfg = pd.read_csv(in_path)

    # --- engineer Radio Site / frequency if missing ---
    if "Radio Site" not in cfg.columns:
        cfg["Radio Site"] = derive_radio_site(cfg["Cell Name"])
    if "frequency" not in cfg.columns:
        cfg["frequency"] = derive_frequency(cfg["Cell Name"])

    # --- drop rows with missing essentials ---
    needed = [
        "Cell Name", "Latitude_Sector", "Longitude_Sector",
        "Azimuth", "expected_cell_range",
    ]
    before = len(cfg)
    cfg = cfg.dropna(subset=needed).reset_index(drop=True)
    if before != len(cfg):
        print(f"Dropped {before - len(cfg)} rows with missing essentials")

    n = len(cfg)
    print(f"Loaded {n} cells")

    # --- spatial index in radians (haversine wants radians) ---
    coords_rad = np.deg2rad(
        cfg[["Latitude_Sector", "Longitude_Sector"]].values
    )
    tree = BallTree(coords_rad, metric="haversine")

    radius_rad = max_dist_m / EARTH_R_M
    indices, distances_rad = tree.query_radius(
        coords_rad, r=radius_rad, return_distance=True
    )

    # --- build pair arrays (i < j, no self-pairs, no duplicates) ---
    a_idx_chunks, b_idx_chunks, dist_chunks = [], [], []
    for i in range(n):
        neigh = indices[i]
        d = distances_rad[i] * EARTH_R_M
        keep = neigh > i
        if not keep.any():
            continue
        a_idx_chunks.append(np.full(int(keep.sum()), i, dtype=np.int64))
        b_idx_chunks.append(neigh[keep].astype(np.int64))
        dist_chunks.append(d[keep])

    if not a_idx_chunks:
        print("No pairs within search radius. Increase --max_dist_m.")
        return

    a_idx = np.concatenate(a_idx_chunks)
    b_idx = np.concatenate(b_idx_chunks)
    dist_m = np.concatenate(dist_chunks)
    print(f"Found {len(a_idx):,} pairs within {max_dist_m:.0f} m")

    # --- pull arrays in vectorized form ---
    a_lat = cfg["Latitude_Sector"].values[a_idx]
    a_lon = cfg["Longitude_Sector"].values[a_idx]
    b_lat = cfg["Latitude_Sector"].values[b_idx]
    b_lon = cfg["Longitude_Sector"].values[b_idx]
    a_az  = cfg["Azimuth"].values[a_idx].astype(float)
    b_az  = cfg["Azimuth"].values[b_idx].astype(float)
    a_rng = cfg["expected_cell_range"].values[a_idx].astype(float)
    b_rng = cfg["expected_cell_range"].values[b_idx].astype(float)

    # --- bearing A→B (initial bearing on a sphere) ---
    lat1 = np.deg2rad(a_lat)
    lat2 = np.deg2rad(b_lat)
    dlon = np.deg2rad(b_lon - a_lon)
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing_ab = (np.rad2deg(np.arctan2(y, x)) + 360) % 360
    bearing_ba = (bearing_ab + 180) % 360

    az_off_a = angular_diff_deg(a_az, bearing_ab)  # 0 = A points exactly at B
    az_off_b = angular_diff_deg(b_az, bearing_ba)  # 0 = B points exactly at A

    half_bw         = beamwidth_deg / 2.0
    a_faces_b       = az_off_a <= half_bw
    b_faces_a       = az_off_b <= half_bw
    face_each_other = a_faces_b & b_faces_a
    range_overlap   = dist_m <= (a_rng + b_rng)

    same_site = cfg["Radio Site"].values[a_idx] == cfg["Radio Site"].values[b_idx]
    same_freq = cfg["frequency"].values[a_idx]  == cfg["frequency"].values[b_idx]

    # --- categorical relation label (vectorized) ---
    relation = np.select(
        [
            same_site & same_freq,
            same_site & ~same_freq,
            ~same_site & face_each_other,
            ~same_site & (a_faces_b | b_faces_a),
            ~same_site & range_overlap,
        ],
        [
            "co_site_same_freq",
            "co_site_diff_freq",
            "facing_neighbor",
            "one_way_facing",
            "overlap_no_facing",
        ],
        default="nearby_unrelated",
    )

    # --- pairs dataframe ---
    pairs = pd.DataFrame({
        "cell_a":          cfg["Cell Name"].values[a_idx],
        "cell_b":          cfg["Cell Name"].values[b_idx],
        "site_a":          cfg["Radio Site"].values[a_idx],
        "site_b":          cfg["Radio Site"].values[b_idx],
        "freq_a":          cfg["frequency"].values[a_idx],
        "freq_b":          cfg["frequency"].values[b_idx],
        "azimuth_a":       a_az,
        "azimuth_b":       b_az,
        "range_a":         a_rng,
        "range_b":         b_rng,
        "distance_m":      dist_m.round(1),
        "bearing_a_to_b":  bearing_ab.round(1),
        "az_off_a_deg":    az_off_a.round(1),
        "az_off_b_deg":    az_off_b.round(1),
        "a_faces_b":       a_faces_b,
        "b_faces_a":       b_faces_a,
        "face_each_other": face_each_other,
        "range_overlap":   range_overlap,
        "same_site":       same_site,
        "same_freq":       same_freq,
        "relation":        relation,
    })
    pairs.to_csv(out_pairs, index=False)
    print(f"Wrote {len(pairs):,} pair rows -> {out_pairs}")

    # --- per-cell summary (mirror pairs so each cell sees itself as 'cell') ---
    long_a = pairs[["cell_a", "cell_b", "distance_m", "relation"]].rename(
        columns={"cell_a": "cell", "cell_b": "neighbor"})
    long_b = pairs[["cell_b", "cell_a", "distance_m", "relation"]].rename(
        columns={"cell_b": "cell", "cell_a": "neighbor"})
    long_view = pd.concat([long_a, long_b], ignore_index=True)

    counts = (
        long_view.groupby(["cell", "relation"]).size()
                 .unstack(fill_value=0)
                 .reindex(columns=RELATION_ORDER, fill_value=0)
    )
    counts["total_neighbors"] = counts.sum(axis=1)

    # number of distinct neighbor sites (proxy for "how dense is this area")
    site_lookup = cfg.set_index("Cell Name")["Radio Site"]
    long_view["neighbor_site"] = long_view["neighbor"].map(site_lookup)
    distinct_sites = (
        long_view.groupby("cell")["neighbor_site"]
                 .nunique()
                 .rename("distinct_neighbor_sites")
    )

    # nearest neighbor distance per cell
    nearest = long_view.groupby("cell")["distance_m"].min().rename("nearest_neighbor_m")

    summary = (
        counts
        .join(distinct_sites)
        .join(nearest)
        .reset_index()
        .rename(columns={"cell": "Cell Name"})
    )

    # also bring in own site / freq for convenience
    summary = summary.merge(
        cfg[["Cell Name", "Radio Site", "frequency", "Azimuth", "expected_cell_range"]],
        on="Cell Name", how="left",
    )

    summary.to_csv(out_summary, index=False)
    print(f"Wrote per-cell summary ({len(summary)} rows) -> {out_summary}")

    # --- quick visibility into what was produced ---
    print("\nRelation breakdown across all pairs:")
    print(pairs["relation"].value_counts().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_path",       default="datasets/final_info_data.csv")
    p.add_argument("--out_pairs",     default="cell_neighbors.csv")
    p.add_argument("--out_summary",   default="cell_neighbor_summary.csv")
    p.add_argument("--max_dist_m",    type=float, default=2000.0)
    p.add_argument("--beamwidth_deg", type=float, default=65.0)
    args = p.parse_args()
    build_neighbors(
        in_path=args.in_path,
        out_pairs=args.out_pairs,
        out_summary=args.out_summary,
        max_dist_m=args.max_dist_m,
        beamwidth_deg=args.beamwidth_deg,
    )
