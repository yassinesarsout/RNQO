"""
optimizer_core.py — pure-function optimization algorithms.

The three algorithms (BCDSA, greedy descent, genetic algorithm) operate on
in-memory NumPy arrays describing the cluster and produce recommendations
for each cell's Δ℘ (transmit-power reduction) and Δℏ (handover-margin
increase).

This module knows nothing about the dashboard, Dash callbacks, or CSV
files. It is unit-testable and reusable.

Calling convention:

    result = run_optimizer(
        lam=np.array([5, 7, 3, ...]),           # current users per cell
        Lambda=np.array([4, 8, 4, ...]),        # capacity ceilings per cell
        eta=np.array([[0, 0.4, 0.1, ...],       # NxN overlap matrix (fraction)
                      [0.4, 0, 0.3, ...],
                      ...]),
        cell_names=["ARI_0001_C01", ...],       # for the per-cell breakdown
        algorithm="bcdsa",                      # or "greedy" / "ga"
        config={"H": 40.0, "Q_min_db": -3.0},   # optional overrides
    )

    # result is a dict, JSON-serialisable for dcc.Store

Returned dict structure:
    {
        "algorithm":            "bcdsa",
        "initial_congestion":   59.5,
        "final_congestion":     15.4,
        "reduction_pct":        74.1,
        "users_offloaded":      81.9,
        "n_cells":              46,
        "n_cells_changed":      45,
        "n_cells_fixed":        19,
        "runtime_s":            0.51,
        "per_cell": [
            {
                "cell_name":         "ARI_0001_C01",
                "lambda":            5.0,
                "Lambda":            4.0,
                "delta_p_db":        2.0,
                "delta_h_db":        0.0,
                "own_offload":       1.4,
                "handover_offload":  0.3,
                "total_offload":     1.7,
                "effective_after":   3.3,
                "excess_before":     1.0,
                "excess_after":      0.0,
                "status":            "fixed",      # "fixed" | "improved" | "adjusted" | "unchanged"
            },
            ...
        ],
    }
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Default configuration — overridable per-call via `config` argument
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG: dict[str, Any] = {
    # Physics
    "H": 40.0,                  # path-loss exponent (positive for urban)
    "Q_min_db": -3.0,           # minimum allowed SINR drop
    # Decision-variable domain
    "delta_range": (0, 1, 2, 3),
    # Penalty weight (large enough that any violation dominates the cost)
    "phi": 1e6,
    # BCDSA hyperparameters (paper-tuned, validated in offline stage 8)
    "bcdsa_t_initial": 100.0,
    "bcdsa_t_final": 0.1,
    "bcdsa_cooling": 0.9,
    "bcdsa_inner_rho": 5,
    "bcdsa_xi_max": 20,
    # GA hyperparameters
    "ga_pop_size": 50,
    "ga_max_gen": 100,
    "ga_elite_frac": 0.02,
    "ga_mutation_frac": 0.18,
    "ga_patience": 10,
    # Misc
    "random_seed": 42,
    "min_users_to_consider": 0.05,   # cells with fewer users than this are not optimized
}


# ─────────────────────────────────────────────────────────────────────────────
# Objective function — used by all three algorithms
# ─────────────────────────────────────────────────────────────────────────────
def _objective(
    delta_p: np.ndarray,
    delta_h: np.ndarray,
    lam: np.ndarray,
    Lambda: np.ndarray,
    eta: np.ndarray,
    cfg: dict,
) -> float:
    """Penalty-augmented objective Λ̃_Υ. Returns scalar; lower is better."""
    H = cfg["H"]
    phi = cfg["phi"]
    Q_min = cfg["Q_min_db"]

    # Eq. 4 — power-driven offload of each cell's own users
    own_offload = lam * (1.0 - (10.0 ** (-delta_p / H)) ** 2)

    # Eq. 5 — handover-driven offload from cell i to neighbor j when j widens its margin
    fac_h = 1.0 - (10.0 ** (-delta_h / H)) ** 2
    handover_offload = (eta * fac_h[np.newaxis, :]).sum(axis=1) * lam

    effective = lam - own_offload - handover_offload
    congestion = np.maximum(0.0, effective - Lambda).sum()

    # Soft quality penalty — weighted by overlap so small overlaps contribute less
    quality_drop = delta_p[:, np.newaxis] + delta_h[np.newaxis, :]
    excess = np.maximum(0.0, quality_drop - (-Q_min))
    quality_penalty = (phi * excess * eta).sum()

    # Load preservation with relative tolerance
    total_after = effective.sum() + own_offload.sum() + handover_offload.sum()
    rel_err = abs(total_after - lam.sum()) / max(lam.sum(), 1.0)
    load_penalty = phi * max(0.0, rel_err - 1e-4)

    return float(congestion + quality_penalty + load_penalty)


def _pure_congestion(
    delta_p: np.ndarray,
    delta_h: np.ndarray,
    lam: np.ndarray,
    Lambda: np.ndarray,
    eta: np.ndarray,
    cfg: dict,
) -> float:
    """Congestion without any penalty terms — for honest reporting."""
    H = cfg["H"]
    own_offload = lam * (1.0 - (10.0 ** (-delta_p / H)) ** 2)
    fac_h = 1.0 - (10.0 ** (-delta_h / H)) ** 2
    handover_offload = (eta * fac_h[np.newaxis, :]).sum(axis=1) * lam
    return float(np.maximum(0.0, lam - own_offload - handover_offload - Lambda).sum())


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: greedy descent
# ─────────────────────────────────────────────────────────────────────────────
def _run_greedy(
    lam: np.ndarray, Lambda: np.ndarray, eta: np.ndarray, cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(lam)
    dp = np.zeros(N)
    dh = np.zeros(N)
    cur = _objective(dp, dh, lam, Lambda, eta, cfg)
    delta_range = cfg["delta_range"]

    for _outer in range(50):
        best_delta = 0.0
        best_change = None
        for i in range(N):
            # Try changing Δ℘ᵢ
            orig = dp[i]
            for v in delta_range:
                if v == orig:
                    continue
                dp[i] = v
                c = _objective(dp, dh, lam, Lambda, eta, cfg)
                if cur - c > best_delta:
                    best_delta = cur - c
                    best_change = ("p", i, v)
                dp[i] = orig
            # Try changing Δℏᵢ
            orig = dh[i]
            for v in delta_range:
                if v == orig:
                    continue
                dh[i] = v
                c = _objective(dp, dh, lam, Lambda, eta, cfg)
                if cur - c > best_delta:
                    best_delta = cur - c
                    best_change = ("h", i, v)
                dh[i] = orig
        if best_change is None:
            break
        kind, i, v = best_change
        if kind == "p":
            dp[i] = v
        else:
            dh[i] = v
        cur -= best_delta

    return dp, dh


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: BCDSA (block coordinated descent simulated annealing)
# ─────────────────────────────────────────────────────────────────────────────
def _run_bcdsa(
    lam: np.ndarray, Lambda: np.ndarray, eta: np.ndarray, cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg["random_seed"])
    N = len(lam)

    dp = np.zeros(N)
    dh = np.zeros(N)
    best_p, best_h = dp.copy(), dh.copy()
    best_cost = _objective(dp, dh, lam, Lambda, eta, cfg)
    current_cost = best_cost

    optimizing = "p"
    T = cfg["bcdsa_t_initial"]
    K = cfg["bcdsa_inner_rho"] * N
    delta_range = cfg["delta_range"]
    xi_max = cfg["bcdsa_xi_max"]
    T_final = cfg["bcdsa_t_final"]
    cooling = cfg["bcdsa_cooling"]

    while T > T_final:
        xi = 0
        for _k in range(K):
            i = rng.integers(0, N)
            old_p, old_h = dp[i], dh[i]
            if optimizing == "p":
                dp[i] = rng.choice(delta_range)
            else:
                dh[i] = rng.choice(delta_range)

            new_cost = _objective(dp, dh, lam, Lambda, eta, cfg)
            d = new_cost - current_cost
            accept = (d < 0) or (rng.random() < math.exp(-d / max(T, 1e-9)))

            cost_before_step = current_cost
            if accept:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_p, best_h = dp.copy(), dh.copy()
            else:
                dp[i], dh[i] = old_p, old_h

            if current_cost == cost_before_step:
                xi += 1
            else:
                xi = 0
            if xi > xi_max:
                optimizing = "h" if optimizing == "p" else "p"
                xi = 0
        T *= cooling

    return best_p, best_h


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm: genetic algorithm
# ─────────────────────────────────────────────────────────────────────────────
def _run_ga(
    lam: np.ndarray, Lambda: np.ndarray, eta: np.ndarray, cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg["random_seed"])
    N = len(lam)
    pop_size = cfg["ga_pop_size"]
    delta_range = cfg["delta_range"]

    # Sparse initialization — keeps chromosomes feasible (see offline stage 7 fix)
    population = np.zeros((pop_size, 2 * N), dtype=int)
    for i in range(pop_size):
        n_changes = rng.integers(1, max(2, N // 4))
        positions = rng.choice(2 * N, size=n_changes, replace=False)
        for p in positions:
            population[i, p] = rng.choice(delta_range)

    def cost_of(chromo):
        return _objective(chromo[::2], chromo[1::2], lam, Lambda, eta, cfg)

    costs = np.array([cost_of(population[i]) for i in range(pop_size)])
    best_cost = costs.min()
    no_improve = 0
    n_elite = max(1, int(pop_size * cfg["ga_elite_frac"]))
    n_mutate = int(pop_size * cfg["ga_mutation_frac"])
    n_cross = pop_size - n_elite - n_mutate

    for _gen in range(cfg["ga_max_gen"]):
        order = np.argsort(costs)
        population = population[order]
        costs = costs[order]

        new_pop = list(population[:n_elite])

        # Crossover
        cross_parents = population[n_elite:].copy()
        rng.shuffle(cross_parents)
        for k in range(n_cross // 2):
            p1, p2 = cross_parents[2 * k], cross_parents[2 * k + 1]
            cp = rng.integers(1, 2 * N)
            new_pop.append(np.concatenate([p1[:cp], p2[cp:]]))
            new_pop.append(np.concatenate([p2[:cp], p1[cp:]]))
        if len(new_pop) < n_elite + n_cross:
            new_pop.append(cross_parents[0])

        # Mutation
        for _m in range(n_mutate):
            base = population[rng.integers(0, pop_size)].copy()
            n_flips = rng.integers(1, 4)
            positions = rng.choice(2 * N, size=n_flips, replace=False)
            for pos in positions:
                base[pos] = rng.choice(delta_range)
            new_pop.append(base)

        population = np.array(new_pop[:pop_size])
        costs = np.array([cost_of(population[i]) for i in range(pop_size)])

        if best_cost - costs.min() > 1e-3:
            best_cost = costs.min()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > cfg["ga_patience"]:
                break

    best_idx = costs.argmin()
    return population[best_idx, ::2].astype(float), population[best_idx, 1::2].astype(float)


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell breakdown — for the recommendation table in the dashboard
# ─────────────────────────────────────────────────────────────────────────────
def _per_cell_breakdown(
    cell_names: list[str],
    lam: np.ndarray, Lambda: np.ndarray, eta: np.ndarray,
    delta_p: np.ndarray, delta_h: np.ndarray, cfg: dict,
) -> list[dict]:
    H = cfg["H"]
    own_offload = lam * (1.0 - (10.0 ** (-delta_p / H)) ** 2)
    fac_h = 1.0 - (10.0 ** (-delta_h / H)) ** 2
    handover_offload = (eta * fac_h[np.newaxis, :]).sum(axis=1) * lam
    effective_after = lam - own_offload - handover_offload
    excess_before = np.maximum(0.0, lam - Lambda)
    excess_after = np.maximum(0.0, effective_after - Lambda)

    records = []
    for i, name in enumerate(cell_names):
        dp_i, dh_i = float(delta_p[i]), float(delta_h[i])
        changed = (dp_i > 0) or (dh_i > 0)
        was_congested = excess_before[i] > 0.05
        still_congested = excess_after[i] > 0.05

        if not changed:
            status = "unchanged"
        elif was_congested and not still_congested:
            status = "fixed"
        elif was_congested and excess_after[i] < excess_before[i] - 0.05:
            status = "improved"
        elif was_congested:
            status = "no-effect"
        else:
            status = "adjusted"

        records.append({
            "cell_name":        name,
            "lambda":           round(float(lam[i]), 3),
            "Lambda":           round(float(Lambda[i]), 2),
            "delta_p_db":       dp_i,
            "delta_h_db":       dh_i,
            "own_offload":      round(float(own_offload[i]), 3),
            "handover_offload": round(float(handover_offload[i]), 3),
            "total_offload":    round(float(own_offload[i] + handover_offload[i]), 3),
            "effective_after":  round(float(effective_after[i]), 3),
            "excess_before":    round(float(excess_before[i]), 3),
            "excess_after":     round(float(excess_after[i]), 3),
            "status":           status,
        })
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
def run_optimizer(
    lam: np.ndarray,
    Lambda: np.ndarray,
    eta: np.ndarray,
    cell_names: list[str],
    algorithm: str = "bcdsa",
    config: dict | None = None,
) -> dict[str, Any]:
    """
    Run the chosen algorithm and return a JSON-serialisable result dict.

    Parameters
    ----------
    lam : array of shape (N,)
        Current number of active users per cell.
    Lambda : array of shape (N,)
        Capacity ceiling Λᵢ per cell (output of the DL predictor).
    eta : array of shape (N, N)
        Overlap matrix η_{i,j}, expressed as a fraction in [0, 1].
        eta[i, j] = fraction of cell i's footprint that overlaps cell j.
    cell_names : list of str
        Cell names in the same order as the arrays.
    algorithm : str
        One of "bcdsa", "greedy", "ga".
    config : dict, optional
        Overrides for DEFAULT_CONFIG.

    Returns
    -------
    dict — see module docstring.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    lam = np.asarray(lam, dtype=float)
    Lambda = np.asarray(Lambda, dtype=float)
    eta = np.asarray(eta, dtype=float)
    if lam.shape != Lambda.shape:
        raise ValueError(f"lam shape {lam.shape} != Lambda shape {Lambda.shape}")
    if eta.shape != (len(lam), len(lam)):
        raise ValueError(f"eta shape {eta.shape} != ({len(lam)}, {len(lam)})")
    if len(cell_names) != len(lam):
        raise ValueError(f"cell_names length {len(cell_names)} != arrays length {len(lam)}")

    N = len(lam)
    initial_cong = _pure_congestion(np.zeros(N), np.zeros(N), lam, Lambda, eta, cfg)

    # Early-out: nothing to optimize
    if initial_cong < 0.05:
        zeros = np.zeros(N)
        return {
            "algorithm":          algorithm,
            "initial_congestion": round(initial_cong, 3),
            "final_congestion":   round(initial_cong, 3),
            "reduction_pct":      0.0,
            "users_offloaded":    0.0,
            "n_cells":            N,
            "n_cells_changed":    0,
            "n_cells_fixed":      0,
            "runtime_s":          0.0,
            "per_cell":           _per_cell_breakdown(cell_names, lam, Lambda, eta, zeros, zeros, cfg),
            "note":               "No congestion in this snapshot — nothing to optimize.",
        }

    t0 = time.time()
    if algorithm == "bcdsa":
        dp, dh = _run_bcdsa(lam, Lambda, eta, cfg)
    elif algorithm == "greedy":
        dp, dh = _run_greedy(lam, Lambda, eta, cfg)
    elif algorithm == "ga":
        dp, dh = _run_ga(lam, Lambda, eta, cfg)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Use 'bcdsa', 'greedy', or 'ga'.")
    runtime = time.time() - t0

    final_cong = _pure_congestion(dp, dh, lam, Lambda, eta, cfg)
    per_cell = _per_cell_breakdown(cell_names, lam, Lambda, eta, dp, dh, cfg)

    users_offloaded = sum(r["total_offload"] for r in per_cell)
    n_changed = sum(1 for r in per_cell if r["delta_p_db"] > 0 or r["delta_h_db"] > 0)
    n_fixed = sum(1 for r in per_cell if r["status"] == "fixed")

    return {
        "algorithm":          algorithm,
        "initial_congestion": round(initial_cong, 3),
        "final_congestion":   round(final_cong, 3),
        "reduction_pct":      round(100.0 * (1.0 - final_cong / max(initial_cong, 1e-9)), 2),
        "users_offloaded":    round(users_offloaded, 3),
        "n_cells":            N,
        "n_cells_changed":    n_changed,
        "n_cells_fixed":      n_fixed,
        "runtime_s":          round(runtime, 3),
        "per_cell":           per_cell,
    }
