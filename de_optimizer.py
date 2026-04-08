"""
de_optimizer.py
---------------
Differential Evolution (DE) optimizer for heliostat field layout.

Replaces the Genetic Algorithm used in:
  Haris et al., Heliyon 9 (2023) e21488

DE strategy: DE/rand/1/bin  (canonical Storn & Price, 1997)
  Mutation:   v = x_r1 + F·(x_r2 − x_r3)
  Crossover:  binomial with rate CR
  Selection:  greedy (child replaces parent if better)

Optimised genes (chromosome):
  [TH, LH, WR, DS]
  Tower Height [50–300 m]
  Heliostat Length [5–20 m]
  Width-to-Length Ratio [1–2]
  Security Distance [0.1–0.5]

Objective: maximise annual mean efficiency while targeting 50 MW output.

No external API calls. Pure NumPy/SciPy.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Tuple
import time

from heliostat_field import FieldParams, radial_staggered_layout, fermat_spiral_layout
from efficiency import annual_mean_efficiency, field_total_power_mw
from solar_geometry import DESIGN_POINTS


# ─── Bounds on the four genes ─────────────────────────────────────────────────
BOUNDS = {
    "tower_height":    (50.0,  300.0),
    "heliostat_length":(5.0,   20.0),
    "width_ratio":     (1.0,   2.0),
    "security_dist":   (0.1,   0.5),
}

LOWER = np.array([v[0] for v in BOUNDS.values()])
UPPER = np.array([v[1] for v in BOUNDS.values()])


# ─── Result container ─────────────────────────────────────────────────────────
@dataclass
class DEResult:
    best_params: FieldParams
    best_efficiency: float               # annual mean efficiency [%]
    best_power_mw: float                 # power at vernal equinox design point
    n_heliostats: int
    convergence_history: List[float] = field(default_factory=list)
    runtime_s: float = 0.0
    layout_type: str = "radial_staggered"


# ─── Fitness function ─────────────────────────────────────────────────────────
TARGET_POWER_MW = 50.0
POWER_PENALTY   = 5.0      # penalty weight for deviation from target power


def _params_from_vector(v: np.ndarray, base_params: FieldParams) -> FieldParams:
    return FieldParams(
        tower_height    = float(v[0]),
        heliostat_length= float(v[1]),
        width_ratio     = float(v[2]),
        security_dist   = float(v[3]),
        min_radius      = base_params.min_radius,
        blocking_factor = base_params.blocking_factor,
        reflectivity    = base_params.reflectivity,
        sunshape_std    = base_params.sunshape_std,
    )


def build_layout(params: FieldParams, layout_type: str) -> np.ndarray:
    if layout_type == "radial_staggered":
        return radial_staggered_layout(params, max_radius=600.0)
    else:
        return fermat_spiral_layout(params, n_heliostats=1300)


def fitness(v: np.ndarray, base_params: FieldParams,
            layout_type: str, design_dni: dict) -> float:
    """
    Fitness = annual_mean_efficiency − POWER_PENALTY·|P − P_target| / P_target
    Higher is better.  Returns −∞ on degenerate layouts.
    """
    params = _params_from_vector(v, base_params)
    try:
        positions = build_layout(params, layout_type)
        if len(positions) < 10:
            return -np.inf

        ann_eff = annual_mean_efficiency(positions, params)

        # Power at vernal equinox
        vernal_day = DESIGN_POINTS["Vernal Equinox"]["day"]
        dni_vernal  = design_dni.get("Vernal Equinox", 858.47)
        power_mw   = field_total_power_mw(positions, params, vernal_day, dni_vernal)

        penalty = POWER_PENALTY * abs(power_mw - TARGET_POWER_MW) / TARGET_POWER_MW
        return ann_eff - penalty

    except Exception:
        return -np.inf


# ─── Differential Evolution ───────────────────────────────────────────────────
def differential_evolution(
        layout_type: str,
        design_dni: dict,
        base_params: FieldParams = None,
        pop_size: int = 30,
        max_generations: int = 100,
        F: float = 0.8,
        CR: float = 0.7,
        tol: float = 1e-4,
        seed: int = 42,
        verbose: bool = True,
) -> DEResult:
    """
    DE/rand/1/bin optimisation of heliostat field parameters.

    Parameters
    ----------
    layout_type      : 'radial_staggered' or 'fermat_spiral'
    design_dni       : dict from solar_geometry.average_design_point_dni()
    base_params      : FieldParams with fixed parameters (min_radius etc.)
    pop_size         : population size (paper uses 30)
    max_generations  : maximum iterations
    F                : differential weight  [0.4 – 1.0]
    CR               : crossover probability [0.0 – 1.0]
    tol              : convergence tolerance (|Δη| < tol → stop)
    seed             : RNG seed for reproducibility
    verbose          : print progress

    Returns
    -------
    DEResult
    """
    rng = np.random.default_rng(seed)
    if base_params is None:
        base_params = FieldParams()

    D = 4   # number of genes: TH, LH, WR, DS

    # ── Initialise population uniformly in [LOWER, UPPER] ──────────────────
    pop = rng.uniform(0, 1, (pop_size, D)) * (UPPER - LOWER) + LOWER
    fit = np.array([fitness(pop[i], base_params, layout_type, design_dni)
                    for i in range(pop_size)])

    best_idx = int(np.argmax(fit))
    best_fit = fit[best_idx].copy()
    history  = [best_fit]

    t0 = time.time()
    if verbose:
        print(f"\n{'─'*60}")
        print(f" DE Optimisation  |  layout={layout_type}  |  pop={pop_size}  |  F={F}  |  CR={CR}")
        print(f"{'─'*60}")

    for gen in range(max_generations):
        for i in range(pop_size):
            # ── Mutation: select 3 distinct indices ≠ i ──────────────────
            candidates = [j for j in range(pop_size) if j != i]
            r1, r2, r3 = rng.choice(candidates, 3, replace=False)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, LOWER, UPPER)

            # ── Binomial crossover ────────────────────────────────────────
            mask   = rng.random(D) < CR
            j_rand = rng.integers(0, D)
            mask[j_rand] = True                # ensure at least one gene crosses
            trial  = np.where(mask, mutant, pop[i])

            # ── Selection ─────────────────────────────────────────────────
            f_trial = fitness(trial, base_params, layout_type, design_dni)
            if f_trial > fit[i]:
                pop[i] = trial
                fit[i] = f_trial
                if f_trial > best_fit:
                    best_fit = f_trial
                    best_idx = i

        history.append(best_fit)

        # ── Convergence check ─────────────────────────────────────────────
        if gen > 5 and abs(history[-1] - history[-6]) < tol:
            if verbose:
                print(f"  Converged at generation {gen+1}  (Δη < {tol})")
            break

        if verbose and (gen + 1) % 10 == 0:
            print(f"  Gen {gen+1:4d} | best fitness = {best_fit:.4f}%")

    runtime = time.time() - t0

    # ── Extract best solution ──────────────────────────────────────────────
    best_v      = pop[best_idx]
    best_params = _params_from_vector(best_v, base_params)
    positions   = build_layout(best_params, layout_type)

    ann_eff    = annual_mean_efficiency(positions, best_params)
    vernal_day = DESIGN_POINTS["Vernal Equinox"]["day"]
    dni_v      = design_dni.get("Vernal Equinox", 858.47)
    power_mw   = field_total_power_mw(positions, best_params, vernal_day, dni_v)

    if verbose:
        print(f"\n  Best annual efficiency : {ann_eff:.2f}%")
        print(f"  Tower height (TH)      : {best_params.tower_height:.1f} m")
        print(f"  Heliostat length (LH)  : {best_params.heliostat_length:.2f} m")
        print(f"  Width ratio (WR)       : {best_params.width_ratio:.3f}")
        print(f"  Security dist (DS)     : {best_params.security_dist:.3f}")
        print(f"  Heliostats             : {len(positions)}")
        print(f"  Power (vernal eq.)     : {power_mw:.2f} MW")
        print(f"  Runtime                : {runtime:.1f} s")

    return DEResult(
        best_params        = best_params,
        best_efficiency    = ann_eff,
        best_power_mw      = power_mw,
        n_heliostats       = len(positions),
        convergence_history= history,
        runtime_s          = runtime,
        layout_type        = layout_type,
    )
