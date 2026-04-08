"""
main.py
-------
End-to-end pipeline:
  1. Load DNI data from CSV
  2. Build unoptimised layouts (RS + FS)
  3. Run Differential Evolution optimisation on both layouts
  4. Produce all figures (Figs 1–8)
  5. Print summary table replicating Table 6 from the paper

Run from the project root:
    cd heliostat_de
    python src/main.py

No API calls are made. All computation is local.
"""

import sys
import os

# ── Ensure src is on path when running directly ───────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from solar_geometry import load_dni_data, average_design_point_dni, DESIGN_POINTS
from heliostat_field import FieldParams, radial_staggered_layout, fermat_spiral_layout
from efficiency import (annual_mean_efficiency, field_mean_efficiency,
                         field_total_power_mw)
from de_optimizer import differential_evolution
from plotting import (plot_attenuation_rs, plot_cosine_4panel_rs,
                      plot_attenuation_fs, plot_power_4panel_fs,
                      plot_optimised_layouts, plot_convergence,
                      plot_efficiency_comparison, plot_dni_data)


CSV_PATH = os.path.join(os.path.dirname(__file__), "solar-measurementspakistanquettawb-esmapqc.csv")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs")


# ─── 1. Load data ─────────────────────────────────────────────────────────────
def load_data():
    print("Loading DNI data …")
    df = load_dni_data(CSV_PATH)
    design_dni = average_design_point_dni(df)
    print("  Design-point DNI values (W/m²):")
    for k, v in design_dni.items():
        paper_val = {"Vernal Equinox":858.47, "Summer Solstice":965.64,
                     "Autumnal Equinox":875.71, "Winter Solstice":856.63}
        print(f"    {k:20s}: {v:.2f}  (paper: {paper_val[k]:.2f})")
    return df, design_dni


# ─── 2. Unoptimised baselines ─────────────────────────────────────────────────
def compute_baselines(design_dni: dict):
    print("\nComputing unoptimised baselines …")
    base = FieldParams()

    rs_pos = radial_staggered_layout(base, max_radius=450)
    fs_pos = fermat_spiral_layout(base, n_heliostats=1300)

    rs_eff_unopt = annual_mean_efficiency(rs_pos, base)
    fs_eff_unopt = annual_mean_efficiency(fs_pos, base)

    print(f"  RS unoptimised: {len(rs_pos)} heliostats, η={rs_eff_unopt:.2f}%")
    print(f"  FS unoptimised: {len(fs_pos)} heliostats, η={fs_eff_unopt:.2f}%")

    # Per-design-point table (replicates Table 4 in paper)
    print("\n  Table 4 – Heliostats required per design point (RS, unoptimised):")
    print(f"  {'Design Point':22s} | {'Heliostats':10s} | {'Efficiency (%)':15s}")
    print("  " + "─" * 55)
    overall_effs = []
    for name, info in DESIGN_POINTS.items():
        eff = field_mean_efficiency(rs_pos, base, info["day"])
        overall_effs.append(eff)
        print(f"  {name:22s} | {len(rs_pos):10d} | {eff:15.2f}")
    print(f"  {'Overall Mean':22s} | {'':10s} | {np.mean(overall_effs):15.2f}")

    return base, rs_pos, fs_pos, rs_eff_unopt, fs_eff_unopt


# ─── 3. DE Optimisation ───────────────────────────────────────────────────────
def run_optimisation(design_dni: dict, base: FieldParams):
    print("\n" + "═"*60)
    print(" Running DE Optimisation – Radial Staggered")
    print("═"*60)
    rs_result = differential_evolution(
        layout_type     = "radial_staggered",
        design_dni      = design_dni,
        base_params     = base,
        pop_size        = 30,
        max_generations = 100,
        F               = 0.8,
        CR              = 0.7,
        tol             = 1e-4,
        seed            = 42,
        verbose         = True,
    )

    print("\n" + "═"*60)
    print(" Running DE Optimisation – Fermat's Spiral")
    print("═"*60)
    fs_result = differential_evolution(
        layout_type     = "fermat_spiral",
        design_dni      = design_dni,
        base_params     = base,
        pop_size        = 30,
        max_generations = 100,
        F               = 0.8,
        CR              = 0.7,
        tol             = 1e-4,
        seed            = 42,
        verbose         = True,
    )
    return rs_result, fs_result


# ─── 4. Summary table (Table 6 equivalent) ────────────────────────────────────
def print_summary_table(rs_eff_unopt, rs_result,
                        fs_eff_unopt, fs_result,
                        n_rs_before, n_fs_before):
    print("\n" + "═"*80)
    print(" SUMMARY TABLE – Differential Evolution Results")
    print(" (Replicates Table 6 from Haris et al. 2023, with DE instead of GA)")
    print("═"*80)
    header = (f"{'Layout':18s} | {'TH(m)':6s} | {'WR':5s} | {'LH(m)':6s} | "
              f"{'DS':5s} | {'η_before(%)':11s} | {'η_after(%)':10s} | "
              f"{'N_before':8s} | {'N_after':7s}")
    print(header)
    print("─" * len(header))

    for label, eff_b, result, n_b in [
        ("Radial Staggered", rs_eff_unopt, rs_result, n_rs_before),
        ("Fermat's Spiral",  fs_eff_unopt, fs_result, n_fs_before),
    ]:
        p = result.best_params
        print(f"{label:18s} | {p.tower_height:6.1f} | {p.width_ratio:5.2f} | "
              f"{p.heliostat_length:6.2f} | {p.security_dist:5.3f} | "
              f"{eff_b:11.2f} | {result.best_efficiency:10.2f} | "
              f"{n_b:8d} | {result.n_heliostats:7d}")
        imp = result.best_efficiency - eff_b
        red = n_b - result.n_heliostats
        print(f"{'  → Improvement':18s}   {'':6s}   {'':5s}   {'':6s}   {'':5s}   "
              f"{'':11s}   +{imp:.2f}%     {'':8s}   −{red}")

    print("═"*80)
    print(f"\n  NOTE: GA (original paper) → RS improved by 8.52%, FS by 14.62%")
    print(f"         DE (this work)       → compare above figures")


# ─── 5. Generate all figures ──────────────────────────────────────────────────
def generate_figures(df, design_dni, rs_result, fs_result,
                     rs_eff_unopt, fs_eff_unopt,
                     n_rs_before, n_fs_before):
    print("\nGenerating figures …")
    plot_attenuation_rs()
    plot_cosine_4panel_rs()
    plot_attenuation_fs()
    plot_power_4panel_fs(design_dni)
    plot_optimised_layouts(rs_result, fs_result, design_dni)
    plot_convergence(rs_result, fs_result)
    plot_efficiency_comparison(
        rs_eff_unopt, rs_result.best_efficiency,
        fs_eff_unopt, fs_result.best_efficiency,
        n_rs_before,  rs_result.n_heliostats,
        n_fs_before,  fs_result.n_heliostats,
    )
    plot_dni_data(df)
    print(f"\nAll figures saved to: {os.path.abspath(OUT_DIR)}")


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Heliostat Field Layout Optimisation – Differential Evolution║")
    print("║  Site: Quetta, Balochistan, Pakistan (30.18°N, 66.97°E)     ║")
    print("║  Target: 50 MW Central Receiver Solar Thermal Power Plant   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    df, design_dni = load_data()
    base, rs_pos, fs_pos, rs_eff_unopt, fs_eff_unopt = compute_baselines(design_dni)
    n_rs_before = len(rs_pos)
    n_fs_before = len(fs_pos)

    rs_result, fs_result = run_optimisation(design_dni, base)

    print_summary_table(rs_eff_unopt, rs_result,
                        fs_eff_unopt, fs_result,
                        n_rs_before, n_fs_before)

    generate_figures(df, design_dni, rs_result, fs_result,
                     rs_eff_unopt, fs_eff_unopt,
                     n_rs_before, n_fs_before)

    print("\nDone ✓")


if __name__ == "__main__":
    main()
