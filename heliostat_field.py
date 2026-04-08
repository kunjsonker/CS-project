"""
heliostat_field.py
------------------
Generates heliostat field layouts:
  1. Radial Staggered  (Siala & Elayeb 2001, Eq. 1-6)
  2. Fermat's Spiral   (golden-angle sunflower distribution)

All equations correspond to the paper:
  Haris et al., Heliyon 9 (2023) e21488
"""

import numpy as np
from dataclasses import dataclass


# ─── Heliostat / field parameter container ────────────────────────────────────
@dataclass
class FieldParams:
    tower_height: float   = 130.0   # TH  [m]          50–300
    heliostat_length: float = 10.95 # LH  [m]          5–20
    width_ratio: float    = 1.0     # WR  (W/L)        1–2
    security_dist: float  = 0.2     # DS  [–]          0.1–0.5
    min_radius: float     = 65.0    # R_min [m]
    blocking_factor: float = 0.97   # f_b
    reflectivity: float   = 0.88    # ρ
    sunshape_std: float   = 2.51e-3 # σ_r [rad]

    @property
    def heliostat_width(self) -> float:
        return self.width_ratio * self.heliostat_length

    @property
    def heliostat_area(self) -> float:
        return self.heliostat_length * self.heliostat_width


# ─── Radial spacing equations (Kistler DELSOL3, 1986) ─────────────────────────
def _altitude_angle(r: float, tower_height: float) -> float:
    """θ_L = arctan(TH / r)  [rad]"""
    return np.arctan(tower_height / (r + 1e-9))


def radial_spacing(theta_L: float, HM: float) -> float:
    """ΔR = HM·(1.44·cot(θ_L) − 1.094 + 3.068·θ_L − 1.1256·θ_L²)"""
    cot = np.cos(theta_L) / (np.sin(theta_L) + 1e-9)
    return HM * (1.44 * cot - 1.094 + 3.068 * theta_L - 1.1256 * theta_L**2)


def azimuthal_spacing(theta_L: float, WM: float) -> float:
    """ΔA = WM·(1.749 + 0.6396·θ_L) + 0.2873/(θ_L − 0.04902)"""
    return WM * (1.749 + 0.6396 * theta_L) + 0.2873 / (theta_L - 0.04902 + 1e-9)


# ─── Radial Staggered Layout ──────────────────────────────────────────────────
def radial_staggered_layout(params: FieldParams, max_radius: float = 500.0,
                             security_extra: float = 0.0) -> np.ndarray:
    """
    Generate (x, y) heliostat positions using the radial staggered method.
    Returns array of shape (N, 2).
    """
    HM = params.heliostat_length * (1 + params.security_dist + security_extra)
    WM = params.heliostat_width  * (1 + params.security_dist + security_extra)

    positions = []
    r = params.min_radius

    while r <= max_radius:
        theta_L = _altitude_angle(r, params.tower_height)
        if theta_L < 0.04903:          # avoid singularity in azimuthal_spacing
            r += HM
            continue

        dA = azimuthal_spacing(theta_L, WM)
        circumference = 2 * np.pi * r
        n_helio = max(1, int(circumference / dA))
        phi_step = 2 * np.pi / n_helio

        # Stagger alternate rings by half a step
        offset = (phi_step / 2) if (len(positions) // max(1, n_helio)) % 2 else 0

        for i in range(n_helio):
            phi = i * phi_step + offset
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            positions.append((x, y))

        dR = radial_spacing(theta_L, HM)
        r += max(dR, HM * 0.5)

    return np.array(positions)


# ─── Fermat's Spiral Layout ───────────────────────────────────────────────────
def fermat_spiral_layout(params: FieldParams, n_heliostats: int = 1500,
                          scale_factor: float = None,
                          security_extra: float = 0.0) -> np.ndarray:
    """
    Generate heliostat positions using Fermat's spiral (golden angle).
    r = c·√n,  θ = n · 137.508°
    Returns array of shape (N, 2).
    """
    HM = params.heliostat_length * (1 + params.security_dist + security_extra)
    if scale_factor is None:
        scale_factor = HM * 0.85

    golden_angle = np.radians(137.508)
    positions = []

    for n in range(1, n_heliostats + 1):
        r = scale_factor * np.sqrt(n)
        if r < params.min_radius:
            continue
        theta = n * golden_angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        positions.append((x, y))

    return np.array(positions)


# ─── Quick smoke test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = FieldParams()
    rs = radial_staggered_layout(p)
    fs = fermat_spiral_layout(p, n_heliostats=1300)
    print(f"Radial Staggered: {len(rs)} heliostats")
    print(f"Fermat's Spiral : {len(fs)} heliostats")
