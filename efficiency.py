"""
efficiency.py
-------------
Optical efficiency calculations for each heliostat:
  - Cosine efficiency      (cos ω_i)
  - Atmospheric attenuation (τ_a)
  - Spillage factor         (f_sp)
  - Overall efficiency      = cos ω · f_b · f_sp · f_at
  - Power per heliostat     = I · ρ · cos ω · f_sp · f_sb · f_at

References: Haris et al. (2023), Eqs. (5)–(14)
            Kistler DELSOL3 (1986)
            Vittitoe & Biggs (1978)
"""

import numpy as np
from scipy.special import erf as scipy_erf
from solar_geometry import solar_elevation, solar_azimuth
from heliostat_field import FieldParams


# ─── Cosine efficiency ────────────────────────────────────────────────────────
def cosine_efficiency(positions: np.ndarray,
                      tower_height: float,
                      alpha_rad: float,
                      azimuth_rad: float) -> np.ndarray:
    """
    Vectorised cosine efficiency for all heliostats (Eq. 5).
    cos 2θ_i = [(z₀-z₁)sin α − e₁ cos α sin A − n₁ cos α cos A]
               / sqrt[(z₀-z₁)² + e₁² + n₁²]

    positions : (N,2) array of (x,y) heliostat centres  [m]
    tower_height : receiver height above ground         [m]
    alpha_rad : solar elevation angle                   [rad]
    azimuth_rad : solar azimuth angle                   [rad]
    """
    e1 = positions[:, 0]        # East component
    n1 = positions[:, 1]        # North component
    dz = tower_height           # z₀ − z₁  (heliostat at ground level)

    numerator = (dz * np.sin(alpha_rad)
                 - e1 * np.cos(alpha_rad) * np.sin(azimuth_rad)
                 - n1 * np.cos(alpha_rad) * np.cos(azimuth_rad))
    denominator = np.sqrt(dz**2 + e1**2 + n1**2) + 1e-9

    cos2theta = numerator / denominator
    # cos efficiency = sqrt((1+cos2θ)/2)  from half-angle identity
    cos_eff = np.sqrt(np.clip((1 + cos2theta) / 2, 0, 1))
    return cos_eff


# ─── Atmospheric attenuation ──────────────────────────────────────────────────
def attenuation_efficiency(positions: np.ndarray,
                           tower_height: float) -> np.ndarray:
    """
    Vittitoe & Biggs (1978), Eq. 14:
    τ_a = 0.99326 − 0.1046·S + 0.017·S² − 0.002845·S³
    where S = slant range [km].
    """
    dist_2d = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    S_km = np.sqrt(dist_2d**2 + tower_height**2) / 1000.0
    tau = 0.99326 - 0.1046 * S_km + 0.017 * S_km**2 - 0.002845 * S_km**3
    return np.clip(tau, 0, 1)


# ─── PH helper function ───────────────────────────────────────────────────────
def _PH(xi_r: float, ar: float) -> float:
    """
    PH(ξ_r, −a_r, a_r) from Eq. 13.
    """
    upper = xi_r + ar
    lower = xi_r - ar
    val = 0.5 * (upper * scipy_erf(upper) + np.exp(-upper**2) / np.sqrt(np.pi)
                 - lower * scipy_erf(lower) - np.exp(-lower**2) / np.sqrt(np.pi))
    return val


# ─── Spillage factor ──────────────────────────────────────────────────────────
def spillage_factor(positions: np.ndarray,
                    params: FieldParams,
                    tower_height: float) -> np.ndarray:
    """
    Vectorised spillage factor f_sp for each heliostat (Eq. 12).
    """
    dist_2d = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    S = np.sqrt(dist_2d**2 + tower_height**2)              # slant range [m]

    sigma_r = params.sunshape_std                           # 2.51 mrad
    # Effective sun-shape dispersion scaled by slant range
    sigma_eff = sigma_r * S                                 # [m]

    Ah = params.heliostat_area
    ar = np.sqrt(Ah) / (2 * np.sqrt(2) * sigma_eff + 1e-9)

    f_sp = np.zeros(len(positions))
    for i, (a, sig) in enumerate(zip(ar, sigma_eff)):
        xi_r_w = params.width_ratio * params.heliostat_length / (2 * np.sqrt(2) * sig + 1e-9)
        xi_r_l = params.heliostat_length / (2 * np.sqrt(2) * sig + 1e-9)
        ph_w = _PH(xi_r_w, a)
        ph_l = _PH(xi_r_l, a)
        f_sp[i] = ph_w * ph_l / (a**2 + 1e-9)

    return np.clip(f_sp, 0, 1)


# ─── Overall efficiency per heliostat ────────────────────────────────────────
def overall_efficiency(positions: np.ndarray,
                       params: FieldParams,
                       day_number: int,
                       hour: float = 11.0) -> np.ndarray:
    """
    η_i = cos ω_i · f_b · f_sp · f_at      (Eq. 11)
    Returns array of shape (N,).
    """
    alpha  = solar_elevation(day_number, hour)
    A      = solar_azimuth(day_number, hour)

    cos_eff = cosine_efficiency(positions, params.tower_height, alpha, A)
    f_at    = attenuation_efficiency(positions, params.tower_height)
    f_sp    = spillage_factor(positions, params, params.tower_height)

    eta = cos_eff * params.blocking_factor * f_sp * f_at
    return np.clip(eta, 0, 1)


# ─── Power per heliostat ──────────────────────────────────────────────────────
def power_per_heliostat(positions: np.ndarray,
                        params: FieldParams,
                        day_number: int,
                        dni: float,
                        hour: float = 11.0) -> np.ndarray:
    """
    P_i = I · ρ · cos ω_i · f_sp · f_sb · f_at   [W/m²]  ×  A_h  →  [W]
    (Eq. 10)
    """
    alpha  = solar_elevation(day_number, hour)
    A      = solar_azimuth(day_number, hour)

    cos_eff = cosine_efficiency(positions, params.tower_height, alpha, A)
    f_at    = attenuation_efficiency(positions, params.tower_height)
    f_sp    = spillage_factor(positions, params, params.tower_height)

    power = (dni * params.reflectivity * cos_eff * f_sp * params.blocking_factor * f_at
             * params.heliostat_area)
    return np.clip(power, 0, None)


# ─── Field-level aggregates ───────────────────────────────────────────────────
def field_mean_efficiency(positions: np.ndarray,
                          params: FieldParams,
                          day_number: int,
                          hour: float = 11.0) -> float:
    """Mean overall efficiency of all heliostats (%)."""
    eta = overall_efficiency(positions, params, day_number, hour)
    return float(np.mean(eta)) * 100.0


def field_total_power_mw(positions: np.ndarray,
                          params: FieldParams,
                          day_number: int,
                          dni: float,
                          hour: float = 11.0) -> float:
    """Total thermal input power of the field [MW]."""
    pw = power_per_heliostat(positions, params, day_number, dni, hour)
    return float(np.sum(pw)) / 1e6


def annual_mean_efficiency(positions: np.ndarray,
                            params: FieldParams) -> float:
    """Average efficiency across the four seasonal design points (%)."""
    from solar_geometry import DESIGN_POINTS
    effs = [field_mean_efficiency(positions, params, info["day"])
            for info in DESIGN_POINTS.values()]
    return float(np.mean(effs))
