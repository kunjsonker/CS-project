"""
solar_geometry.py
-----------------
Solar position calculations and DNI data loading for Quetta, Pakistan.
Coordinates: 30.1798 N, 66.9750 E
"""

import numpy as np
import pandas as pd
from datetime import datetime


# ─── Site constants ───────────────────────────────────────────────────────────
LATITUDE_DEG  = 30.1798          # Quetta latitude (degrees North)
LATITUDE_RAD  = np.radians(LATITUDE_DEG)
LONGITUDE_DEG = 66.9750          # Quetta longitude (degrees East)
TIMEZONE      = 5                # UTC+5


# ─── Design-point day numbers & solar declinations ────────────────────────────
DESIGN_POINTS = {
    "Vernal Equinox":   {"day": 80,  "date": "March 21"},
    "Summer Solstice":  {"day": 172, "date": "June 21"},
    "Autumnal Equinox": {"day": 266, "date": "September 23"},
    "Winter Solstice":  {"day": 355, "date": "December 21"},
}


def solar_declination(day_number: int) -> float:
    """Return solar declination angle in radians for a given day of year."""
    return np.radians(23.45 * np.sin(np.radians(360 * (284 + day_number) / 365)))


def solar_hour_angle(hour: float) -> float:
    """Convert solar time (hours) to hour angle in radians."""
    return np.radians(15 * (hour - 12))


def solar_elevation(day_number: int, hour: float = 11.0) -> float:
    """
    Return solar elevation angle α (radians).
    α = arcsin(sin δ · sin φ + cos δ · cos ω · cos φ)
    """
    delta = solar_declination(day_number)
    omega = solar_hour_angle(hour)
    sin_alpha = (np.sin(delta) * np.sin(LATITUDE_RAD)
                 + np.cos(delta) * np.cos(omega) * np.cos(LATITUDE_RAD))
    return np.arcsin(np.clip(sin_alpha, -1, 1))


def solar_azimuth(day_number: int, hour: float = 11.0) -> float:
    """Return solar azimuth angle A (radians, measured from South)."""
    delta = solar_declination(day_number)
    omega = solar_hour_angle(hour)
    alpha = solar_elevation(day_number, hour)
    cos_A = (np.sin(delta) - np.sin(alpha) * np.sin(LATITUDE_RAD)) / \
            (np.cos(alpha) * np.cos(LATITUDE_RAD) + 1e-12)
    A = np.arccos(np.clip(cos_A, -1, 1))
    return A if hour <= 12 else (2 * np.pi - A)


# ─── DNI loading ──────────────────────────────────────────────────────────────
def load_dni_data(csv_path: str) -> pd.DataFrame:
    """Load and clean the Quetta solar measurement CSV."""
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.dropna(subset=["dni"])
    df["dni"] = pd.to_numeric(df["dni"], errors="coerce")
    df = df[df["dni"] >= 0]
    df["month"] = df["time"].dt.month
    df["day"]   = df["time"].dt.day
    df["hour"]  = df["time"].dt.hour + df["time"].dt.minute / 60
    return df


def average_design_point_dni(df: pd.DataFrame) -> dict:
    """
    Compute average DNI at 11:00 for each of the four design points,
    averaging across the available years (2015-2017).
    Returns dict matching paper's Table 3 values.
    """
    design_dni = {}
    windows = {
        "Vernal Equinox":   (3, 21),
        "Summer Solstice":  (6, 21),
        "Autumnal Equinox": (9, 23),
        "Winter Solstice":  (12, 21),
    }
    for name, (m, d) in windows.items():
        subset = df[(df["month"] == m) & (df["day"] == d) &
                    (df["hour"] >= 10.8) & (df["hour"] <= 11.2)]
        if len(subset):
            design_dni[name] = subset["dni"].mean()
        else:
            # Fall back to ±3-day window
            subset = df[(df["month"] == m) &
                        (df["day"].between(d - 3, d + 3)) &
                        (df["hour"] >= 10.5) & (df["hour"] <= 11.5)]
            design_dni[name] = subset["dni"].mean() if len(subset) else 860.0
    return design_dni


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for name, info in DESIGN_POINTS.items():
        alpha = np.degrees(solar_elevation(info["day"]))
        print(f"{name:20s} | day={info['day']:3d} | elev={alpha:.2f}°")
