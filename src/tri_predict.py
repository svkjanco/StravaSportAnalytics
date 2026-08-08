"""
Triathlon split predictor: models swim + bike + run + T1/T2 SEPARATELY,
because you can't Riegel across disciplines.

  swim: avg pace = CSS_pace / swim_intensity  (intensity = fraction of threshold speed)
  bike: target power = FTP * bike_IF, then a physics model gives speed from power
        (solves  P = v*(0.5*rho*CdA*v^2 + Crr*m*g + m*g*grade)  for v)
  run : avg pace = run_threshold_pace / run_intensity, with an optional
        durability penalty scaling with how hard the bike leg was.

Presets encode typical race intensities; every value is overridable.
"""
from __future__ import annotations

import numpy as np

G = 9.81

# distances (m) + typical intensities per race format
PRESETS = {
    "ironman": dict(swim_m=3800, bike_m=180200, run_m=42195,
                    swim_i=0.90, bike_if=0.70, run_i=0.84, t1=180, t2=240),
    "70.3":    dict(swim_m=1900, bike_m=90100,  run_m=21097,
                    swim_i=0.93, bike_if=0.80, run_i=0.88, t1=120, t2=150),
    "olympic": dict(swim_m=1500, bike_m=40000,  run_m=10000,
                    swim_i=0.97, bike_if=0.88, run_i=0.92, t1=90,  t2=90),
    "sprint":  dict(swim_m=750,  bike_m=20000,  run_m=5000,
                    swim_i=1.00, bike_if=0.92, run_i=0.95, t1=60,  t2=60),
}


def speed_from_power(power_w, mass_kg, cda, crr, rho=1.225,
                     grade=0.0, drivetrain_loss=0.02) -> float:
    """Solve steady-state cycling power equation for speed (m/s) on given grade."""
    p = power_w * (1.0 - drivetrain_loss)
    a = 0.5 * rho * cda                       # aero (v^3 term coeff)
    c = crr * mass_kg * G + mass_kg * G * grade   # rolling + gravity (v term)
    roots = np.roots([a, 0.0, c, -p])         # a v^3 + c v - p = 0
    real = [r.real for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    return max(real) if real else np.nan


def _fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def predict_triathlon(ftp, run_threshold_pace_s_per_km, swim_css_s_per_100m,
                      bike_physics: dict, race="ironman",
                      durability=True, **overrides) -> dict:
    """
    bike_physics: dict(mass_kg, cda, crr, rho, drivetrain_loss)
    race: preset name, or pass swim_m/bike_m/run_m + intensities via overrides.
    """
    cfg = dict(PRESETS[race]) if race in PRESETS else {}
    cfg.update(overrides)

    # --- swim ---
    swim_pace_100 = swim_css_s_per_100m / cfg["swim_i"]
    swim_s = (cfg["swim_m"] / 100.0) * swim_pace_100

    # --- bike ---
    bike_power = ftp * cfg["bike_if"]
    v = speed_from_power(bike_power, bike_physics["mass_kg"], bike_physics["cda"],
                         bike_physics["crr"], bike_physics.get("rho", 1.225),
                         grade=0.0, drivetrain_loss=bike_physics.get("drivetrain_loss", 0.02))
    bike_s = cfg["bike_m"] / v if v and not np.isnan(v) else np.nan

    # --- run (with optional bike-induced durability penalty) ---
    run_i = cfg["run_i"]
    if durability:
        # harder bike (higher IF) + longer race => slower run. Small, bounded.
        penalty = min(0.06, max(0.0, (cfg["bike_if"] - 0.68) * 0.15)
                      * (cfg["run_m"] / 42195))
        run_i *= (1.0 - penalty)
    run_pace_km = run_threshold_pace_s_per_km / run_i
    run_s = (cfg["run_m"] / 1000.0) * run_pace_km

    total = swim_s + cfg["t1"] + bike_s + cfg["t2"] + run_s
    return {
        "race": race,
        "swim": _fmt(swim_s), "swim_s": swim_s,
        "t1": _fmt(cfg["t1"]),
        "bike": _fmt(bike_s), "bike_s": bike_s, "bike_kmh": (v * 3.6 if v else np.nan),
        "bike_power_w": bike_power,
        "t2": _fmt(cfg["t2"]),
        "run": _fmt(run_s), "run_s": run_s, "run_pace_per_km": _fmt(run_pace_km),
        "total": _fmt(total), "total_s": total,
    }
