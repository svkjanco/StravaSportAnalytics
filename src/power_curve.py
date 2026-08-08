"""
Mean-maximal power curve + Critical Power (CP) / W' model -> FTP estimate.

Output #3: estimate & track FTP without a formal test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# durations (seconds) used to build the curve
DURATIONS = [5, 15, 30, 60, 120, 180, 300, 480, 600, 720, 900, 1200, 1800, 3600]


def best_efforts(watts: list[float], durations=DURATIONS) -> dict[int, float]:
    """Best average power for each duration in one activity."""
    if not watts:
        return {}
    s = pd.Series(watts, dtype="float64")
    out = {}
    for d in durations:
        if len(s) >= d:
            out[d] = float(s.rolling(d, min_periods=d).mean().max())
    return out


def power_curve_envelope(activity_df: pd.DataFrame, streams_lookup,
                         window_days: int = 90, as_of: pd.Timestamp | None = None
                         ) -> dict[int, float]:
    """Best power at each duration across all power rides in a trailing window.
    streams_lookup: callable(activity_id) -> streams dict."""
    as_of_ts = activity_df["date"].max() if as_of is None else as_of
    start = as_of_ts - pd.Timedelta(days=window_days)
    mask = (activity_df["date"] > start) & (activity_df["date"] <= as_of_ts)
    curve: dict[int, float] = {}
    for _, row in activity_df.loc[mask].iterrows():
        try:
            streams = streams_lookup(row["id"])
        except Exception:
            continue  # aktivita bez streamov (404) -> preskoc
        for d, p in best_efforts(streams.get("watts") or []).items():
            curve[d] = max(curve.get(d, 0.0), p)
    return dict(sorted(curve.items()))


def fit_cp(curve: dict[int, float], lo_s: int = 120, hi_s: int = 720
           ) -> dict[str, float]:
    """Fit P = W'/t + CP over efforts between lo_s and hi_s seconds."""
    pts = [(t, p) for t, p in curve.items() if lo_s <= t <= hi_s and p > 0]
    if len(pts) < 2:
        return {"cp": np.nan, "w_prime": np.nan, "ftp_est": np.nan, "n_points": len(pts)}
    t = np.array([p[0] for p in pts], float)
    p = np.array([p[1] for p in pts], float)
    # linear regression p = W' * (1/t) + CP
    x = 1.0 / t
    A = np.vstack([x, np.ones_like(x)]).T
    (w_prime, cp), *_ = np.linalg.lstsq(A, p, rcond=None)
    ftp_20 = curve.get(1200, np.nan)
    ftp_est = cp if not np.isnan(cp) else (0.95 * ftp_20 if not np.isnan(ftp_20) else np.nan)
    return {"cp": float(cp), "w_prime": float(w_prime),
            "ftp_est": float(ftp_est), "n_points": len(pts)}


def ftp_timeline(activity_df: pd.DataFrame, streams_lookup,
                 step_days: int = 14, window_days: int = 90) -> pd.DataFrame:
    """Rolling FTP estimate over the whole history (tracks fitness in watts)."""
    if activity_df.empty:
        return pd.DataFrame(columns=["date", "cp", "w_prime", "ftp_est"])
    dates = pd.date_range(
        activity_df["date"].min() + pd.Timedelta(days=window_days),
        activity_df["date"].max(), freq=f"{step_days}D",
    )
    rows = []
    for as_of in dates:
        curve = power_curve_envelope(activity_df, streams_lookup,
                                     window_days=window_days, as_of=as_of)
        fit = fit_cp(curve)
        rows.append({"date": as_of, **fit})
    return pd.DataFrame(rows)