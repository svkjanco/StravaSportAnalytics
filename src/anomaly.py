"""
Output #4: fatigue / overtraining detection.

Two complementary signals:

  (a) Aerobic decoupling per long aerobic effort - how much your HR:power (or
      HR:pace) efficiency drifts from the first half to the second half. Rising
      decoupling on steady efforts is a classic aerobic-fatigue / low-durability
      marker.

  (b) IsolationForest over daily training-state features (TSB, ATL, ramp rate,
      HR-at-power drift). Flags days that look statistically abnormal vs. your
      own history - useful as an early "something's off" alarm (illness,
      accumulated fatigue, non-functional overreaching).

This is unsupervised: it learns *your* normal, so it works on one athlete.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def aerobic_decoupling(streams: dict, min_seconds: int = 1800) -> float:
    """Pw:Hr (or Pace:Hr) decoupling %, first half vs second half.
    <5% = well-coupled/durable, >5% = drifting/fatiguing. NaN if too short."""
    hr = np.array(streams.get("heartrate") or [], float)
    watts = np.array(streams.get("watts") or [], float)
    vel = np.array(streams.get("velocity_smooth") or [], float)

    signal = watts if watts.size else vel  # prefer power, else speed
    n = min(len(hr), len(signal))
    if n < min_seconds:
        return np.nan
    hr, signal = hr[:n], signal[:n]
    mid = n // 2

    def ef(sig, h):
        h = h[h > 0]
        sig = sig[: len(h)]
        if len(h) == 0 or np.mean(h) == 0:
            return np.nan
        return np.mean(sig) / np.mean(h)  # efficiency factor

    ef1, ef2 = ef(signal[:mid], hr[:mid]), ef(signal[mid:], hr[mid:])
    if np.isnan(ef1) or np.isnan(ef2) or ef1 == 0:
        return np.nan
    return float((ef1 - ef2) / ef1 * 100.0)  # positive = HR drifted up


def build_daily_features(pmc: pd.DataFrame,
                        activity_df: pd.DataFrame,
                        streams_lookup) -> pd.DataFrame:
    """Assemble per-day features for anomaly scoring."""
    df = pmc.copy()
    df["ramp"] = df["ctl"].diff(7)  # 7-day CTL ramp rate

    # attach mean decoupling of that day's long efforts
    decouple = {}
    for _, a in activity_df.iterrows():
        d = aerobic_decoupling(streams_lookup(a["id"]))
        if not np.isnan(d):
            decouple.setdefault(a["date"], []).append(d)
    df["decoupling"] = df["date"].map(
        lambda d: np.mean(decouple[d]) if d in decouple else np.nan
    )
    return df


def detect_anomalies(feature_df: pd.DataFrame,
                     cols=("tsb", "atl", "ramp", "decoupling"),
                     contamination: float = 0.05) -> pd.DataFrame:
    """IsolationForest over selected features. Returns df with anomaly flag/score."""
    df = feature_df.copy()
    use = [c for c in cols if c in df.columns]
    X = df[use].copy()
    # fill NaNs with column medians so sparse features (decoupling) don't drop days
    X = X.fillna(X.median(numeric_only=True))
    if len(X) < 20:
        df["anomaly"] = False
        df["anomaly_score"] = 0.0
        return df
    model = IsolationForest(contamination=contamination, random_state=42)
    df["anomaly"] = model.fit_predict(X.values) == -1
    df["anomaly_score"] = -model.score_samples(X.values)  # higher = more anomalous
    return df
