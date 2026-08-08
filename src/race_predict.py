"""
Output #1: race-time prediction.

Two layers, because on one athlete's data a pure black-box model overfits:

  (a) Riegel baseline  -  physiology-based, needs only ONE recent best effort:
        T2 = T1 * (D2/D1) ** 1.06
      Great for "I ran a strong 10k, what's my marathon?" style questions.

  (b) ML refinement  -  if you have several past races labelled, a Ridge/GBR
      model learns a correction from training-state features (CTL, ATL, TSB,
      recent volume) around the Riegel prediction. With few races it stays
      conservative; more races -> more signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RIEGEL_EXP = 1.06  # classic endurance fatigue exponent


def riegel(known_time_s: float, known_dist_m: float, target_dist_m: float,
           exp: float = RIEGEL_EXP) -> float:
    """Predict time for target distance from one known effort."""
    return known_time_s * (target_dist_m / known_dist_m) ** exp


def _features_on(date, pmc: pd.DataFrame) -> dict:
    """Training-state features as of a given date."""
    row = pmc[pmc["date"] <= pd.to_datetime(date)].tail(1)
    if row.empty:
        return {"ctl": 0.0, "atl": 0.0, "tsb": 0.0}
    r = row.iloc[0]
    return {"ctl": float(r["ctl"]), "atl": float(r["atl"]), "tsb": float(r["tsb"])}


def build_race_dataset(races: pd.DataFrame, pmc: pd.DataFrame) -> pd.DataFrame:
    """races: columns [date, distance_m, time_s]. Adds Riegel baseline (from the
    athlete's own best prior effort) + PMC features + the residual to learn."""
    races = races.sort_values("date").reset_index(drop=True)
    rows = []
    for _, r in races.iterrows():
        # best pace effort strictly before this race, as the Riegel anchor
        prior = races[races["date"] < r["date"]]
        if prior.empty:
            base = r["time_s"]  # no anchor -> baseline == actual (residual 0)
        else:
            # anchor = prior race with fastest equivalent (lowest riegel-normalised)
            anchor = prior.assign(
                pred=lambda d: riegel(d["time_s"], d["distance_m"], r["distance_m"])
            ).sort_values("pred").iloc[0]
            base = riegel(anchor["time_s"], anchor["distance_m"], r["distance_m"])
        feats = _features_on(r["date"], pmc)
        rows.append({**r.to_dict(), "riegel_pred_s": base,
                     "residual_s": r["time_s"] - base, **feats})
    return pd.DataFrame(rows)


class RacePredictor:
    """Riegel + learned residual correction from training state."""

    def __init__(self):
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        self.fitted = False
        self.feat_cols = ["ctl", "atl", "tsb"]

    def fit(self, dataset: pd.DataFrame):
        usable = dataset.dropna(subset=["residual_s"])
        if len(usable) >= 4:  # need a few races to learn anything
            X = usable[self.feat_cols].values
            y = usable["residual_s"].values
            self.model.fit(X, y)
            self.fitted = True
        return self

    def predict(self, known_time_s, known_dist_m, target_dist_m,
                pmc: pd.DataFrame, on_date) -> dict:
        base = riegel(known_time_s, known_dist_m, target_dist_m)
        correction = 0.0
        if self.fitted:
            feats = _features_on(on_date, pmc)
            X = np.array([[feats[c] for c in self.feat_cols]])
            correction = float(self.model.predict(X)[0])
        return {
            "riegel_s": base,
            "correction_s": correction,
            "predicted_s": base + correction,
            "predicted_hms": _fmt(base + correction),
        }


def _fmt(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"
