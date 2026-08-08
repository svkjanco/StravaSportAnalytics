"""
HR-zone / intensity-distribution analysis (Friel %LTHR zones).

Zones as fraction of LTHR:
  Z1 recovery   < 0.81
  Z2 aerobic    0.81-0.89
  Z3 tempo      0.89-0.93
  Z4 threshold  0.93-0.99
  Z5 vo2max+    >= 0.99

Useful for the classic "80/20 polarized" check: healthy weeks are dominated
by Z1/Z2, with Z4/Z5 kept small and deliberate.
"""
from __future__ import annotations

import pandas as pd

ZONES = [
    (0.00, 0.81, "Z1 regeneracia"),
    (0.81, 0.89, "Z2 aerobne"),
    (0.89, 0.93, "Z3 tempo"),
    (0.93, 0.99, "Z4 prah"),
    (0.99, 99.0, "Z5 VO2max+"),
]
ZONE_NAMES = [name for *_, name in ZONES]


def zone_of(frac: float) -> str:
    for lo, hi, name in ZONES:
        if lo <= frac < hi:
            return name
    return ZONE_NAMES[-1]


def session_zone_seconds(hr_stream: list[float], lthr: float) -> dict[str, int]:
    """Seconds spent in each HR zone for one activity (1 Strava sample ~= 1s)."""
    out = {name: 0 for name in ZONE_NAMES}
    if not hr_stream or not lthr:
        return out
    for hr in hr_stream:
        out[zone_of(hr / lthr)] += 1
    return out


def build_zone_table(activity_df: pd.DataFrame, streams_lookup, lthr: float) -> pd.DataFrame:
    """One row per activity with seconds spent in each HR zone."""
    rows = []
    for _, a in activity_df.iterrows():
        hr = streams_lookup(a["id"]).get("heartrate") or []
        rows.append({"id": a["id"], "date": a["date"], "sport": a.get("sport"),
                     **session_zone_seconds(hr, lthr)})
    if not rows:
        return pd.DataFrame(columns=["id", "date", "sport", *ZONE_NAMES])
    return pd.DataFrame(rows)
