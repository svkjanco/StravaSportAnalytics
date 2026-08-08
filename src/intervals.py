"""
Interval detection + workout classification.

Finds the *structure* of a session (reps vs. steady) from the power stream
(or pace/speed for runs), assigns each rep a physiological zone, and labels the
whole session (VO2max intervals, threshold intervals, tempo/sweet-spot,
sprints, or steady/endurance).

Zones by fraction of FTP (or of run threshold speed):
  recovery <0.55 | endurance 0.55-0.75 | tempo 0.75-0.90
  threshold 0.90-1.05 | vo2max 1.05-1.20 | anaerobic >1.20
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONES = [
    (0.00, 0.55, "recovery"), (0.55, 0.75, "endurance"),
    (0.75, 0.90, "tempo"),    (0.90, 1.05, "threshold"),
    (1.05, 1.20, "vo2max"),   (1.20, 99.0, "anaerobic"),
]


def zone_of(frac: float) -> str:
    for lo, hi, name in ZONES:
        if lo <= frac < hi:
            return name
    return "anaerobic"


def _find_reps(above: np.ndarray, min_rep_s: int, merge_gap_s: int):
    """Return (start, end) index pairs of work segments, merging short gaps."""
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return []
    # split into contiguous runs
    splits = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    segs = [(int(r[0]), int(r[-1]) + 1) for r in splits]
    # merge segments separated by a gap smaller than merge_gap_s
    merged = [segs[0]]
    for s, e in segs[1:]:
        if s - merged[-1][1] <= merge_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, e) for s, e in merged if (e - s) >= min_rep_s]


def detect_intervals(streams: dict, ftp: float, run_thr_speed_mps: float | None = None,
                     work_frac: float = 0.88, min_rep_s: int = 20,
                     merge_gap_s: int = 10) -> dict:
    """Detect reps and classify. Uses power if present, else speed for runs."""
    watts = streams.get("watts") or []
    if watts:
        sig = np.asarray(watts, float)
        ref = ftp
    else:
        vel = streams.get("velocity_smooth") or []
        sig = np.asarray(vel, float)
        ref = run_thr_speed_mps
    if ref is None or ref <= 0 or sig.size == 0:
        return {"n_reps": 0, "session_type": "unknown", "reps": [], "zone_time_s": {}}

    smooth = pd.Series(sig).rolling(10, min_periods=1, center=True).median().values
    frac = smooth / ref

    # time in each zone (whole session)
    zone_time = {name: 0 for *_, name in ZONES}
    for f in frac:
        zone_time[zone_of(f)] += 1

    above = frac >= work_frac
    reps = []
    for s, e in _find_reps(above, min_rep_s, merge_gap_s):
        seg = frac[s:e]
        avg_frac = float(np.mean(seg))
        reps.append({"start_s": s, "dur_s": e - s,
                     "avg_frac_ftp": round(avg_frac, 3),
                     "avg_watts": round(float(np.mean(smooth[s:e])), 1),
                     "zone": zone_of(avg_frac)})

    return {"n_reps": len(reps), "reps": reps,
            "session_type": _classify(reps, zone_time),
            "zone_time_s": zone_time}


def _classify(reps, zone_time) -> str:
    if not reps:
        # steady session: label by dominant work zone
        work = {k: v for k, v in zone_time.items() if k not in ("recovery",)}
        dom = max(work, key=work.get) if work else "recovery"
        return f"steady/{dom}"

    zones = [r["zone"] for r in reps]
    durs = [r["dur_s"] for r in reps]

    def frac_in(z):
        return sum(z == zz for zz in zones) / len(zones)

    # very short + supra-threshold => sprints
    if any(z == "anaerobic" for z in zones) and np.median(durs) <= 30:
        return "sprints/anaerobic"
    if frac_in("vo2max") + frac_in("anaerobic") >= 0.5:
        return "VO2max intervals"
    if frac_in("threshold") >= 0.5:
        return "threshold intervals"
    if frac_in("tempo") >= 0.5:
        return "tempo / sweet-spot"
    return "mixed intervals"


def classify_history(activity_df: pd.DataFrame, streams_lookup, ftp: float,
                     run_thr_speed_mps: float | None = None) -> pd.DataFrame:
    """Run detection over all activities -> one row per activity with its type."""
    rows = []
    for _, a in activity_df.iterrows():
        res = detect_intervals(streams_lookup(a["id"]), ftp, run_thr_speed_mps)
        rows.append({"id": a["id"], "date": a["date"], "sport": a.get("sport"),
                     "n_reps": res["n_reps"], "session_type": res["session_type"]})
    return pd.DataFrame(rows)
