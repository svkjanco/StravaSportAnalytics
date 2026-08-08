"""
Feature engineering with PER-SPORT training load (bike / run / swim).

TSS by discipline:
  bike  : power TSS if watts present, else HR fallback
  run   : pace-based rTSS from run threshold pace, else HR fallback
  swim  : pace-based sTSS from swim threshold pace (CSS), else HR fallback

Generic identity used throughout:  TSS = hours * IF**2 * 100
(equivalent to Coggan's power TSS, and the standard rTSS/sTSS form).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Strava's own "Race" workout_type (set when you tag an activity as a race in
# the Strava UI/app): 1 = run race, 11 = ride race. Swim has no workout_type,
# so those fall back to the name check below.
RACE_WORKOUT_TYPES = {1, 11}
RACE_NAME_RX = re.compile(r"\b(race|pretek\w*|z[aá]vod\w*)\b", re.IGNORECASE)

SPORT_MAP = {
    "Ride": "bike", "VirtualRide": "bike", "GravelRide": "bike",
    "MountainBikeRide": "bike", "EBikeRide": "bike",
    "Run": "run", "TrailRun": "run", "VirtualRun": "run",
    "Swim": "swim", "OpenWaterSwim": "swim",
}


def sport_of(strava_type: str) -> str:
    return SPORT_MAP.get(strava_type, "other")


def is_race(summary: dict) -> bool:
    """Race if tagged as such in Strava, or the title says so (race/pretek/zavod)."""
    if summary.get("workout_type") in RACE_WORKOUT_TYPES:
        return True
    return bool(RACE_NAME_RX.search(summary.get("name") or ""))


# ---------------------------------------------------------------------------
# Power metrics
# ---------------------------------------------------------------------------
def normalized_power(watts, sample_hz: float = 1.0) -> float:
    if not watts:
        return np.nan
    s = pd.Series(watts, dtype="float64")
    window = max(1, int(round(30 * sample_hz)))
    roll = s.rolling(window, min_periods=window).mean().dropna()
    if roll.empty:
        return np.nan
    return float((np.mean(roll.values ** 4)) ** 0.25)


# ---------------------------------------------------------------------------
# Discipline-specific TSS (all reduce to hours * IF**2 * 100)
# ---------------------------------------------------------------------------
def tss_power(duration_s, np_watts, ftp) -> float:
    if not ftp or np.isnan(np_watts):
        return np.nan
    return (duration_s / 3600.0) * (np_watts / ftp) ** 2 * 100.0


def tss_run(duration_s, distance_m, thr_pace_s_per_km) -> float:
    """rTSS from pace. IF = threshold_pace / avg_pace (faster => IF>1)."""
    if not distance_m or distance_m <= 0 or not thr_pace_s_per_km:
        return np.nan
    avg_pace = duration_s / (distance_m / 1000.0)          # s per km
    inten = thr_pace_s_per_km / avg_pace
    return (duration_s / 3600.0) * inten ** 2 * 100.0


def tss_swim(duration_s, distance_m, thr_pace_s_per_100m) -> float:
    """sTSS from pace. IF = CSS_pace / avg_pace per 100m."""
    if not distance_m or distance_m <= 0 or not thr_pace_s_per_100m:
        return np.nan
    avg_pace = duration_s / (distance_m / 100.0)           # s per 100m
    inten = thr_pace_s_per_100m / avg_pace
    return (duration_s / 3600.0) * inten ** 2 * 100.0


def tss_hr(duration_s, avg_hr, lthr) -> float:
    """HR fallback. One hour at threshold HR == 100."""
    if not lthr or not avg_hr or np.isnan(avg_hr):
        return np.nan
    return (duration_s / 3600.0) * (avg_hr / lthr) ** 2 * 100.0


# ---------------------------------------------------------------------------
# Per-activity load (dispatch by sport)
# ---------------------------------------------------------------------------
def activity_load(summary, streams, params) -> dict:
    """params: dict/obj with FTP, LTHR, RUN_THRESHOLD_PACE, SWIM_THRESHOLD_PACE."""
    sport = sport_of(summary.get("type"))
    watts = streams.get("watts") or []
    hr = streams.get("heartrate") or []
    duration = summary.get("moving_time") or summary.get("elapsed_time") or 0
    distance = summary.get("distance", np.nan)
    avg_hr = float(np.mean(hr)) if hr else summary.get("average_heartrate", np.nan)
    np_w = normalized_power(watts)

    g = (lambda k, d=None: getattr(params, k, d) if not isinstance(params, dict)
         else params.get(k, d))
    ftp = g("FTP")
    lthr = g("LTHR")

    tss = np.nan
    source = None
    if sport == "bike" and watts:
        tss, source = tss_power(duration, np_w, ftp), "power"
    elif sport == "run":
        tss, source = tss_run(duration, distance, g("RUN_THRESHOLD_PACE")), "pace"
    elif sport == "swim":
        tss, source = tss_swim(duration, distance, g("SWIM_THRESHOLD_PACE")), "pace"
    if np.isnan(tss):  # fallback for any sport
        tss, source = tss_hr(duration, avg_hr, lthr), "hr"

    return {
        "id": summary["id"],
        "date": pd.to_datetime(summary["start_date_local"]).normalize(),
        "name": summary.get("name"),
        "type": summary.get("type"),
        "sport": sport,
        "duration_s": duration,
        "distance_m": distance,
        "avg_hr": avg_hr,
        "np_watts": np_w,
        "avg_watts": summary.get("average_watts", np.nan),
       "tss": tss,
        "tss_source": source,
        "device_watts": bool(summary.get("device_watts", False)),
        "is_race": is_race(summary),
    }


def build_activity_table(summaries, get_streams, params) -> pd.DataFrame:
    rows = []
    for s in summaries:
        try:
            streams = get_streams(s["id"])
        except Exception as e:
            print(f"streams failed for {s['id']}: {e}")
            streams = {}
        rows.append(activity_load(s, streams, params))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def race_table(activity_df: pd.DataFrame) -> pd.DataFrame:
    """Activities auto-detected as races (Strava 'Race' tag or name keyword).
    Columns line up with what race_predict.build_race_dataset expects."""
    cols = ["id", "date", "name", "sport", "distance_m", "time_s"]
    if activity_df.empty or "is_race" not in activity_df:
        return pd.DataFrame(columns=cols)
    races = activity_df[activity_df["is_race"]].rename(columns={"duration_s": "time_s"})
    return races[cols].sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Daily timeline with per-sport split + combined total
# ---------------------------------------------------------------------------
def daily_timeline(activity_df: pd.DataFrame) -> pd.DataFrame:
    if activity_df.empty:
        return pd.DataFrame(columns=["date", "tss", "tss_bike", "tss_run",
                                     "tss_swim", "duration_s", "n_activities"])
    total = (activity_df.groupby("date")
             .agg(tss=("tss", "sum"), duration_s=("duration_s", "sum"),
                  n_activities=("id", "count")).reset_index())
    per_sport = (activity_df.pivot_table(index="date", columns="sport",
                 values="tss", aggfunc="sum", fill_value=0).reset_index())
    per_sport.columns = ["date"] + [f"tss_{c}" for c in per_sport.columns[1:]]
    daily = total.merge(per_sport, on="date", how="left")

    full = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full, fill_value=0)
    daily = daily.rename_axis("date").reset_index()
    for c in ["tss_bike", "tss_run", "tss_swim"]:
        if c not in daily.columns:
            daily[c] = 0.0
    return daily
