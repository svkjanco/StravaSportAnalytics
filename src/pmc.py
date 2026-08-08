"""
Performance Management Chart (PMC): the fitness / fatigue / form model.

  CTL (Chronic Training Load) = ~42-day exponentially weighted avg of daily TSS  -> "fitness"
  ATL (Acute Training Load)   = ~7-day  exponentially weighted avg of daily TSS  -> "fatigue"
  TSB (Training Stress Balance)= yesterday's CTL - yesterday's ATL               -> "form"

This is the workhorse output #2 (fitness/form trend). It's a deterministic
sports-science model, not ML - which is exactly why it's reliable on one
athlete's data.
"""
import numpy as np
import pandas as pd


def _ewma(series: pd.Series, time_constant: float) -> pd.Series:
    """Exponentially weighted moving average with a day-based time constant."""
    alpha = 1.0 - np.exp(-1.0 / time_constant)
    return series.ewm(alpha=alpha, adjust=False).mean()


def compute_pmc(daily: pd.DataFrame, ctl_tc: int = 42, atl_tc: int = 7) -> pd.DataFrame:
    """Input: daily timeline with columns [date, tss]. Output adds ctl/atl/tsb."""
    df = daily.copy().sort_values("date").reset_index(drop=True)
    df["ctl"] = _ewma(df["tss"], ctl_tc)
    df["atl"] = _ewma(df["tss"], atl_tc)
    # form is based on *yesterday's* values (standard TrainingPeaks convention)
    df["tsb"] = (df["ctl"].shift(1) - df["atl"].shift(1)).fillna(0)
    return df


def project_pmc(daily_pmc: pd.DataFrame, future_daily_tss: dict,
                ctl_tc: int = 42, atl_tc: int = 7) -> pd.DataFrame:
    """Project CTL/ATL/TSB forward given a planned {date: tss} schedule.
    Lets you ask 'what will my form be on race day if I train like X?'."""
    df = daily_pmc.copy()
    last = df.iloc[-1]
    a_ctl = 1.0 - np.exp(-1.0 / ctl_tc)
    a_atl = 1.0 - np.exp(-1.0 / atl_tc)

    ctl, atl = last["ctl"], last["atl"]
    rows = []
    for date in pd.to_datetime(sorted(future_daily_tss)):
        prev_ctl, prev_atl = ctl, atl
        tss = future_daily_tss[date]
        ctl = ctl + a_ctl * (tss - ctl)
        atl = atl + a_atl * (tss - atl)
        rows.append({"date": date, "tss": tss, "ctl": ctl, "atl": atl,
                     "tsb": prev_ctl - prev_atl})
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
