# %% [markdown]
# # Strava ML/analytics pipeline (multisport)
# Cells (# %%) run interactively in VS Code, or run as a plain script.

# %% Imports & config
import matplotlib.pyplot as plt

from src import strava_client as sc
from src import features, pmc, power_curve, race_predict, anomaly
from src import tri_predict, intervals

try:
    import config
except ImportError:
    raise SystemExit("Copy config.example.py -> config.py and fill in your keys.")

# %% First-run auth -> uz vybavene cez auth.py, necha sa vypnute
# print(sc.authorize_url(config.CLIENT_ID))
# sc.exchange_code(config.CLIENT_ID, config.CLIENT_SECRET, config.AUTH_CODE)

client = sc.StravaClient(config.CLIENT_ID, config.CLIENT_SECRET)
streams_lookup = client.get_streams

# %% Pull + feature engineering (per-sport TSS)
summaries = client.list_activities()
act = features.build_activity_table(summaries, streams_lookup, config)
daily = features.daily_timeline(act)
print(act.groupby("sport")[["tss"]].agg(["count", "mean"]))

# %% PMC (combined form across all sports) + per-sport load
pmc_df = pmc.compute_pmc(daily)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(pmc_df.date, pmc_df.ctl, label="CTL (fitness)")
ax.plot(pmc_df.date, pmc_df.atl, label="ATL (fatigue)", alpha=.7)
ax.fill_between(pmc_df.date, pmc_df.tsb, 0, alpha=.2, label="TSB (form)")
ax.legend(); ax.set_title("PMC"); plt.tight_layout()
plt.savefig("cache/pmc.png", dpi=110)

# %% FTP / CP tracking
ftp_tl = power_curve.ftp_timeline(act[act["device_watts"]], streams_lookup, step_days=14, window_days=90)
print(ftp_tl[["date", "cp", "w_prime", "ftp_est"]].tail())

# %% Single-sport running race prediction (Riegel + ML residual)
# Races are auto-detected from Strava's "Race" tag (workout_type) or a
# name containing race/pretek/zavod - tag your race activities in Strava
# and they'll show up here without manual entry.
races = features.race_table(act)
print(races[["date", "name", "sport", "distance_m", "time_s"]])

run_races = races[races["sport"] == "run"]
predictor = race_predict.RacePredictor()
if not run_races.empty:
    predictor.fit(race_predict.build_race_dataset(run_races, pmc_df))
# print(predictor.predict(3000, 10000, 21097, pmc_df, "2026-05-01"))

# %% Triathlon / Ironman SPLIT prediction (discipline-specific)
im = tri_predict.predict_triathlon(
    ftp=config.FTP,
    run_threshold_pace_s_per_km=config.RUN_THRESHOLD_PACE,
    swim_css_s_per_100m=config.SWIM_THRESHOLD_PACE,
    bike_physics=config.BIKE_PHYSICS,
    race="ironman",           # or "70.3", "olympic", "sprint"
)
print("IM:", {k: im[k] for k in ["swim", "bike", "run", "total", "bike_kmh"]})

# %% Interval detection + workout classification
wtype = intervals.classify_history(
    act, streams_lookup, ftp=config.FTP,
    run_thr_speed_mps=1000 / config.RUN_THRESHOLD_PACE,  # m/s at threshold
)
print(wtype["session_type"].value_counts())

# %% Fatigue / overtraining anomalies
feat = anomaly.build_daily_features(pmc_df, act, streams_lookup)
flagged = anomaly.detect_anomalies(feat)
print(flagged[flagged.anomaly][["date", "tsb", "ramp", "decoupling"]].tail())
