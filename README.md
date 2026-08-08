# Strava ML / analytics

Personal training analytics from your Strava data. Four outputs:

| # | Output | Approach | Why |
|---|--------|----------|-----|
| 1 | Race-time prediction | Riegel baseline + Ridge residual on training state | Data-light, doesn't overfit on few races |
| 2 | Fitness / fatigue / form | PMC (CTL/ATL/TSB), Banister-style EWMA | Deterministic, interpretable |
| 3 | FTP / power tracking | Mean-max power curve + Critical Power fit | No formal test needed |
| 4 | Fatigue / overtraining alerts | Aerobic decoupling + IsolationForest | Unsupervised, learns *your* normal |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.py config.py    # then fill in CLIENT_ID / CLIENT_SECRET / FTP / LTHR
```

Create a Strava API app at https://www.strava.com/settings/api
(Authorization Callback Domain: `localhost`).

## First-run OAuth (once)
In `main.py`, uncomment the auth cell:
1. `print(sc.authorize_url(config.CLIENT_ID))` -> open the URL, approve.
2. Browser redirects to `http://localhost/exchange_token?...&code=XXXX...`
   (it 404s - that's fine). Copy the `code` value into `config.AUTH_CODE`.
3. Run `sc.exchange_code(...)`. Token is saved to `cache/token.json` and
   auto-refreshes after that; you won't repeat this.

## Run
Open `main.py` in VS Code (the `# %%` markers make it an interactive notebook),
or `python main.py`. Streams are cached under `cache/streams/`, so re-runs are
cheap and won't hit the rate limit (100 req / 15 min, 1000 / day).

## Notes on the modelling
- With one athlete's data, classic sports-science models beat black-box ML on
  accuracy and interpretability - that's why #2 and #3 are formulas, not nets.
- ML is used where it adds value: #1 (learned correction) and #4 (unsupervised
  anomalies). Both degrade gracefully with little data.
- TSS uses power when available, else an HR fallback. Keep `FTP`/`LTHR` current.
- Race prediction needs you to supply labelled races (date/distance/time).
```
