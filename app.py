"""
Strava analytics GUI (Streamlit).  Run from project root:  streamlit run app.py
"""
import datetime as dt
import io
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import features, pmc, power_curve, intervals, tri_predict, race_predict, hr_zones
from src import strava_client as sc

CACHE = Path("cache")
STREAMS = CACHE / "streams"
ACT_CACHE = CACHE / "activities.json"
st.set_page_config(page_title="Strava analytics", layout="wide")

# ---- consistent chart style ----
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#cfcfcf", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": "#222",
    "axes.labelcolor": "#555", "xtick.color": "#666", "ytick.color": "#666",
    "legend.frameon": False,
})
COL = {"bike": "#2c7fb8", "run": "#e6550d", "swim": "#31a354",
       "ctl": "#08519c", "atl": "#f16913", "pos": "#31a354", "neg": "#de2d26",
       "ftp": "#6a51a3", "accent": "#756bb1"}


def datefmt(ax):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def range_chart(fig, title, yaxis_title, height=440):
    """Shared look + a draggable zoom/range-select strip for a time-series figure.
    Title sits on its own top row; legend (left) and range buttons (right) share
    the row below it, so nothing overlaps regardless of light/dark theme."""
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top",
                  font=dict(size=15)),
        yaxis_title=yaxis_title, height=height, hovermode="x unified",
        margin=dict(l=10, r=10, t=95, b=10),
        legend=dict(orientation="h", y=1.12, yanchor="bottom", x=0, xanchor="left",
                   font=dict(size=11)),
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.09,
                             bgcolor="rgba(128,128,128,0.15)",
                             bordercolor="rgba(128,128,128,0.5)", borderwidth=1),
            rangeselector=dict(
                y=1.12, yanchor="bottom", x=1, xanchor="right",
                bgcolor="rgba(128,128,128,0.25)", activecolor="#4c78a8",
                bordercolor="rgba(128,128,128,0.5)", borderwidth=1,
                font=dict(size=11),
                buttons=[
                    dict(count=42, label="6t", step="day", stepmode="backward"),
                    dict(count=91, label="3m", step="day", stepmode="backward"),
                    dict(count=365, label="1r", step="day", stepmode="backward"),
                    dict(step="all", label="Vsetko"),
                ],
            ),
        ),
    )
    return fig


def disk_streams(activity_id) -> dict:
    f = STREAMS / f"{activity_id}.json"
    return json.loads(f.read_text()) if f.exists() else {}


@st.cache_data(show_spinner="Nacitavam aktivity...")
def load_data(_refresh_key: int = 0):
    if ACT_CACHE.exists() and _refresh_key == 0:
        summaries = json.loads(ACT_CACHE.read_text(encoding="utf-8"))
    else:
        client = sc.StravaClient(config.CLIENT_ID, config.CLIENT_SECRET)
        summaries = client.list_activities()
        ACT_CACHE.write_text(json.dumps(summaries), encoding="utf-8")
    act = features.build_activity_table(summaries, disk_streams, config)
    daily = features.daily_timeline(act)
    act["date"] = pd.to_datetime(act["date"]).dt.tz_localize(None)
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)
    power_ids = {int(a["id"]) for a in summaries if a.get("device_watts")}
    return act, daily, power_ids


@st.cache_data(show_spinner="Analyzujem power krivky (raz)...")
def per_activity_curves(_act, _power_ids):
    out = {}
    for _, r in _act.iterrows():
        if r["sport"] != "bike" or int(r["id"]) not in _power_ids:
            continue
        w = disk_streams(r["id"]).get("watts") or []
        if w:
            out[int(r["id"])] = (pd.Timestamp(r["date"]), power_curve.best_efforts(w))
    return out


def ftp_from_curves(curves, step_days=14, window_days=90) -> pd.DataFrame:
    if not curves:
        return pd.DataFrame(columns=["date", "cp", "w_prime", "ftp_est"])
    all_dates = [d for d, _ in curves.values()]
    dates = pd.date_range(min(all_dates) + pd.Timedelta(days=window_days),
                          max(all_dates), freq=f"{step_days}D")
    rows = []
    for as_of in dates:
        start = as_of - pd.Timedelta(days=window_days)
        env = {}
        for d, be in curves.values():
            if start < d <= as_of:
                for dur, p in be.items():
                    env[dur] = max(env.get(dur, 0.0), p)
        rows.append({"date": as_of, **power_curve.fit_cp(env)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Klasifikujem treningy...")
def load_intervals(_act):
    run_thr = 1000.0 / config.RUN_THRESHOLD_PACE
    return intervals.classify_history(_act, disk_streams, ftp=config.FTP,
                                      run_thr_speed_mps=run_thr)


@st.cache_data(show_spinner="Pocitam HR zony...")
def load_zones(_act):
    return hr_zones.build_zone_table(_act, disk_streams, config.LTHR)


PB_DISTS = {"1 km": 1000, "5 km": 5000, "10 km": 10000, "Polmaraton": 21097}
SWIM_PB_DISTS = {"100 m": 100, "200 m": 200, "400 m": 400, "800 m": 800, "1500 m": 1500}
# Pool-swim distance streams jump a full lap length (~25m) in a single sample at
# each wall turn, so a naive sliding-window best-time can "see" a burst of several
# such jumps as an impossibly fast split. Anything faster than this is that
# artifact, not a real swim - discard it.
MIN_SWIM_PACE_100M_S = 60


def _best_time(dist, tim, target):
    dist = np.asarray(dist, float); tim = np.asarray(tim, float)
    n = len(dist); i = 0; best = np.inf
    for j in range(n):
        while i < j and dist[j] - dist[i] >= target:
            best = min(best, tim[j] - tim[i]); i += 1
    return None if best == np.inf else best


def sport_bests(_act, sport, dists):
    out = {}
    for _, r in _act[_act["sport"] == sport].iterrows():
        s = disk_streams(r["id"])
        d, t = s.get("distance"), s.get("time")
        if not d or not t or len(d) != len(t) or len(d) < 10:
            continue
        pbs = {}
        for label, dist in dists.items():
            if d[-1] >= dist:
                bt = _best_time(d, t, dist)
                if bt:
                    pbs[label] = bt
        if pbs:
            out[int(r["id"])] = (pd.Timestamp(r["date"]), pbs)
    return out


@st.cache_data(show_spinner="Hladam behove osobaky (raz)...")
def run_bests(_act):
    return sport_bests(_act, "run", PB_DISTS)


@st.cache_data(show_spinner="Hladam plavecke osobaky (raz)...")
def swim_bests(_act):
    out = sport_bests(_act, "swim", SWIM_PB_DISTS)
    cleaned = {}
    for aid, (date, pbs) in out.items():
        kept = {lab: s for lab, s in pbs.items()
               if s >= SWIM_PB_DISTS[lab] / 100 * MIN_SWIM_PACE_100M_S}
        if kept:
            cleaned[aid] = (date, kept)
    return cleaned


@st.cache_data(show_spinner="Pocitam plavecke tempo v case...")
def swim_pace_timeline(_act, ref_dist=400, step_days=14, window_days=90):
    """Best pace/100m for a reference distance, rolling window - obdoba FTP timeline."""
    efforts = []
    for _, r in _act[_act["sport"] == "swim"].iterrows():
        s = disk_streams(r["id"])
        d, t = s.get("distance"), s.get("time")
        if not d or not t or len(d) != len(t) or d[-1] < ref_dist:
            continue
        bt = _best_time(d, t, ref_dist)
        if bt and bt >= ref_dist / 100 * MIN_SWIM_PACE_100M_S:
            efforts.append((pd.Timestamp(r["date"]), bt))
    if not efforts:
        return pd.DataFrame(columns=["date", "pace_100m_s"])
    dates = pd.date_range(min(d for d, _ in efforts) + pd.Timedelta(days=window_days),
                          max(d for d, _ in efforts), freq=f"{step_days}D")
    rows = []
    for as_of in dates:
        start = as_of - pd.Timedelta(days=window_days)
        window = [bt for d0, bt in efforts if start < d0 <= as_of]
        if window:
            rows.append({"date": as_of, "pace_100m_s": min(window) / (ref_dist / 100)})
    return pd.DataFrame(rows)


def hms(sec):
    sec = int(round(sec)); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def pace_str(s_per_unit, unit="km"):
    s = int(round(s_per_unit)); m, s = divmod(s, 60)
    return f"{m}:{s:02d}/{unit}"


# ---------------------------------------------------------------------------
if "refresh" not in st.session_state:
    st.session_state.refresh = 0
act, daily, power_ids = load_data(st.session_state.refresh)
curves = per_activity_curves(act, power_ids)
ftp_tl = ftp_from_curves(curves)
min_d, max_d = daily["date"].min().date(), daily["date"].max().date()

st.sidebar.header("Nastavenia")
quick = st.sidebar.radio("Obdobie",
                         ["Vlastne", "6 tyzdnov", "3 mesiace", "Sezona (1 rok)", "Vsetko"],
                         index=3)
if quick == "Vlastne":
    # Streamlit's built-in rychle skratky ("Past week" a pod.) pocitaju od
    # skutocneho dnesneho datumu, nie od poslednej aktivity - hranice preto
    # musia siahat aj tam, nielen po data, inak skratka nahlasi chybu.
    today = dt.date.today()
    picked = st.sidebar.date_input(
        "Vyber rozsah", value=(min_d, max_d),
        min_value=min(min_d, today - dt.timedelta(days=366)),
        max_value=max(max_d, today) + dt.timedelta(days=1),
        format="YYYY-MM-DD")
    if isinstance(picked, (tuple, list)):
        if len(picked) == 2:
            start_d, end_d = picked
        elif len(picked) == 1:
            start_d = end_d = picked[0]  # koniec este nevybrany, cakaj na druhy klik
        else:
            start_d, end_d = min_d, max_d
    else:
        start_d, end_d = picked, picked
else:
    end_d = max_d
    span = {"6 tyzdnov": 42, "3 mesiace": 91, "Sezona (1 rok)": 365,
            "Vsetko": (max_d - min_d).days}[quick]
    start_d = max(min_d, end_d - dt.timedelta(days=span))
st.sidebar.caption(f"Zobrazujem {start_d} -> {end_d}")
st.sidebar.divider()
ctl_tc = st.sidebar.slider("CTL casova konstanta (dni)", 28, 56, 42,
                           help="Ako dlho sa 'pamata' fitnes.")
atl_tc = st.sidebar.slider("ATL casova konstanta (dni)", 5, 14, 7,
                           help="Okno unavy.")
st.sidebar.divider()
if st.sidebar.button("Obnovit zo Stravy"):
    st.session_state.refresh += 1
    st.cache_data.clear()
    st.rerun()

start_ts, end_ts = pd.Timestamp(start_d), pd.Timestamp(end_d)
def in_range(df, col="date"):
    return df[(df[col] >= start_ts) & (df[col] <= end_ts)]

pmc_full = pmc.compute_pmc(daily, ctl_tc=ctl_tc, atl_tc=atl_tc)
latest = pmc_full.iloc[-1]

st.title("Strava analytics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("CTL (fitnes)", f"{latest['ctl']:.0f}", help="Kondicia za ~6 tyzdnov.")
c2.metric("ATL (unava)", f"{latest['atl']:.0f}", help="Cerstva unava za ~7 dni.")
c3.metric("TSB (forma)", f"{latest['tsb']:+.0f}",
          help="+5 az +15 svieze, pod -20 unavene.")
if not ftp_tl["ftp_est"].dropna().empty:
    c4.metric("FTP odhad", f"{ftp_tl['ftp_est'].dropna().iloc[-1]:.0f} W")

st.subheader("Posledne aktivity")
recent = act.sort_values("date", ascending=False).head(3)
st.dataframe(pd.DataFrame({
    "datum": recent["date"].dt.date,
    "sport": recent["sport"],
    "nazov": recent["name"],
    "trvanie": recent["duration_s"].map(hms),
    "vzdialenost": recent["distance_m"].apply(
        lambda d: f"{d / 1000:.1f} km" if pd.notna(d) and d > 0 else "-"),
    "TSS": recent["tss"].round(0),
}), use_container_width=True, hide_index=True)

tabs = st.tabs(["Forma", "Zataz", "Beh", "Bike", "Plavanie",
                "Treningy", "Triatlon", "Report"])

# ---- FORMA: aktualny stav (PMC) + planovanie/taper ----
with tabs[0]:
    with st.expander("Co to znamena?"):
        st.markdown(
            "- **CTL (modra)** = kondicia. Chces, aby cez sezonu **stupala**.\n"
            "- **ATL (oranzova)** = unava za posledny tyzden.\n"
            "- **TSB (plocha)**: **zelena nad 0** = oddychnuty, **cervena pod 0** = zatazeny.\n"
            "- Pred pretekom znizis objem, aby TSB **vyskocil do zelena** (taper).\n"
            "- Nizsie **Planovanie / Taper**: naplanuj objem do dna pretekov a uvidis "
            "predpovedanu formu v den D - ten isty PMC model, len projektovany dopredu.")
    v = in_range(pmc_full)
    fig = go.Figure()
    fig.add_scatter(x=v["date"], y=v["tsb"].clip(lower=0), mode="none", fill="tozeroy",
                    fillcolor="rgba(49,163,84,0.22)", name="TSB kladne (svieze)")
    fig.add_scatter(x=v["date"], y=v["tsb"].clip(upper=0), mode="none", fill="tozeroy",
                    fillcolor="rgba(222,45,38,0.18)", name="TSB zaporne (unava)")
    fig.add_scatter(x=v["date"], y=v["ctl"], mode="lines", name="CTL (fitnes)",
                    line=dict(color=COL["ctl"], width=2.6))
    fig.add_scatter(x=v["date"], y=v["atl"], mode="lines", name="ATL (unava)",
                    line=dict(color=COL["atl"], width=1.6))
    fig.add_hline(y=0, line_color="#999", line_width=0.8)
    range_chart(fig, "Performance Management Chart", "TSS/den  |  forma (TSB)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Tip: potiahni po spodnom pruhu (alebo po grafe) a vyber si presny usek, "
              "dvojklik resetuje zoom.")

    st.subheader("Forma k datumu")
    pick = st.date_input("Datum", value=max_d, min_value=min_d, max_value=max_d)
    row = pmc_full[pmc_full["date"] <= pd.Timestamp(pick)].tail(1)
    if not row.empty:
        r = row.iloc[0]
        k1, k2, k3 = st.columns(3)
        k1.metric("CTL", f"{r['ctl']:.0f}")
        k2.metric("ATL", f"{r['atl']:.0f}")
        k3.metric("TSB", f"{r['tsb']:+.0f}")
        tsb = r["tsb"]
        verdict = ("sviezy / nabity" if tsb > 5 else
                   "normalny treningovy stav" if tsb > -10 else
                   "produktivna unava (tvrdy blok)" if tsb > -30 else
                   "velka unava - pozor na prepalenie")
        st.info(f"Stav k {pick}: **{verdict}**")

    st.divider()
    st.subheader("Planovanie / Taper pred pretekom")
    race_date = st.date_input("Datum preteku", value=max_d + dt.timedelta(days=56),
                              min_value=max_d + dt.timedelta(days=1))
    recent_weekly_tss = daily[daily["date"] > pd.Timestamp(max_d) - pd.Timedelta(days=14)]["tss"].sum() / 2
    default_weekly = int(min(1500, max(50, round(recent_weekly_tss))))
    cp1, cp2, cp3 = st.columns(3)
    weekly_tss = cp1.number_input("Planovane tyzdenne TSS (do taperu)", 50, 1500, default_weekly, 10)
    taper_days = cp2.slider("Dlzka taperu (dni)", 3, 21, 10)
    taper_cut = cp3.slider("Znizenie objemu v taperi (%)", 20, 80, 50)

    horizon = pd.date_range(pd.Timestamp(max_d) + pd.Timedelta(days=1), pd.Timestamp(race_date), freq="D")
    taper_start = pd.Timestamp(race_date) - pd.Timedelta(days=taper_days)
    daily_base = weekly_tss / 7.0
    future_tss = {d: (daily_base * (1 - taper_cut / 100.0) if d >= taper_start else daily_base)
                  for d in horizon}
    proj = pmc.project_pmc(pmc_full, future_tss, ctl_tc=ctl_tc, atl_tc=atl_tc)
    hist = proj[proj["date"] <= pd.Timestamp(max_d)].tail(90)
    fut = proj[proj["date"] > pd.Timestamp(max_d)]

    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["ctl"], mode="lines", name="CTL (doteraz)",
                    line=dict(color=COL["ctl"], width=2.2))
    fig.add_scatter(x=hist["date"], y=hist["atl"], mode="lines", name="ATL (doteraz)",
                    line=dict(color=COL["atl"], width=1.3))
    fig.add_scatter(x=fut["date"], y=fut["ctl"], mode="lines", name="CTL (plan)",
                    line=dict(color=COL["ctl"], width=2.2, dash="dash"))
    fig.add_scatter(x=fut["date"], y=fut["atl"], mode="lines", name="ATL (plan)",
                    line=dict(color=COL["atl"], width=1.3, dash="dash"))
    fig.add_scatter(x=fut["date"], y=fut["tsb"], mode="none", fill="tozeroy",
                    fillcolor="rgba(117,107,177,0.2)", name="TSB (plan)")
    fig.add_vline(x=pd.Timestamp(max_d), line_dash="dot", line_color="#999")
    fig.add_vline(x=pd.Timestamp(race_date), line_color="#333", line_width=1.3)
    range_chart(fig, "Projekcia formy do dna pretekov", "TSS/den  |  forma (TSB)")
    st.plotly_chart(fig, use_container_width=True)

    race_row = proj[proj["date"] == pd.Timestamp(race_date)]
    if not race_row.empty:
        r = race_row.iloc[0]
        tsb = r["tsb"]
        verdict = ("skvela forma na pretek" if tsb > 15 else
                   "dobra forma" if tsb > 5 else
                   "OK, ale nie uplne oddychnuty" if tsb > -5 else
                   "pravdepodobne este unaveny - zvaz dlhsi/hlbsi taper")
        k1, k2, k3 = st.columns(3)
        k1.metric("CTL v den D", f"{r['ctl']:.0f}")
        k2.metric("ATL v den D", f"{r['atl']:.0f}")
        k3.metric("TSB v den D", f"{tsb:+.0f}")
        st.info(f"Predpoved formy k {race_date}: **{verdict}**")

# ---- ZATAZ: po sportoch + rok na rok ----
with tabs[1]:
    with st.expander("Co to znamena?"):
        st.markdown("Tyzdenne **TSS** rozdelene na disciplíny (vyssi stlpec = tvrdsi tyzden), "
                    "a nizsie porovnanie s rovnakym obdobim minuly rok.")
    d = in_range(daily).set_index("date")
    order = [c for c in ["tss_bike", "tss_run", "tss_swim"] if c in d.columns]
    weekly = d[order].resample("W").sum()
    names = {"tss_bike": ("Bike", COL["bike"]), "tss_run": ("Beh", COL["run"]),
             "tss_swim": ("Plavanie", COL["swim"])}
    fig = go.Figure()
    for c in order:
        lab, color = names[c]
        fig.add_bar(x=weekly.index, y=weekly[c].values, name=lab, marker_color=color)
    fig.update_layout(barmode="stack")
    range_chart(fig, "Tyzdenne zatazenie po sportoch", "TSS / tyzden")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(in_range(act).groupby("sport")["tss"].agg(["count", "sum"]).round(0),
                 use_container_width=True)

    st.divider()
    st.subheader("Rok na rok")
    cur = in_range(act)
    prev_start, prev_end = start_ts - pd.DateOffset(years=1), end_ts - pd.DateOffset(years=1)
    prev = act[(act["date"] >= prev_start) & (act["date"] <= prev_end)]
    if prev.empty:
        st.info("Ziadne aktivity z minuleho roka pre toto obdobie.")
    else:
        def _summarize(df):
            return df.groupby("sport").agg(
                tss=("tss", "sum"),
                km=("distance_m", lambda s: s.sum() / 1000),
                hodiny=("duration_s", lambda s: s.sum() / 3600),
                pocet=("id", "count"),
            )
        cur_s, prev_s = _summarize(cur), _summarize(prev)
        sports = sorted(set(cur_s.index) | set(prev_s.index))
        comp = pd.DataFrame({
            "TSS teraz": cur_s.reindex(sports)["tss"].fillna(0),
            "TSS vlani": prev_s.reindex(sports)["tss"].fillna(0),
            "km teraz": cur_s.reindex(sports)["km"].fillna(0),
            "km vlani": prev_s.reindex(sports)["km"].fillna(0),
            "pocet teraz": cur_s.reindex(sports)["pocet"].fillna(0),
            "pocet vlani": prev_s.reindex(sports)["pocet"].fillna(0),
        }).round(0)

        fig = go.Figure()
        fig.add_bar(x=[s.capitalize() for s in sports], y=comp["TSS teraz"],
                   name=str(start_d.year), marker_color=COL["accent"])
        fig.add_bar(x=[s.capitalize() for s in sports], y=comp["TSS vlani"],
                   name=str(prev_start.year), marker_color="#999")
        fig.update_layout(
            barmode="group", height=380,
            title=dict(text="TSS: zvolene obdobie vs. rovnake obdobie vlani",
                      x=0, xanchor="left", font=dict(size=15)),
            yaxis_title="TSS spolu", margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(orientation="h", y=1.1, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(comp, use_container_width=True)

        d_tss = comp["TSS teraz"].sum() - comp["TSS vlani"].sum()
        base = comp["TSS vlani"].sum()
        pct = (d_tss / base * 100) if base else np.nan
        znak = "+" if d_tss >= 0 else ""
        st.caption(f"Celkovy TSS {znak}{d_tss:.0f} ({pct:+.0f}%) oproti rovnakemu obdobiu vlani.")

# ---- BEH ----
with tabs[2]:
    st.subheader("Behove osobaky (z dat)")
    rb = run_bests(act)
    rows = [{"vzdialenost": lab, "cas_s": sec, "date": date}
            for aid, (date, pbs) in rb.items() if start_ts <= date <= end_ts
            for lab, sec in pbs.items()]
    if rows:
        pb_df = pd.DataFrame(rows)
        best = (pb_df.sort_values("cas_s").groupby("vzdialenost").first()
                .reindex(list(PB_DISTS.keys())).dropna())
        show = pd.DataFrame({
            "cas": best["cas_s"].map(hms),
            "tempo": [pace_str(best.loc[i, "cas_s"] / (PB_DISTS[i] / 1000))
                      for i in best.index],
            "datum": best["date"].dt.date,
        })
        st.dataframe(show, use_container_width=True)
    else:
        st.info("Ziadne behy s distance datami v tomto obdobi.")

    st.divider()
    st.subheader("Tyzdenny behovy objem")
    runs = in_range(act[act["sport"] == "run"]).set_index("date")
    if not runs.empty:
        km = (runs["distance_m"].resample("W").sum() / 1000).round(1)
        fig = go.Figure()
        fig.add_bar(x=km.index, y=km.values, marker_color=COL["run"], name="km/tyzden")
        range_chart(fig, "Tyzdenny behovy objem", "km / tyzden", height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Spolu za obdobie: {runs['distance_m'].sum()/1000:.0f} km.")

    st.divider()
    st.subheader("Detekovane preteky")
    st.caption("Zo Strava 'Race' typu aktivity, alebo z nazvu (race / pretek / zavod). "
              "Oznac si preteky priamo v Strave a objavia sa tu bez rucneho zadavania.")
    races_all = features.race_table(act)
    run_races = races_all[races_all["sport"] == "run"]
    if races_all.empty:
        st.info("Ziadne aktivity oznacene ako pretek.")
    else:
        show_races = pd.DataFrame({
            "datum": races_all["date"].dt.date,
            "nazov": races_all["name"],
            "sport": races_all["sport"],
            "km": (races_all["distance_m"] / 1000).round(2),
            "cas": races_all["time_s"].map(hms),
        })
        st.dataframe(show_races, use_container_width=True)

    st.divider()
    st.subheader("Predikcia behovych casov")
    predictor = race_predict.RacePredictor()
    if len(run_races) >= 4:
        predictor.fit(race_predict.build_race_dataset(run_races, pmc_full))
    cc1, cc2 = st.columns(2)
    known_km = cc1.number_input("Vzdialenost (km)", 1.0, 50.0, 10.0, 0.5)
    known_time = cc2.text_input("Cas (mm:ss alebo h:mm:ss)", "40:00")
    try:
        parts = [int(x) for x in known_time.split(":")]
        known_s = parts[0]*60 + parts[1] if len(parts) == 2 else parts[0]*3600 + parts[1]*60 + parts[2]
        dd = {"5 km": 5000, "10 km": 10000, "Polmaraton": 21097, "Maraton": 42195}
        if predictor.fitted:
            res = {lab: predictor.predict(known_s, known_km * 1000, dm, pmc_full, max_d)
                   for lab, dm in dd.items()}
            preds = {lab: r["predicted_s"] for lab, r in res.items()}
            st.caption(f"Riegel + korekcia podla aktualnej formy (CTL/ATL/TSB), "
                      f"natrenovane na {len(run_races)} detekovanych pretekoch.")
        else:
            preds = {lab: race_predict.riegel(known_s, known_km*1000, dm) for lab, dm in dd.items()}
            if len(run_races):
                st.caption(f"Cisty Riegel odhad - na korekciu podla formy treba aspon "
                          f"4 detekovane behove preteky (mas {len(run_races)}).")
        st.table(pd.DataFrame({
            "cas": {k: hms(v) for k, v in preds.items()},
            "tempo": {k: pace_str(preds[k] / (dd[k]/1000)) for k in dd},
        }))
    except Exception:
        st.warning("Zadaj cas ako 40:00 alebo 1:25:00.")

    st.divider()
    st.subheader("Treningove tempa (z tvojho prahu)")
    T = config.RUN_THRESHOLD_PACE
    zones = {"Lahke / regeneracne": 1.30, "Dlhy beh": 1.18, "Maraton": 1.06,
             "Prahove (tempo)": 1.00, "VO2max intervaly": 0.93, "Opakovania / sprinty": 0.90}
    st.table(pd.DataFrame({"tempo": {k: pace_str(T*f) for k, f in zones.items()}}))
    st.caption(f"Odvodene z behoveho prahu {pace_str(T)} (config.py -> RUN_THRESHOLD_PACE).")

# ---- BIKE: FTP / vykon ----
with tabs[3]:
    with st.expander("Co to znamena?"):
        st.markdown("Odhad **FTP** vo wattoch v case (90-dnove okno, len jazdy s merakom). "
                    "Ked krivka **rastie**, si silnejsi na bajku.")
    f = in_range(ftp_tl).dropna(subset=["ftp_est"])
    if f.empty:
        st.info("Malo power dat v tomto obdobi.")
    else:
        fig = go.Figure()
        fig.add_scatter(x=f["date"], y=f["ftp_est"], mode="lines+markers", name="FTP odhad",
                        line=dict(color=COL["ftp"], width=2.2), marker=dict(size=5))
        fig.add_hline(y=config.FTP, line_dash="dash", line_color="#999",
                     annotation_text=f"config FTP ({config.FTP} W)", annotation_position="top left")
        range_chart(fig, "Vyvoj FTP / Critical Power", "Watty")
        st.plotly_chart(fig, use_container_width=True)

# ---- TRENINGY: typy + HR zony ----
with tabs[5]:
    with st.expander("Co to znamena?"):
        st.markdown("Rozdelenie tréningov podla struktury (rovnomerne / prahove / "
                    "VO2max / sprinty), a nizsie rozlozenie casu v HR zonach "
                    "(80/20 pravidlo: prevazna vacsina casu by mala byt v Z1/Z2).")
    wt = load_intervals(act)
    counts = in_range(wt)["session_type"].value_counts().sort_values()
    if counts.empty:
        st.info("Ziadne treningy v tomto obdobi.")
    else:
        fig = go.Figure()
        fig.add_bar(x=counts.values, y=counts.index, orientation="h",
                   marker_color=COL["accent"], text=counts.values, textposition="outside")
        fig.update_layout(
            title=dict(text="Typy treningov", x=0, xanchor="left", font=dict(size=15)),
            xaxis_title="pocet treningov", height=max(260, 40 * len(counts)),
            margin=dict(l=10, r=30, t=50, b=10), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("HR zony (intenzita)")
    zt = load_zones(act)
    zr = in_range(zt)
    zone_colors = ["#31a354", "#78c679", "#fec44f", "#fe9929", "#de2d26"]
    totals = zr[hr_zones.ZONE_NAMES].sum() if not zr.empty else pd.Series(0, index=hr_zones.ZONE_NAMES)
    if totals.sum() == 0:
        st.info("Ziadne HR data v tomto obdobi.")
    else:
        pct = (totals / totals.sum() * 100)
        fig = go.Figure()
        for name, color in zip(hr_zones.ZONE_NAMES, zone_colors):
            fig.add_bar(x=[pct[name]], y=["Rozlozenie"], orientation="h",
                       name=f"{name} ({pct[name]:.0f}%)", marker_color=color)
        fig.update_layout(
            barmode="stack", height=190,
            title=dict(text="Intenzita - rozlozenie casu v zonach",
                      x=0, xanchor="left", font=dict(size=15)),
            xaxis=dict(title="% casu", range=[0, 100]),
            yaxis=dict(visible=False),
            legend=dict(orientation="h", y=-0.4, x=0.5, xanchor="center", font=dict(size=11)),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trend po tyzdnoch")
        w = zr.set_index("date")[hr_zones.ZONE_NAMES].resample("W").sum()
        w_pct = w.div(w.sum(axis=1), axis=0).fillna(0) * 100
        fig2 = go.Figure()
        for name, color in zip(hr_zones.ZONE_NAMES, zone_colors):
            fig2.add_bar(x=w_pct.index, y=w_pct[name].values, name=name, marker_color=color)
        fig2.update_layout(barmode="stack")
        range_chart(fig2, "Vyvoj rozlozenia intenzity", "% casu / tyzden")
        st.plotly_chart(fig2, use_container_width=True)

        easy_pct = pct.get("Z1 regeneracia", 0) + pct.get("Z2 aerobne", 0)
        st.caption(f"Lahka zona (Z1+Z2): {easy_pct:.0f}% casu s HR datami. "
                  f"Polarizovany model odporuca ~80%.")

# ---- TRIATLON ----
with tabs[6]:
    with st.expander("Co to znamena?"):
        st.markdown("Odhad splitov na triatlon. Bike cas cez fyzikalny model (vykon->rychlost). "
                    "Rovinata trat, idealne prevedenie - kopcovita trat bude pomalsia.")
    col_a, col_b = st.columns([1, 2])
    with col_a:
        race = st.selectbox("Pretek", ["ironman", "70.3", "olympic", "sprint"])
        default_ftp = int(ftp_tl["ftp_est"].dropna().iloc[-1]) if not ftp_tl["ftp_est"].dropna().empty else config.FTP
        ftp_in = st.number_input("FTP (W)", 150, 500, default_ftp)
        bike_if = st.slider("Bike intenzita (IF)", 0.60, 0.95,
                            float(tri_predict.PRESETS[race]["bike_if"]), 0.01)
    res = tri_predict.predict_triathlon(
        ftp=ftp_in, run_threshold_pace_s_per_km=config.RUN_THRESHOLD_PACE,
        swim_css_s_per_100m=config.SWIM_THRESHOLD_PACE,
        bike_physics=config.BIKE_PHYSICS, race=race, bike_if=bike_if)
    with col_b:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Plavanie", res["swim"])
        m2.metric("Bike", res["bike"], f"{res['bike_kmh']:.1f} km/h")
        m3.metric("Beh", res["run"])
        m4.metric("SPOLU", res["total"])
        st.caption(f"Bike @ {res['bike_power_w']:.0f} W.")

# ---- PLAVANIE ----
with tabs[4]:
    with st.expander("Co to znamena?"):
        st.markdown(
            "Plavecke osobaky, tyzdenny objem a trend tempa (najlepsie tempo na "
            "referencnu vzdialenost v 90-dnovom okne - obdoba FTP grafu, ukazuje "
            "zlepsovanie sa v bazene).")
    st.subheader("Plavecke osobaky (z dat)")
    sb = swim_bests(act)
    swim_rows = [{"vzdialenost": lab, "cas_s": sec, "date": date}
                for aid, (date, pbs) in sb.items() if start_ts <= date <= end_ts
                for lab, sec in pbs.items()]
    if swim_rows:
        spb_df = pd.DataFrame(swim_rows)
        sbest = (spb_df.sort_values("cas_s").groupby("vzdialenost").first()
                .reindex(list(SWIM_PB_DISTS.keys())).dropna())
        sshow = pd.DataFrame({
            "cas": sbest["cas_s"].map(hms),
            "tempo": [pace_str(sbest.loc[i, "cas_s"] / (SWIM_PB_DISTS[i] / 100), "100m")
                     for i in sbest.index],
            "datum": sbest["date"].dt.date,
        })
        st.dataframe(sshow, use_container_width=True)
    else:
        st.info("Ziadne plavania s distance datami v tomto obdobi.")

    st.divider()
    st.subheader("Tyzdenny plavecky objem")
    pool = in_range(act[act["sport"] == "swim"]).set_index("date")
    if not pool.empty:
        vol = pool["distance_m"].resample("W").sum().round(0)
        fig = go.Figure()
        fig.add_bar(x=vol.index, y=vol.values, marker_color=COL["swim"], name="m/tyzden")
        range_chart(fig, "Tyzdenny plavecky objem", "m / tyzden", height=340)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Spolu za obdobie: {pool['distance_m'].sum():.0f} m.")
    else:
        st.info("Ziadne plavania v tomto obdobi.")

    st.divider()
    st.subheader("Trend tempa (400 m)")
    spt = in_range(swim_pace_timeline(act))
    if spt.empty:
        st.info("Malo dat na trend - treba viac plavani nad 400 m.")
    else:
        fig2 = go.Figure()
        fig2.add_scatter(x=spt["date"], y=spt["pace_100m_s"], mode="lines+markers",
                         name="Tempo /100m", line=dict(color=COL["swim"], width=2.2),
                         marker=dict(size=5))
        range_chart(fig2, "Vyvoj plaveckeho tempa (najlepsie 400 m v 90-dnovom okne)",
                   "s / 100m", height=380)
        fig2.update_yaxes(autorange="reversed")  # nahor = rychlejsie tempo (zlepsenie)
        st.plotly_chart(fig2, use_container_width=True)

# ---- REPORT ----
with tabs[7]:
    with st.expander("Co to znamena?"):
        st.markdown("Jednostrankovy suhrn poslednych 7 dni na stiahnutie - "
                    "napr. na poslanie trenerovi alebo do denniku.")
    week_start = pd.Timestamp(max_d) - pd.Timedelta(days=6)
    wk_act = act[act["date"] >= week_start]
    wk_daily = daily[daily["date"] >= week_start]
    latest_row = pmc_full.iloc[-1]

    fig = plt.figure(figsize=(8.5, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 1.1, 1.3, 1.5], hspace=0.6)

    ax_title = fig.add_subplot(gs[0]); ax_title.axis("off")
    ax_title.text(0, 0.7, "Tyzdenny report", fontsize=20, weight="bold")
    ax_title.text(0, 0.15, f"{week_start.date()} - {max_d}", fontsize=11, color="#555")

    ax_m = fig.add_subplot(gs[1]); ax_m.axis("off")
    metrics_txt = (
        f"CTL (fitnes):    {latest_row['ctl']:.0f}\n"
        f"ATL (unava):     {latest_row['atl']:.0f}\n"
        f"TSB (forma):     {latest_row['tsb']:+.0f}\n"
        f"TSS za tyzden:   {wk_daily['tss'].sum():.0f}\n"
        f"Pocet treningov: {len(wk_act)}\n"
        f"Cas spolu:       {wk_act['duration_s'].sum() / 3600:.1f} h"
    )
    ax_m.text(0, 1, metrics_txt, fontsize=12, va="top", family="monospace")

    ax_pmc = fig.add_subplot(gs[2])
    v90 = pmc_full[pmc_full["date"] >= pd.Timestamp(max_d) - pd.Timedelta(days=90)]
    ax_pmc.plot(v90["date"], v90["ctl"], color=COL["ctl"], label="CTL")
    ax_pmc.plot(v90["date"], v90["atl"], color=COL["atl"], label="ATL")
    ax_pmc.axvline(mdates.date2num(week_start), color="#999", ls=":", lw=.8)
    ax_pmc.set_title("Forma (90 dni)"); ax_pmc.legend(fontsize=8); datefmt(ax_pmc)

    ax_sport = fig.add_subplot(gs[3])
    by_sport = wk_act.groupby("sport")["tss"].sum()
    if not by_sport.empty:
        colors_s = [COL.get(s, "#999") for s in by_sport.index]
        ax_sport.bar(by_sport.index, by_sport.to_numpy(), color=colors_s)
    ax_sport.set_title("TSS po sportoch (tento tyzden)")
    fig.tight_layout()

    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    st.download_button("Stiahnut report (PNG)", data=buf.getvalue(),
                       file_name=f"strava_report_{max_d}.png", mime="image/png")