"""XGBoost NASCAR Predictor: every car's chances in the next Cup race.

Run:  streamlit run app.py

The next race is found on NASCAR's published schedule. The page starts from the
forecast race_train.py saved; "Refresh from NASCAR" re-reads the entry list (and the
starting grid once qualifying is done) and re-simulates the race live.

Tabs: Race forecast · Driver view · How good is it? · Data
"""
import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import race_model as rm
from race_data import entry_rows, next_race, upcoming_rows

# Chart palette (validated for colour-blind separation on both the light and dark
# surfaces; see docs/app.md). ACCENT is a red-orange that sits with the blue livery but
# stays clearly apart from the DNF red.
BLUE, ACCENT, MUTED, INK = "#2a78d6", "#e45d2b", "#898781", "#0b0b0b"
POS, NEG = BLUE, ACCENT          # diverging poles for SHAP contributions
CRITICAL = "#d03b3b"             # DNF only, always paired with an x marker + label
GRID = "rgba(137,135,129,0.25)"

LINKEDIN_URL = "https://www.linkedin.com/in/tony-honesto-4195023"
GITHUB_URL = "https://github.com/atonyhonesto"

# Page styling: a deep royal-blue livery with bright blue, sky blue, white and a red accent,
# plus slanted racing type. Theme colours for Streamlit's own widgets live in
# .streamlit/config.toml. Everything here uses translucent fills or fixed-colour blocks, so it
# reads in both light and dark mode.
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:ital,wght@1,600;1,700;1,800&display=swap');
:root { --navy: #0a2463; --royal: #2f6fe0; --sky: #7cc7ff; --red: #e03a3a; --night: #0d1320;
        --race: 'Barlow Condensed', 'Arial Narrow', sans-serif; }

/* Banner */
.hero { position: relative; overflow: hidden; border-radius: 6px; padding: 22px 28px 30px;
        background: linear-gradient(100deg, #061538 0%, var(--navy) 45%, #1f56c9 100%); }
.hero::after { content: ""; position: absolute; top: 0; right: -60px; width: min(260px, 38%); height: 100%;
               background: repeating-linear-gradient(90deg, #fff 0 12px, transparent 12px 22px,
                                                     var(--red) 22px 30px, transparent 30px 42px);
               opacity: .28; transform: skewX(-24deg); }
.hero .name { font-family: var(--race); font-style: italic; font-weight: 800; letter-spacing: .5px;
              line-height: 1; font-size: clamp(2rem, 5vw, 3.2rem); color: #fff; }
.hero .name span { color: var(--sky); }
.hero .stripe { height: 4px; width: 130px; margin-top: 10px; background: var(--red); transform: skewX(-24deg);
                box-shadow: 0 5px 0 #fff; }
.hero .by { margin-top: 14px; font-size: .95rem; color: #dbe8ff; }
.hero .by a { color: var(--sky); font-weight: 600; text-decoration: none; }
.hero .by a:hover { text-decoration: underline; }
.hero .checker { position: absolute; left: 0; right: 0; bottom: 0; height: 10px; opacity: .45;
                 background: conic-gradient(#fff 25%, var(--night) 0 50%, #fff 0 75%, var(--night) 0) 0 0 / 10px 10px; }

/* Section headings: slanted racing type behind a red-to-blue flag */
[data-testid="stMain"] h3 { font-family: var(--race); font-style: italic; font-weight: 800; letter-spacing: .3px; }
[data-testid="stMain"] h3::before { content: ""; display: inline-block; width: 7px; height: .85em; margin-right: 10px;
    vertical-align: -.08em; background: linear-gradient(180deg, var(--red), var(--royal)); transform: skewX(-18deg); }

/* Tabs */
[data-testid="stTab"] p { font-family: var(--race); font-style: italic; font-weight: 700;
    font-size: 1.15rem; text-transform: uppercase; letter-spacing: .6px; }
@media (max-width: 640px) { [data-testid="stTab"] p { font-size: .95rem; letter-spacing: .2px; } }

/* Panels around charts and tables: a livery stripe across the top */
[class*="st-key-panel_"] { position: relative; overflow: hidden; }
[class*="st-key-panel_"]::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 4px;
    background: linear-gradient(90deg, var(--navy), var(--royal) 55%, var(--sky)); }

/* The hook under the banner, and the "How it works" button beside it */
.hook { font-size: 1.05rem; line-height: 1.45; padding: 4px 0 2px; }
.hook b { font-family: var(--race); font-style: italic; font-weight: 800; font-size: 1.35rem; margin-right: 4px; }
/* Navy with sky-blue type in every state (Streamlit greys it out while the panel is open) */
button[data-testid="stPopoverButton"], button[data-testid="stPopoverButton"]:focus,
button[data-testid="stPopoverButton"]:active, button[data-testid="stPopoverButton"][aria-expanded="true"] {
    background: var(--navy) !important; color: var(--sky) !important; border: 2px solid var(--navy) !important; }
button[data-testid="stPopoverButton"] p { font-family: var(--race); font-style: italic; font-weight: 700;
    font-size: 1.05rem; text-transform: uppercase; letter-spacing: .8px; }
button[data-testid="stPopoverButton"]:hover, button[data-testid="stPopoverButton"][aria-expanded="true"] {
    border-color: var(--red) !important; }

/* Tech badges */
.tech { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 4px; }
.tech span { font-family: var(--race); font-style: italic; font-weight: 700; letter-spacing: .5px;
             text-transform: uppercase; font-size: .95rem; padding: 2px 10px; border-radius: 3px;
             border: 1px solid rgba(47,111,224,.55); }

/* Stat tiles */
[data-testid="stMetric"] { border-left: 5px solid var(--royal) !important; }
[data-testid="stMetricValue"] { font-family: var(--race); font-style: italic; font-weight: 700; }
[data-testid="stMetricLabel"] p { text-transform: uppercase; letter-spacing: .6px; font-size: .78rem; font-weight: 600; }

/* Race info chips */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 2px 0 6px; }
.chip { font-size: .85rem; padding: 3px 11px; border-radius: 999px; white-space: nowrap;
        background: rgba(47,111,224,.10); border: 1px solid rgba(47,111,224,.40); }
.chip.sky { background: rgba(124,199,255,.20); border-color: rgba(124,199,255,.75); }
.chip.hot { color: #fff; border: none; font-weight: 600;
            background: linear-gradient(90deg, var(--navy), var(--red)); }

/* Driver card */
.dcard { display: flex; align-items: center; gap: 18px; margin: 6px 0 14px; flex-wrap: wrap; }
.dnum { font-family: var(--race); font-style: italic; font-weight: 800; font-size: 2.6rem; line-height: 1;
        color: #fff; padding: 6px 20px 4px; border-radius: 4px; border-bottom: 4px solid var(--red);
        background: linear-gradient(120deg, var(--navy), #1f56c9); transform: skewX(-12deg); }
.dnum span { display: inline-block; transform: skewX(12deg); }
.dname { font-family: var(--race); font-style: italic; font-weight: 800; font-size: 2rem; line-height: 1.05; }
.dmeta { font-size: .9rem; opacity: .75; margin-top: 2px; }

/* Footer */
.foot { margin-top: 36px; padding-top: 14px; font-size: .85rem; position: relative;
        display: flex; flex-wrap: wrap; gap: 10px 24px; justify-content: space-between; align-items: center; }
.foot::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 6px; opacity: .5;
                background: conic-gradient(#898781 25%, transparent 0 50%, #898781 0 75%, transparent 0) 0 0 / 6px 6px; }
.foot a { color: var(--royal); font-weight: 600; text-decoration: none; }
.foot .stack span { font-family: var(--race); font-style: italic; font-weight: 700; letter-spacing: .4px;
                    padding: 2px 9px; margin-right: 6px; border-radius: 3px; border: 1px solid rgba(137,135,129,.45); }
.foot .note { opacity: .65; flex-basis: 100%; }
</style>
"""

HERO = """
<div class="hero">
  <div class="name"><span>XGBoost</span> NASCAR Predictor</div>
  <div class="stripe"></div>
  <div class="by">Built by <b>Tony Honesto</b> &middot;
    <a href="{linkedin}" target="_blank">LinkedIn</a> &middot;
    <a href="{github}" target="_blank">GitHub</a></div>
  <div class="checker"></div>
</div>
"""

FOOTER = """
<div class="foot">
  <div>Built by <b>Tony Honesto</b> &middot; <a href="{linkedin}" target="_blank">LinkedIn</a> &middot;
       <a href="{github}" target="_blank">GitHub</a></div>
  <div class="stack"><span>XGBoost</span><span>Monte Carlo</span><span>SHAP</span><span>Plotly</span><span>Streamlit</span></div>
  <div class="note">An independent fan project using NASCAR's public data feeds. Not affiliated with NASCAR,
       any team or any sponsor. Forecasts are for fun, not betting advice.</div>
</div>
"""

TYPE_LABELS = {"short": "short track", "intermediate": "intermediate oval",
               "superspeedway": "superspeedway", "road": "road course"}
FEATURE_NAMES = {
    "start_pct_field": "Starting spot (share of field)",
    "qual_gap_to_pole_pct": "Qualifying gap to pole (%)",
    "track_type": "Kind of track",
    "track_miles": "Track length (miles)",
    "make": "Manufacturer",
    "playoff": "Playoff race",
    "career_starts": "Career Cup starts (since 2018)",
    "form_fin": "Last 5 races: average finish",
    "form_rating": "Last 5 races: driver rating",
    "form_run_pos": "Last 5 races: running position",
    "form_top10": "Last 5 races: top-10 rate",
    "form_dnf": "Last 10 races: DNF rate",
    "season_rating": "This season: driver rating",
    "trk_starts": "Starts at this track",
    "trk_mean_fin": "This track: average finish",
    "trk_mean_rating": "This track: driver rating",
    "trk_top10_rate": "This track: top-10 rate",
    "type_mean_fin": "Similar tracks: average finish",
    "type_mean_rating": "Similar tracks: driver rating",
    "type_top15_laps": "Similar tracks: laps in top 15",
    "team_form_fin": "Team, last 5 races: average finish",
    "team_form_rating": "Team, last 5 races: driver rating",
}

st.set_page_config(page_title="XGBoost NASCAR Predictor", page_icon="🏁", layout="wide")


# ---------------------------------------------------------------- data

def data_version():
    """When the data, models or forecast last changed on disk.

    The loaders below take this as an argument, so their caches refresh by themselves when
    a push brings new files. Without it, the live app kept serving the previous race's
    numbers until it was rebooted.
    """
    files = (rm.TABLE, rm.FOLDER / "meta.json", rm.REPORTS / "race_next.json", rm.REPORTS / "race_metrics.csv")
    return max((f.stat().st_mtime for f in files if f.exists()), default=0.0)


@st.cache_data
def get_table(version):
    return rm.load_table()


@st.cache_resource
def get_models(version):
    return rm.load()


@st.cache_data
def get_reports(version):
    names = ("backtest", "metrics", "metrics_by_type", "calibration")
    paths = {k: rm.REPORTS / f"race_{k}.csv" for k in names}
    if not all(p.exists() for p in paths.values()):
        return None
    return {k: pd.read_csv(p, dtype={"car_number": str}) for k, p in paths.items()}


@st.cache_data
def get_saved_forecast(version):
    csv, meta = rm.REPORTS / "race_next.csv", rm.REPORTS / "race_next.json"
    if not (csv.exists() and meta.exists()):
        return None, None
    return pd.read_csv(csv, dtype={"car_number": str}), json.loads(meta.read_text())


@st.cache_data(ttl=1800, show_spinner=False)
def get_next_race(after_date):
    return next_race(after_date)


@st.cache_data(ttl=600, show_spinner="Reading NASCAR's entry list and simulating the race 10,000 times...")
def live_forecast(race_json, nonce, version):
    """Fresh forecast from the live entry list (and grid, once qualifying is done)."""
    race = json.loads(race_json)
    entries = entry_rows(race)
    if entries.empty:
        return None
    models, meta = get_models(version)
    rows = upcoming_rows(get_table(version), race, entries)
    fc, finish = rm.forecast_race(models, rows, meta["temperature"])
    return fc.join(rm.finish_distribution(finish))


def style(fig, height=380, y_title=None, x_title=None):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif'),
    )
    fig.update_xaxes(title=x_title, showgrid=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False)
    return fig


def finish_range_chart(fc, has_result, highlight=None):
    """One line per car: likely range of finishes, expected finish, and the real result if known."""
    fc = fc.sort_values("exp_finish").reset_index(drop=True)
    names = "#" + fc["car_number"] + " " + fc["driver"]
    colors = [ACCENT if c == highlight else BLUE for c in fc["car_number"]]
    fig = go.Figure()
    fig.add_bar(y=names, x=fc["finish_p90"] - fc["finish_p10"], base=fc["finish_p10"], orientation="h",
                name="Likely range (80% of simulations)",
                marker=dict(color=colors, opacity=0.35, cornerradius=4),
                customdata=fc[["finish_p10", "finish_p90"]],
                hovertemplate="%{y}: P%{customdata[0]:.0f}–P%{customdata[1]:.0f}<extra></extra>")
    fig.add_scatter(y=names, x=fc["exp_finish"], mode="markers", name="Expected finish",
                    marker=dict(color=colors, size=9, line=dict(color="white", width=1)),
                    hovertemplate="%{y}: expected P%{x:.1f}<extra></extra>")
    if has_result:
        ok = fc["dnf"] == 0
        fig.add_scatter(y=names[ok], x=fc.loc[ok, "finish_position"], mode="markers", name="Actual finish",
                        marker=dict(color=INK, size=8, symbol="diamond"),
                        hovertemplate="%{y}: finished P%{x}<extra></extra>")
        fig.add_scatter(y=names[~ok], x=fc.loc[~ok, "finish_position"], mode="markers",
                        name="Actual finish (DNF)", marker=dict(color=CRITICAL, size=9, symbol="x"),
                        hovertemplate="%{y}: DNF, classified P%{x}<extra></extra>")
    n = len(fc)
    style(fig, height=max(520, 22 * n), x_title="Finishing position (1 = win)")
    fig.update_layout(bargap=0.35)
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)", tickfont=dict(size=11))
    fig.update_xaxes(range=[0.5, n + 0.5], tickvals=[1, 5, 10, 15, 20, 25, 30, 35, 40],
                     showgrid=True, gridcolor=GRID)
    return fig


def chances_table(fc, has_result):
    cols = {"car_number": "Car", "driver": "Driver", "team": "Team", "start_position": "Start",
            "exp_finish": "Exp. finish", "p_win": "Win", "p_top3": "Top 3", "p_top5": "Top 5",
            "p_top10": "Top 10", "p_dnf": "DNF"}
    if has_result:
        cols["finish_position"] = "Actual"
    show = fc.sort_values("exp_finish")[list(cols)].rename(columns=cols)
    show.insert(0, "Rank", range(1, len(show) + 1))
    show["Car"] = "#" + show["Car"]
    pct = lambda label, top: st.column_config.ProgressColumn(label, format="percent", min_value=0, max_value=top)
    st.dataframe(show, hide_index=True, height=min(36 * len(show) + 40, 900), column_config={
        "Start": st.column_config.NumberColumn(format="P%d"),
        "Exp. finish": st.column_config.NumberColumn(format="%.1f"),
        "Win": pct("Win", max(0.15, show["Win"].max())), "Top 3": pct("Top 3", max(0.4, show["Top 3"].max())),
        "Top 5": pct("Top 5", max(0.5, show["Top 5"].max())), "Top 10": pct("Top 10", 1.0),
        "DNF": pct("DNF", max(0.4, show["DNF"].max())),
        "Actual": st.column_config.NumberColumn(format="P%d"),
    })


def panel(key):
    """A bordered card with a livery stripe across the top (styled in STYLE)."""
    return st.container(border=True, key=f"panel_{key}")


def race_chips(race, grid_known):
    chips = [f'<span class="chip">{html.escape(c)}</span>' for c in (
        f"{pd.Timestamp(race['race_date']):%a %b %d, %Y}", f"{race['scheduled_laps']} laps",
        f"{race['track_miles']:.3g}-mile {TYPE_LABELS[race['track_type']]}")]
    if grid_known is not None:
        chips.append('<span class="chip sky">'
                     + ("Grid set by qualifying" if grid_known else "Random grids until qualifying") + "</span>")
    if race["playoff"]:
        chips.append('<span class="chip hot">Playoff race</span>')
    return '<div class="chips">' + "".join(chips) + "</div>"


# The tech stack, in badge order (shown in the "How it works" panel).
TECH = ["Python", "Requests", "pandas", "NumPy", "XGBoost", "SHAP", "Monte Carlo", "Plackett–Luce",
        "Plotly", "Streamlit", "GitHub Actions"]


def tech_badges():
    return '<div class="tech">' + "".join(f"<span>{html.escape(t)}</span>" for t in TECH) + "</div>"


def tech_story(table, meta, reports):
    """How the app is built, from raw feeds to the page, with real counts."""
    tested = (f"{reports['backtest']['race_id'].nunique()} races since January 2025"
              if reports is not None else "every race since January 2025")
    return f"""
**1 · Data pipeline.** Python and Requests pull NASCAR's public results and loop-data feeds
for every Cup points race since 2018: {len(table):,} car entries in {table['race_id'].nunique()}
races at {table['track_id'].nunique()} tracks. Each finished race is downloaded once and cached.
`race_data.py` turns them into one table where every history column (recent form, record at
this track and at similar tracks, team form) uses **earlier races only**, so the model never
peeks at the result it's predicting.

**2 · Two XGBoost models.** An **XGBRanker** (`rank:pairwise`) learns *who finishes ahead of
whom* from {len(rm.FEATURES)} features. It only learns from cars still running at the finish,
because a wreck is bad luck, not slowness. An **XGBClassifier** gives each car's chance of a
DNF. Track type and manufacturer go in as native categorical features.

**3 · Monte Carlo race simulation.** Each car's score plus random (Gumbel) noise sets the
finishing order, which matches exactly what the pairwise ranker learned, and DNFs are drawn
separately. That runs 10,000 times. How much luck to add (the temperature, now
{meta['temperature']}) is fitted on races the model hadn't seen, using the Plackett–Luce
likelihood. Before qualifying, each batch of simulations draws a random starting grid.

**4 · Honest testing.** A walk-forward backtest over {tested}: before each race, the
models were retrained on earlier races only. The results are scored against simple baselines
(the starting grid, recent form, everyone equal), and every score is shown in *How good is
it?*, including where the model loses (superspeedways).

**5 · Explaining it.** SHAP values (`pred_contribs`) in *Driver view* show which factors push
each car up or down the order.

**6 · The app.** Streamlit with cached data that refreshes itself when new results are
pushed, and light and dark themes. Plotly charts use a palette checked for colour-blind
readers. It's tested with Streamlit's AppTest and Playwright screenshots, kept awake by a
GitHub Action, and deployed on Streamlit Community Cloud. The models were trained on
{meta['trained_races']} races through {meta['trained_through']}.

**What it doesn't know:** practice speeds, pit strategy, weather, and who is racing for the
championship.
"""


def driver_card(d, rank, n):
    start = f" · starts P{int(d['start_position'])}" if pd.notna(d["start_position"]) else ""
    meta = f"{d['team']} · {d['make']} · ranked {rank} of {n} by expected finish{start}"
    return (f'<div class="dcard"><div class="dnum"><span>#{html.escape(str(d["car_number"]))}</span></div>'
            f'<div><div class="dname">{html.escape(str(d["driver"]))}</div>'
            f'<div class="dmeta">{html.escape(meta)}</div></div></div>')


# ---------------------------------------------------------------- the next race

version = data_version()
try:
    table = get_table(version)
    models, meta = get_models(version)
except FileNotFoundError as err:
    st.error(f"{err}")
    st.stop()

race = get_next_race(table["race_date"].max())
saved_fc, saved_race = get_saved_forecast(version)
reports = get_reports(version)
st.session_state.setdefault("nonce", 0)

st.html(STYLE)
st.html(HERO.format(linkedin=LINKEDIN_URL, github=GITHUB_URL))

# The hook for fans, and a button for the tech side
hook, tech = st.columns([4, 1], vertical_alignment="center")
hook.html(f'<div class="hook"><b>Who wins the next Cup race?</b> Every car\'s chances, from XGBoost models '
          f'trained on {table["race_id"].nunique()} races since 2018 and 10,000 simulated races.</div>')
with tech.popover("How it works", icon=":material/settings:", width="stretch", key="tech"):
    st.subheader("How it's built")
    st.html(tech_badges())
    st.markdown(tech_story(table, meta, reports))

grid_known = None
if race is None:
    st.info("There's no upcoming Cup race on NASCAR's published schedule yet. The model's track record "
            "is under *How good is it?*.")
    fc = None
else:
    use_saved = (saved_race is not None and saved_race["race_id"] == race["race_id"]
                 and st.session_state["nonce"] == 0)
    fc = saved_fc if use_saved else live_forecast(json.dumps(race), st.session_state["nonce"], version)
    if fc is not None:
        grid_known = fc["start_position"].notna().all()

    head, button = st.columns([4, 1], vertical_alignment="bottom")
    with head:
        st.subheader(f"{race['race_name']} · {race['track']}")
        st.html(race_chips(race, grid_known))
    with button:
        if st.button("Refresh from NASCAR", width="stretch",
                     help="Re-reads the entry list and, once qualifying is done, the starting grid, "
                          "then re-simulates the race."):
            st.session_state["nonce"] += 1
            st.rerun()

    if fc is None:
        st.warning("NASCAR hasn't posted the entry list for this race yet. Try *Refresh from NASCAR* "
                   "later in race week.")
    else:
        st.write(f"**{len(fc)} cars entered.** "
                 + ("The starting grid is set by qualifying." if grid_known else
                    "The starting grid isn't set yet, so each batch of simulations draws a random grid. "
                    "Refresh after qualifying for the real one.")
                 + f" Forecast from 10,000 simulated races"
                 + (" (saved)." if use_saved else ", refreshed live."))

tab_race, tab_driver, tab_eval, tab_data = st.tabs(["Race forecast", "Driver view", "How good is it?", "Data"])

# ---------------------------------------------------------------- race forecast

with tab_race:
    if fc is None:
        st.write("No forecast to show yet.")
    else:
        n = len(fc)
        fav = fc.loc[fc["p_win"].idxmax()]
        top10 = fc.loc[fc["p_top10"].idxmax()]
        c1, c2, c3 = st.columns(3)
        c1.metric("Favourite to win", f"#{fav['car_number']} {fav['driver']}", f"{fav['p_win']:.1%} win chance",
                  delta_color="off", delta_arrow="off", border=True)
        c2.metric("Most likely top 10", f"#{top10['car_number']} {top10['driver']}",
                  f"{top10['p_top10']:.0%} top-10 chance", delta_color="off", delta_arrow="off", border=True)
        c3.metric("How open is it?", f"{fav['p_win']:.1%} vs {1 / n:.1%}",
                  "favourite's win chance vs an even split", delta_color="off", delta_arrow="off", border=True)

        with panel("finish_range"):
            st.subheader("Where each car is likely to finish")
            st.plotly_chart(finish_range_chart(fc, has_result=False, highlight=st.session_state.get("driver")))
            st.caption("Cars sorted by expected finish. The pale bar covers the middle 80% of simulated "
                       "finishes. The car picked in *Driver view* is shown in orange.")

        with panel("chances"):
            st.subheader("Every car's chances")
            chances_table(fc, has_result=False)
            st.caption("Progress bars are scaled per column so differences are easy to see; the numbers are "
                       "the actual chances.")

# ---------------------------------------------------------------- driver view

with tab_driver:
    if fc is None:
        st.write("No forecast to show yet.")
    else:
        order = fc.sort_values("exp_finish")
        options = list(order["car_number"])
        names = dict(zip(order["car_number"], "#" + order["car_number"] + " " + order["driver"]))
        if st.session_state.get("driver") not in options:
            st.session_state["driver"] = options[0]
        car = st.selectbox("Car", options, format_func=names.get, key="driver")
        row = fc[fc["car_number"] == car].iloc[[0]]
        d = row.iloc[0]
        rank = options.index(car) + 1

        st.html(driver_card(d, rank, len(fc)))
        cols = st.columns(5)
        for col, (label, value) in zip(cols, (("Win", f"{d['p_win']:.1%}"), ("Top 5", f"{d['p_top5']:.0%}"),
                                              ("Top 10", f"{d['p_top10']:.0%}"),
                                              ("Expected finish", f"{d['exp_finish']:.1f}"),
                                              ("DNF", f"{d['p_dnf']:.0%}"))):
            col.metric(label, value, border=True)

        left, right = st.columns(2)
        with left, panel("sims"):
            st.subheader("Where they finished in 10,000 simulations")
            pos_cols = [c for c in fc.columns if c.startswith("pos_")][: len(fc)]
            dist = row[pos_cols].iloc[0].to_numpy(dtype=float)
            fig = go.Figure(go.Bar(x=np.arange(1, len(dist) + 1), y=dist, marker=dict(color=BLUE, cornerradius=4),
                                   hovertemplate="P%{x}: %{y:.1%} of simulations<extra></extra>"))
            style(fig, height=360, x_title="Finishing position", y_title="Share of simulations")
            fig.update_layout(showlegend=False, bargap=0.15)
            fig.update_yaxes(tickformat=".0%", rangemode="tozero")
            fig.update_xaxes(tickvals=[1, 5, 10, 15, 20, 25, 30, 35, 40])
            st.plotly_chart(fig)
            st.caption("Finishes at the back include simulated DNFs.")
        with right, panel("shap"):
            st.subheader("What the model sees")
            shap_row = rm.prepare(row)
            if not grid_known:
                # The ranker almost never saw a missing grid in training, so explain the car
                # from a mid-field start rather than from "unknown".
                shap_row["start_position"] = len(fc) / 2
                shap_row["start_pct_field"] = 0.5
            contrib = rm.contributions(models, shap_row).iloc[0]
            contrib = contrib[contrib.abs() > 1e-3].sort_values(key=abs).tail(9)

            def shown(f):
                v = shap_row[f].iloc[0]
                if f == "start_pct_field" and not grid_known:
                    return "0.5 (mid-field; grid not set)"
                if pd.isna(v):
                    return "not known"
                return str(v) if isinstance(v, str) else f"{v:.3g}"

            labels = [f"{FEATURE_NAMES.get(f, f)} = {shown(f)}" for f in contrib.index]
            fig = go.Figure(go.Bar(x=contrib.values, y=labels, orientation="h",
                                   marker=dict(color=[POS if v > 0 else NEG for v in contrib.values], cornerradius=4),
                                   hovertemplate="%{y}<br>push: %{x:+.3f}<extra></extra>"))
            style(fig, height=360, x_title="← pushes back · pushes forward →")
            fig.update_layout(showlegend=False)
            fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(zeroline=True, zerolinecolor=MUTED, showgrid=True, gridcolor=GRID)
            st.plotly_chart(fig)
            st.caption("SHAP values from the XGBRanker: how much each factor moves this car's score up the "
                       "order (blue) or down it (orange)." + ("" if grid_known else " Until the grid is set, "
                       "this explains the car from a mid-field start."))

        hist = table[table["driver_id"] == d["driver_id"]].sort_values("race_date")
        with panel("form"):
            st.subheader("Recent form: last 10 races")
            recent = hist.tail(10)
            if recent.empty:
                st.write("No Cup races in the data yet.")
            else:
                labels = recent["race_date"].str[5:] + " " + recent["track"].str.replace(
                    r" (Motor |International |Super)?(Speedway|Raceway|Race)$", "", regex=True)
                out = recent["dnf"] == 1
                fig = go.Figure()
                fig.add_scatter(x=labels, y=recent["finish_position"], mode="lines", line=dict(color=GRID, width=2),
                                showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=labels[~out], y=recent.loc[~out, "finish_position"], mode="markers", name="Finish",
                                marker=dict(color=BLUE, size=11, line=dict(color="white", width=2)),
                                customdata=recent.loc[~out, ["start_position"]],
                                hovertemplate="%{x}: P%{y} (started P%{customdata[0]})<extra></extra>")
                fig.add_scatter(x=labels[out], y=recent.loc[out, "finish_position"], mode="markers", name="DNF",
                                marker=dict(color=CRITICAL, size=11, symbol="x"),
                                customdata=recent.loc[out, ["status"]],
                                hovertemplate="%{x}: P%{y} (%{customdata[0]})<extra></extra>")
                fig.add_hrect(y0=0.5, y1=10.5, fillcolor=BLUE, opacity=0.06, line_width=0,
                              annotation_text="Top 10", annotation_position="top left")
                style(fig, height=320, y_title="Finish")
                fig.update_yaxes(range=[41, 0.5], tickvals=[1, 10, 20, 30, 40])
                st.plotly_chart(fig)

        at_track = hist[hist["track_id"] == race["track_id"]]
        with panel("track"):
            st.subheader(f"At {race['track']}")
            if at_track.empty:
                st.write("First Cup start here since 2018.")
            else:
                st.caption(f"{len(at_track)} starts since 2018 · average finish "
                           f"{at_track['finish_position'].mean():.1f} · top 10 in {at_track['top10'].mean():.0%}")
                show = at_track.sort_values("race_date", ascending=False)[
                    ["race_date", "start_position", "finish_position", "status", "laps_led", "rating"]]
                st.dataframe(show.rename(columns={"race_date": "Date", "start_position": "Start",
                                                  "finish_position": "Finish", "status": "Status",
                                                  "laps_led": "Laps led", "rating": "Driver rating"}),
                             hide_index=True,
                             column_config={"Driver rating": st.column_config.NumberColumn(format="%.1f")})

# ---------------------------------------------------------------- how good is it?

with tab_eval:
    if reports is None:
        st.warning("Run `python race_train.py` to create the backtest.")
    else:
        bt = reports["backtest"]
        metrics = reports["metrics"]
        st.subheader("Does it beat simple baselines?")
        st.write(f"Walk-forward test over **{bt['race_id'].nunique()} races since January 2025** "
                 f"({len(bt):,} car entries): before each race, the models were trained only on earlier races, "
                 "then the race was simulated. Each score is compared with a simple baseline, such as the "
                 "starting grid or each driver's recent form.")

        def pick(prefix, baseline=None):
            rows = metrics[metrics["metric"].str.startswith(prefix)]
            if baseline:
                rows = rows[rows["baseline_name"] == baseline]
            return rows.iloc[0]

        tiles = (("Rank correlation", pick("Rank", "starting grid"), "{:.2f}", "{:+.2f}", "normal"),
                 ("Top-10 picks that hit", pick("Predicted top 10", "starting grid"), "{:.1f} of 10", "{:+.1f}",
                  "normal"),
                 ("Top-10 log loss", pick("Top-10 log loss"), "{:.3f}", "{:+.3f}", "inverse"),
                 ("Win log loss", pick("Win log loss"), "{:.3f}", "{:+.3f}", "inverse"))
        for col, (label, m, fmt, dfmt, colour) in zip(st.columns(4), tiles):
            col.metric(label, fmt.format(m["model"]), f"{dfmt.format(m['model'] - m['baseline'])} vs {m['baseline_name']}",
                       delta_color=colour, border=True,
                       help=f"{m['metric']}. Model {m['model']:.3f}, baseline {m['baseline']:.3f}.")
        st.caption("For log loss, lower is better, so a negative difference is a win for the model.")

        with panel("metrics"):
            st.subheader("Every score")
            show = metrics.drop(columns="races").copy()
            show["model_better"] = show["model_better"].map({True: "✅ yes", False: "❌ no"})
            st.dataframe(show.rename(columns={"metric": "Score", "model": "Model", "baseline": "Baseline",
                                              "baseline_name": "Baseline is", "model_better": "Model better?"}),
                         hide_index=True, column_config={"Model": st.column_config.NumberColumn(format="%.3f"),
                                                         "Baseline": st.column_config.NumberColumn(format="%.3f")})

        left, right = st.columns(2)
        with left, panel("by_type"):
            st.subheader("By kind of track")
            bt_type = reports["metrics_by_type"]
            rc = bt_type[bt_type["metric"].str.startswith("Rank") & (bt_type["baseline_name"] == "starting grid")]
            rc = rc.set_index("track_type").reindex([t for t in TYPE_LABELS if t in set(rc["track_type"])])
            labels = [f"{TYPE_LABELS[t].capitalize()} ({int(r)} races)" for t, r in rc["races"].items()]
            fig = go.Figure()
            fig.add_bar(x=labels, y=rc["model"], name="XGBoost", marker=dict(color=BLUE, cornerradius=4),
                        hovertemplate="%{x}: %{y:.2f}<extra>XGBoost</extra>")
            fig.add_bar(x=labels, y=rc["baseline"], name="Starting grid", marker=dict(color=MUTED, cornerradius=4),
                        hovertemplate="%{x}: %{y:.2f}<extra>Starting grid</extra>")
            style(fig, height=360, y_title="Rank correlation with the result")
            fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
            st.plotly_chart(fig)
            st.caption("1 would be a perfect prediction of the finishing order, 0 no better than random. "
                       "Short tracks are the most predictable; superspeedways are close to a lottery.")
        with right, panel("calibration"):
            st.subheader("Are the top-10 chances honest?")
            cal = reports["calibration"]
            cal = cal[cal["cars"] >= 20]
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                            line=dict(color=MUTED, width=1, dash="dash"), hoverinfo="skip")
            fig.add_scatter(x=cal["predicted"], y=cal["actual"], mode="markers+lines", name="XGBoost",
                            line=dict(color=BLUE, width=2),
                            marker=dict(size=[max(8, min(26, c / 40)) for c in cal["cars"]], color=BLUE,
                                        line=dict(color="white", width=2)),
                            customdata=cal["cars"],
                            hovertemplate="Predicted %{x:.0%} · actually %{y:.0%} (%{customdata} cars)<extra></extra>")
            style(fig, height=360, x_title="Predicted top-10 chance", y_title="Share that finished top 10")
            fig.update_xaxes(tickformat=".0%", range=[0, 1])
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(fig)
            st.caption("Cars grouped by predicted top-10 chance (bigger dots, bigger groups). On the dashed "
                       "line, the percentages can be taken at face value.")

        with panel("importance"):
            st.subheader("What the ranker relies on")
            imp = rm.importance(models).head(10).sort_values()
            fig = go.Figure(go.Bar(x=imp.values, y=[FEATURE_NAMES.get(f, f) for f in imp.index], orientation="h",
                                   marker=dict(color=BLUE, cornerradius=4),
                                   hovertemplate="%{y}: %{x:.0%} of total gain<extra></extra>"))
            style(fig, height=380, x_title="Share of total gain")
            fig.update_layout(showlegend=False)
            fig.update_xaxes(tickformat=".0%", showgrid=True, gridcolor=GRID)
            fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig)
            st.caption(f"Feature importance of the XGBRanker trained on all {meta['trained_races']} races "
                       f"through {meta['trained_through']}. Simulation temperature: {meta['temperature']}.")

        with panel("past_race"):
            st.subheader("Look back at a past race")
            past = bt.drop_duplicates("race_id").sort_values("race_date", ascending=False)
            race_pick = st.selectbox("Race", list(past["race_id"]),
                                     format_func=dict(zip(past["race_id"], past["race_label"])).get)
            one = bt[bt["race_id"] == race_pick]
            rho = one["exp_finish"].rank().corr(one["finish_position"].rank())
            hits = int(one.nsmallest(10, "exp_finish")["top10"].sum())
            winner = one.loc[one["finish_position"].idxmin()]
            win_rank = int(one["exp_finish"].rank().loc[winner.name])
            c1, c2, c3 = st.columns(3)
            c1.metric("Model's top 10 that hit", f"{hits} of 10", border=True)
            c2.metric("Rank correlation", f"{rho:.2f}", border=True)
            c3.metric("Winner's model rank", f"{win_rank} of {len(one)}", f"#{winner['car_number']} {winner['driver']}",
                      delta_color="off", delta_arrow="off", border=True)
            st.caption("Forecast made as if before the race: models trained only on earlier races, with the real "
                       "starting grid.")
            st.plotly_chart(finish_range_chart(one, has_result=True))
            st.caption("Black diamonds are real finishes; red × marks cars that didn't finish.")

# ---------------------------------------------------------------- data

with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Car entries", f"{len(table):,}", border=True)
    c2.metric("Cup points races", f"{table['race_id'].nunique()}", border=True)
    c3.metric("Tracks", f"{table['track_id'].nunique()}", border=True)
    c4.metric("Drivers", f"{table['driver_id'].nunique()}", border=True)

    with panel("table"):
        st.subheader("Every car, every race")
        st.write(f"{table['race_date'].min()} to {table['race_date'].max()}. All the history columns "
                 "(`form_`, `trk_`, `type_`, `team_`, `season_`) only use earlier races.")
        st.dataframe(table.tail(500), hide_index=True)
        st.caption("Showing the latest 500 rows; the download has all of them.")
        st.download_button("Download the full table (CSV)", table.to_csv(index=False), "races.csv",
                           mime="text/csv", icon=":material/download:")
    if fc is not None:
        with panel("inputs"):
            st.subheader("This race's model inputs")
            st.dataframe(fc[["car_number", "driver"] + rm.FEATURES], hide_index=True)

st.html(FOOTER.format(linkedin=LINKEDIN_URL, github=GITHUB_URL))
