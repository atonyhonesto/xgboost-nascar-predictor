"""XGBoost NASCAR Predictor: every car's chances in the next Cup race.

Run:  streamlit run app.py

The next race is found on NASCAR's published schedule. The page starts from the
forecast race_train.py saved; "Refresh from NASCAR" re-reads the entry list (and the
starting grid once qualifying is done) and re-simulates the race live.

Tabs: Race forecast · Driver view · How good is it? · Data
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import race_model as rm
from race_data import entry_rows, next_race, upcoming_rows

# Chart palette (validated, colour-blind-safe reference palette; see docs/app.md).
BLUE, ORANGE, MUTED, INK = "#2a78d6", "#eb6834", "#898781", "#0b0b0b"
POS, NEG = "#2a78d6", "#e34948"  # diverging poles for SHAP contributions
CRITICAL = "#d03b3b"             # DNF, always paired with an x marker + label
GRID = "rgba(137,135,129,0.25)"

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

@st.cache_data
def get_table():
    return rm.load_table()


@st.cache_resource
def get_models():
    return rm.load()


@st.cache_data
def get_reports():
    names = ("backtest", "metrics", "metrics_by_type", "calibration")
    paths = {k: rm.REPORTS / f"race_{k}.csv" for k in names}
    if not all(p.exists() for p in paths.values()):
        return None
    return {k: pd.read_csv(p, dtype={"car_number": str}) for k, p in paths.items()}


@st.cache_data
def get_saved_forecast():
    csv, meta = rm.REPORTS / "race_next.csv", rm.REPORTS / "race_next.json"
    if not (csv.exists() and meta.exists()):
        return None, None
    return pd.read_csv(csv, dtype={"car_number": str}), json.loads(meta.read_text())


@st.cache_data(ttl=1800, show_spinner=False)
def get_next_race(after_date):
    return next_race(after_date)


@st.cache_data(ttl=600, show_spinner="Reading NASCAR's entry list and simulating the race 10,000 times...")
def live_forecast(race_json, nonce):
    """Fresh forecast from the live entry list (and grid, once qualifying is done)."""
    race = json.loads(race_json)
    entries = entry_rows(race)
    if entries.empty:
        return None
    models, meta = get_models()
    rows = upcoming_rows(get_table(), race, entries)
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
    colors = [ORANGE if c == highlight else BLUE for c in fc["car_number"]]
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


# ---------------------------------------------------------------- the next race

try:
    table = get_table()
    models, meta = get_models()
except FileNotFoundError as err:
    st.error(f"{err}")
    st.stop()

race = get_next_race(table["race_date"].max())
saved_fc, saved_race = get_saved_forecast()
st.session_state.setdefault("nonce", 0)

st.title("XGBoost NASCAR Predictor")
if race is None:
    st.info("There's no upcoming Cup race on NASCAR's published schedule yet. The model's track record "
            "is under *How good is it?*.")
    fc = None
else:
    use_saved = (saved_race is not None and saved_race["race_id"] == race["race_id"]
                 and st.session_state["nonce"] == 0)
    fc = saved_fc if use_saved else live_forecast(json.dumps(race), st.session_state["nonce"])

    head, button = st.columns([4, 1], vertical_alignment="bottom")
    with head:
        st.subheader(f"{race['race_name']} · {race['track']}")
        st.caption(f"{pd.Timestamp(race['race_date']):%A, %B %d, %Y} · {race['scheduled_laps']} laps · "
                   f"{race['track_miles']:.3g}-mile {TYPE_LABELS[race['track_type']]}"
                   + (" · playoff race" if race["playoff"] else ""))
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
        grid_known = fc["start_position"].notna().all()
        st.write(f"**{len(fc)} cars entered.** "
                 + ("The starting grid is set by qualifying." if grid_known else
                    "The starting grid isn't set yet, so each batch of simulations draws a random grid. "
                    "Refresh after qualifying for the real one.")
                 + f" Forecast from 10,000 simulated races"
                 + (" (saved)." if use_saved else ", refreshed live."))

tab_race, tab_driver, tab_eval, tab_data = st.tabs(["Race forecast", "Driver view", "How good is it?", "Data"])
reports = get_reports()

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

        st.subheader("Where each car is likely to finish")
        st.plotly_chart(finish_range_chart(fc, has_result=False, highlight=st.session_state.get("driver")))
        st.caption("Cars sorted by expected finish. The pale bar covers the middle 80% of simulated finishes. "
                   "The car picked in *Driver view* is shown in orange.")

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

        cols = st.columns(5)
        for col, (label, value) in zip(cols, (("Win", f"{d['p_win']:.1%}"), ("Top 5", f"{d['p_top5']:.0%}"),
                                              ("Top 10", f"{d['p_top10']:.0%}"),
                                              ("Expected finish", f"{d['exp_finish']:.1f}"),
                                              ("DNF", f"{d['p_dnf']:.0%}"))):
            col.metric(label, value, border=True)
        st.caption(f"{d['driver']} · {d['team']} · {d['make']} · ranked {rank} of {len(fc)} by expected finish"
                   + (f" · starts P{int(d['start_position'])}" if pd.notna(d["start_position"]) else ""))

        left, right = st.columns(2)
        with left:
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
        with right:
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
                       "order (blue) or down it (red)." + ("" if grid_known else " Until the grid is set, "
                       "this explains the car from a mid-field start."))

        hist = table[table["driver_id"] == d["driver_id"]].sort_values("race_date")
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
        st.subheader(f"At {race['track']}")
        if at_track.empty:
            st.write("First Cup start here since 2018.")
        else:
            st.caption(f"{len(at_track)} starts since 2018 · average finish {at_track['finish_position'].mean():.1f}"
                       f" · top 10 in {at_track['top10'].mean():.0%}")
            show = at_track.sort_values("race_date", ascending=False)[
                ["race_date", "start_position", "finish_position", "status", "laps_led", "rating"]]
            st.dataframe(show.rename(columns={"race_date": "Date", "start_position": "Start",
                                              "finish_position": "Finish", "status": "Status",
                                              "laps_led": "Laps led", "rating": "Driver rating"}),
                         hide_index=True, column_config={"Driver rating": st.column_config.NumberColumn(format="%.1f")})

# ---------------------------------------------------------------- how good is it?

with tab_eval:
    if reports is None:
        st.warning("Run `python race_train.py` to create the backtest.")
    else:
        bt = reports["backtest"]
        st.subheader("Does it beat simple baselines?")
        st.write(f"Walk-forward test over **{bt['race_id'].nunique()} races since January 2025** "
                 f"({len(bt):,} car entries): before each race, the models were trained only on earlier races, "
                 "then the race was simulated. Each score is compared with a simple baseline, such as the "
                 "starting grid or each driver's recent form.")
        show = reports["metrics"].drop(columns="races").copy()
        show["model_better"] = show["model_better"].map({True: "✅ yes", False: "❌ no"})
        st.dataframe(show.rename(columns={"metric": "Score", "model": "Model", "baseline": "Baseline",
                                          "baseline_name": "Baseline is", "model_better": "Model better?"}),
                     hide_index=True, column_config={"Model": st.column_config.NumberColumn(format="%.3f"),
                                                     "Baseline": st.column_config.NumberColumn(format="%.3f")})

        left, right = st.columns(2)
        with left:
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
        with right:
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
        st.caption(f"Feature importance of the XGBRanker trained on all {meta['trained_races']} races through "
                   f"{meta['trained_through']}. Simulation temperature: {meta['temperature']}.")

        st.subheader("Look back at a past race")
        past = bt.drop_duplicates("race_id").sort_values("race_date", ascending=False)
        pick = st.selectbox("Race", list(past["race_id"]),
                            format_func=dict(zip(past["race_id"], past["race_label"])).get)
        one = bt[bt["race_id"] == pick]
        rho = one["exp_finish"].rank().corr(one["finish_position"].rank())
        hits = int(one.nsmallest(10, "exp_finish")["top10"].sum())
        winner = one.loc[one["finish_position"].idxmin()]
        win_rank = int(one["exp_finish"].rank().loc[winner.name])
        st.write(f"Forecast made as if before the race (models trained only on earlier races, real starting grid). "
                 f"**{hits} of the model's top 10** finished in the top 10, rank correlation **{rho:.2f}**. "
                 f"Winner: #{winner['car_number']} {winner['driver']}, ranked {win_rank} of {len(one)} by the model.")
        st.plotly_chart(finish_range_chart(one, has_result=True))
        st.caption("Black diamonds are real finishes; red × marks cars that didn't finish.")

# ---------------------------------------------------------------- data

with tab_data:
    st.subheader("Every car, every race")
    st.write(f"{len(table):,} car entries in {table['race_id'].nunique()} Cup points races, "
             f"{table['race_date'].min()} to {table['race_date'].max()}. All the history columns "
             "(`form_`, `trk_`, `type_`, `team_`, `season_`) only use earlier races.")
    st.dataframe(table.tail(500), hide_index=True)
    st.caption("Showing the latest 500 rows; the download has all of them.")
    st.download_button("Download the full table (CSV)", table.to_csv(index=False), "races.csv", mime="text/csv")
    if fc is not None:
        st.subheader("This race's model inputs")
        st.dataframe(fc[["car_number", "driver"] + rm.FEATURES], hide_index=True)
