"""Streamlit dashboard for the Larson-at-Talladega predictor.

Run:  streamlit run app.py

Tabs: Forecast (the upcoming race), Race history, Backtest (model vs baseline), Data.
Models are trained in memory from the enriched table, using the same code as train.py.
"""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from enrich import NEXT_RACE, NEXT_RACE_ID
from fetch_nascar import starting_info
from model import (DATA, DEFAULT_FEATURE_SET, FEATURE_SETS, backtest, forecast, load_table,
                   score, set_race_inputs, split, sweep_start_positions, train_all)

# Chart palette (validated reference palette; see docs/app.md).
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"
POS, NEG = "#2a78d6", "#e34948"  # diverging poles for feature contributions
CRITICAL = "#d03b3b"             # status colour for DNF, always paired with an x marker + label
GRID = "rgba(137,135,129,0.25)"

CAR = f"{NEXT_RACE['car_number']} Car"  # display name used throughout the page

FEATURE_NAMES = {
    "start_pct_field": "Start position (share of field)",
    "qual_gap_to_pole_pct": "Qualifying gap to pole (%)",
    "is_playoff": "Playoff race",
    "is_gen7": "Gen7 car",
    "is_hendrick": "Hendrick Motorsports",
    "prior_talladega_starts": "Prior Talladega starts",
    "prev_fin": "Last Talladega finish",
    "hist_mean_fin": "Average Talladega finish",
    "prev_rating": "Last driver rating",
    "hist_mean_rating": "Average driver rating",
    "prev_avg_run_pos": "Last avg running position",
    "hist_mean_avg_run_pos": "Average running position",
    "hist_mean_laps_led": "Average laps led",
    "hist_dnf_rate": "DNF rate so far",
    "hist_crash_rate": "Crash rate so far",
    "prev_pct_laps_top10": "Last race: laps in top 10",
    "hist_mean_pct_laps_top10": "Average laps in top 10",
    "prev_pct_laps_top5": "Last race: laps in top 5",
    "hist_mean_late_race_avg_pos": "Average late-race position",
    "hist_mean_run_pos_std": "Average position swing",
    "hist_mean_median_tire_stop_sec": "Average tire-stop time (s)",
    "hist_mean_pit_road_pos_net": "Average pit-road positions",
    "hist_mean_incident_notes": "Average incident notes",
}

st.set_page_config(page_title=f"{CAR} Talladega Predictor", page_icon="🏁", layout="wide")


# ---------------------------------------------------------------- cached data/model

@st.cache_data
def get_table():
    return load_table()


@st.cache_data
def get_history():
    raw = pd.read_csv(DATA / "larson_talladega_raw.csv")
    metrics = pd.read_csv(DATA / "larson_talladega_race_metrics.csv")
    hist = raw.merge(metrics[["race_date", "pct_laps_top10", "pct_laps_top5"]], on="race_date", how="left")
    hist["race_label"] = hist["season"].astype(str) + " " + hist["season_half"]
    return hist.sort_values("race_date").reset_index(drop=True)


@st.cache_resource
def get_models(feature_set):
    past, _ = split(get_table())
    return train_all(past, feature_set)


@st.cache_data
def get_backtest(feature_set):
    past, _ = split(get_table())
    bt = backtest(past, feature_set)
    return bt, score(bt)


def style(fig, height=380, y_title=None, x_title=None):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif'),
    )
    fig.update_xaxes(title=x_title, showgrid=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False)
    return fig


# ---------------------------------------------------------------- sidebar inputs

def pull_lineup():
    info = starting_info(NEXT_RACE["season"], NEXT_RACE_ID, NEXT_RACE["car_number"])
    if info is None:
        st.session_state["lineup_msg"] = "Lineup not posted yet. Qualifying is Saturday, Oct 24."
        return
    st.session_state["start"] = info["start_position"]
    has_speeds = bool(info["qualifying_speed"] and info["pole_speed"])
    st.session_state["use_qual"] = has_speeds
    if has_speeds:
        st.session_state["qual"] = float(info["qualifying_speed"])
        st.session_state["pole"] = float(info["pole_speed"])
    st.session_state["lineup_msg"] = f"Pulled from NASCAR: starts P{info['start_position']}."


table = get_table()
past, upcoming = split(table)
if upcoming.empty:
    st.error("The table has no upcoming race. Update NEXT_RACE in enrich.py and run it again.")
    st.stop()
next_row = upcoming.iloc[[0]]
field_size = int(next_row["field_size"].iloc[0])

for key, default in (("start", 20), ("use_qual", False), ("qual", 181.0), ("pole", 182.0)):
    st.session_state.setdefault(key, default)

with st.sidebar:
    st.header("Race inputs")
    st.button("Pull lineup from NASCAR", on_click=pull_lineup, width="stretch",
              help=f"Reads the {CAR}'s start position and qualifying speeds once the lineup is posted.")
    if msg := st.session_state.get("lineup_msg"):
        st.caption(msg)
    start = st.slider("Start position", 1, field_size, key="start")
    use_qual = st.checkbox("Include qualifying speeds", key="use_qual")
    qual = pole = None
    if use_qual:
        qual = st.number_input(f"{CAR} qualifying speed (mph)", 150.0, 220.0, key="qual",
                               step=0.01, format="%.3f")
        pole = st.number_input("Pole speed (mph)", 150.0, 220.0, key="pole", step=0.01, format="%.3f")
        if qual > pole:
            st.warning(f"The {CAR} speed is above the pole speed. Check the numbers.")
    st.divider()
    feature_set = st.radio(
        "Feature set", list(FEATURE_SETS), index=list(FEATURE_SETS).index(DEFAULT_FEATURE_SET),
        format_func=lambda s: f"{s} ({len(FEATURE_SETS[s])} features)",
        help="core: results and loop data back to 2014. enriched: adds lap, pit and race-note "
             "features, which exist only from 2021 on.")

models = get_models(feature_set)
row = set_race_inputs(next_row, start, qual, pole)
pred = forecast(models, row)
bt, scores = get_backtest(feature_set)

# ---------------------------------------------------------------- header

st.title(f"{CAR} at Talladega")
st.caption(f"{NEXT_RACE['season']} fall race, {pd.Timestamp(NEXT_RACE['race_date']):%A %b %d, %Y} · "
           f"{NEXT_RACE['team']} · XGBoost trained on {len(past)} races")
wins = int(scores["model_better"].sum())
st.info(f"**Read these numbers with care.** In the race-by-race backtest the model beats the {CAR}'s plain "
        f"historical averages on {wins} of {len(scores)} scores. With {len(past)} races and pack-racing "
        "crashes, most of the result is luck. See the Backtest tab.", icon="ℹ️")

tab_fc, tab_hist, tab_bt, tab_data = st.tabs(["Forecast", "Race history", "Backtest", "Data"])

# ---------------------------------------------------------------- forecast

with tab_fc:
    c1, c2, c3 = st.columns(3)
    c1.metric("Top-10 chance", f"{pred['top10']:.0%}",
              f"{(pred['top10'] - models['top10'].fallback) * 100:+.0f} pts vs track history", border=True)
    c2.metric("DNF chance", f"{pred['dnf']:.0%}",
              f"{(pred['dnf'] - models['dnf'].fallback) * 100:+.0f} pts vs track history",
              delta_color="inverse", border=True)
    c3.metric("Expected finish", f"{pred['finish']:.1f}",
              f"{pred['finish'] - models['finish'].fallback:+.1f} vs track average",
              delta_color="inverse", border=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Chances by start position")
        sweep = sweep_start_positions(models, set_race_inputs(next_row, None, qual, pole))
        fig = go.Figure()
        for col, name, color in (("top10", "Top-10 chance", BLUE), ("dnf", "DNF chance", ORANGE)):
            fig.add_scatter(x=sweep["start_position"], y=sweep[col], name=name, mode="lines",
                            line=dict(color=color, width=2),
                            hovertemplate=f"{name}: %{{y:.0%}}<extra></extra>")
            fig.add_annotation(x=sweep["start_position"].iloc[-1], y=sweep[col].iloc[-1], text=name,
                               xanchor="left", showarrow=False, xshift=6, font=dict(size=12))
        fig.add_vline(x=start, line=dict(color=MUTED, width=1, dash="dot"))
        fig.add_annotation(x=start, y=0, yref="paper", text=f"P{start}", showarrow=False,
                           xanchor="left", yanchor="bottom", xshift=4, font=dict(size=12))
        style(fig, y_title="Probability", x_title="Start position")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        fig.update_xaxes(range=[0.5, field_size + 12], tickvals=[1, 10, 20, 30, 40])
        st.plotly_chart(fig)
        st.caption("Everything except start position is held fixed. A flat line means the model "
                   "found no reliable link between start position and the outcome.")

    with right:
        st.subheader("What moves the top-10 chance")
        contrib = models["top10"].contributions(row)
        contrib = contrib[contrib.abs() > 0.002].sort_values(key=abs).tail(8)
        if contrib.empty:
            st.write("The model is at its base rate; no feature moves it.")
        else:
            labels = [f"{FEATURE_NAMES.get(f, f)} = "
                      + ("not set" if pd.isna(row[f].iloc[0]) else f"{row[f].iloc[0]:.3g}")
                      for f in contrib.index]
            fig = go.Figure(go.Bar(
                x=contrib.values, y=labels, orientation="h",
                marker=dict(color=[POS if v > 0 else NEG for v in contrib.values], cornerradius=4),
                hovertemplate="%{y}<br>push: %{x:+.3f} log-odds<extra></extra>"))
            style(fig, height=380, x_title="← lowers · raises →")
            fig.update_layout(hovermode="closest", showlegend=False)
            fig.update_yaxes(gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(zeroline=True, zerolinecolor=MUTED, showgrid=True, gridcolor=GRID)
            st.plotly_chart(fig)
            st.caption("SHAP contributions from XGBoost for this forecast. Blue bars raise the chance; "
                       "red bars lower it.")

# ---------------------------------------------------------------- history

with tab_hist:
    hist = get_history()
    st.subheader("Start and finish, every Talladega race")
    dnf = hist["status"] != "running"
    fig = go.Figure()
    for _, r in hist.iterrows():
        fig.add_scatter(x=[r["race_label"]] * 2, y=[r["start_position"], r["finish_position"]],
                        mode="lines", line=dict(color=GRID, width=2), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=hist["race_label"], y=hist["start_position"], name="Start", mode="markers",
                    marker=dict(color=MUTED, size=9, line=dict(color="white", width=2)),
                    hovertemplate="Start: P%{y}<extra></extra>")
    fig.add_scatter(x=hist.loc[~dnf, "race_label"], y=hist.loc[~dnf, "finish_position"],
                    name="Finish (running)", mode="markers",
                    marker=dict(color=BLUE, size=11, line=dict(color="white", width=2)),
                    hovertemplate="Finish: P%{y}<extra></extra>")
    fig.add_scatter(x=hist.loc[dnf, "race_label"], y=hist.loc[dnf, "finish_position"],
                    name="Finish (DNF)", mode="markers",
                    marker=dict(color=CRITICAL, size=11, symbol="x"),
                    customdata=hist.loc[dnf, "status"],
                    hovertemplate="Finish: P%{y} (%{customdata})<extra></extra>")
    fig.add_hrect(y0=0.5, y1=10.5, fillcolor=BLUE, opacity=0.06, line_width=0,
                  annotation_text="Top 10", annotation_position="top left")
    style(fig, height=420, y_title="Position")
    fig.update_yaxes(range=[hist["field_size"].max() + 1, 0.5], tickvals=[1, 10, 20, 30, 40])
    st.plotly_chart(fig)
    top10_rate = (hist["finish_position"] <= 10).mean()
    st.caption(f"{len(hist)} races · top 10 in {top10_rate:.0%} · did not finish in {dnf.mean():.0%}. "
               "Position 1 is at the top. Races before 2021 were run in the #42 Chip Ganassi car by the "
               f"same driver: this history follows the driver now in the {CAR}.")

    with_laps = hist.dropna(subset=["pct_laps_top10"])
    if len(with_laps):
        st.subheader("Share of laps run inside the top 10 (2021 on)")
        fig = go.Figure(go.Bar(
            x=with_laps["race_label"], y=with_laps["pct_laps_top10"],
            marker=dict(color=BLUE, cornerradius=4),
            customdata=with_laps["finish_position"],
            hovertemplate="%{y:.0%} of laps in top 10 · finished P%{customdata}<extra></extra>"))
        style(fig, height=300, y_title="Laps in top 10")
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig)
        st.caption("From NASCAR's lap-by-lap feed. Earlier races have no lap data.")

    cols = ["race_label", "start_position", "finish_position", "status", "laps_led", "driver_rating"]
    st.dataframe(hist[[c for c in cols if c in hist]], hide_index=True)

# ---------------------------------------------------------------- backtest

with tab_bt:
    st.subheader("Does the model beat the plain averages?")
    st.write(f"Walk-forward test: for each of the last {len(bt)} races, the model is trained only on "
             "the races before it, then scored on that race. The baseline predicts the average so far "
             "(top-10 rate, DNF rate, average finish). Lower is better for every score.")
    show = scores.copy()
    show["target"] = show["target"].map({"top10": "Top 10", "dnf": "DNF", "finish": "Finish position"})
    show["metric"] = show["metric"].map({"brier": "Brier score", "log_loss": "Log loss",
                                         "mae": "Mean abs. error (positions)"})
    show["model_better"] = show["model_better"].map({True: "✅ yes", False: "❌ no"})
    st.dataframe(show.rename(columns={"target": "Target", "metric": "Score", "model": "Model",
                                      "baseline": "Baseline", "model_better": "Model better?"}),
                 hide_index=True, column_config={"Model": st.column_config.NumberColumn(format="%.3f"),
                                                 "Baseline": st.column_config.NumberColumn(format="%.3f")})

    target = st.segmented_control("Target", ["top10", "dnf", "finish"], default="top10",
                                  format_func=lambda t: {"top10": "Top 10", "dnf": "DNF",
                                                         "finish": "Finish position"}[t])
    target = target or "top10"
    fig = go.Figure()
    fig.add_scatter(x=bt["race_label"], y=bt[f"pred_{target}"], name="Model", mode="lines+markers",
                    line=dict(color=BLUE, width=2), marker=dict(size=8))
    fig.add_scatter(x=bt["race_label"], y=bt[f"base_{target}"], name="Baseline (average so far)",
                    mode="lines", line=dict(color=MUTED, width=2, dash="dash"))
    if target == "finish":
        fig.add_scatter(x=bt["race_label"], y=bt["actual_finish"], name="Actual finish", mode="markers",
                        marker=dict(color=ORANGE, size=10, line=dict(color="white", width=2)))
        style(fig, height=400, y_title="Finish position")
        fig.update_yaxes(autorange="reversed")
    else:
        happened = bt[bt[f"actual_{target}"] == 1]
        label = "Top 10 happened" if target == "top10" else "DNF happened"
        fig.add_scatter(x=happened["race_label"], y=happened[f"pred_{target}"], name=label,
                        mode="markers", marker=dict(color=ORANGE, size=14, symbol="star",
                                                    line=dict(color="white", width=1)))
        style(fig, height=400, y_title="Predicted probability")
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
    st.plotly_chart(fig)
    st.caption("Stars mark races where the outcome happened. A useful model puts its stars higher "
               "than the baseline line and its other points lower.")
    with st.expander("Race-by-race backtest numbers"):
        st.dataframe(bt, hide_index=True)

# ---------------------------------------------------------------- data

with tab_data:
    st.subheader("Model table")
    st.write("One row per race. The last row is the race being predicted; its targets are blank.")
    st.dataframe(table, hide_index=True)
    st.download_button("Download CSV", table.to_csv(index=False), "larson_talladega_model_enriched.csv",
                       mime="text/csv")
