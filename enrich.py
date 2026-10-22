"""Build the enriched, leak-free model table for Larson at Talladega.

Steps:
  1. Rebuild every lagged history feature from the raw CSV and check it matches
     the supplied model CSV (sanity check on the original agent's work).
  2. Add per-race metrics from NASCAR feeds (2021+ only) and from the raw file.
     These are measured *during* a race, so they only enter the model lagged.
  3. Append the upcoming race as the final row, with its targets left blank.

Outputs (in data/):
  larson_talladega_race_metrics.csv    per-race in-race metrics (NOT model inputs)
  larson_talladega_model_enriched.csv  model table; last row is the race to predict
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_nascar import fetch_race, talladega_races

DATA = Path(__file__).parent / "data"

# The race being predicted. Start position and qualifying speed stay blank until
# qualifying (Sat Oct 24); predict.py takes them as inputs.
NEXT_RACE = {
    "season": 2026, "race_date": "2026-10-25", "season_half": "fall", "playoff_race": "Y",
    "team": "Hendrick Motorsports", "manufacturer": "Chevrolet", "crew_chief": "Cliff Daniels",
    "car_number": "5", "field_size": 40, "car_gen": "gen7",
    # Assumed carried over from spring 2026; confirm against Goodyear's race-week notes.
    "tire_left_code": "D-5142", "tire_right_code": "D-5218",
}

NEXT_RACE_ID = 5631  # cf.nascar.com race_id for the race above; used to pull the lineup

LATE_RACE_SHARE = 0.20   # final fifth of the laps, where Talladega is usually decided
MAX_SERVICE_STOP_SEC = 30  # longer stops are damage repair, not crew speed
NEW_METRICS = ("late_race_avg_pos", "pct_laps_top5", "pct_laps_top10", "run_pos_std",
               "median_tire_stop_sec", "pit_road_pos_net", "incident_notes", "pit_penalties")
INCIDENT_WORDS = re.compile(
    r"caution|spin|wreck|crash|accident|incident|damage|contact|stack|into the wall|hits the wall", re.I)


# ---------------------------------------------------------------- feed metrics

def lap_metrics(lap_json, car):
    rows = [
        (str(drv["Number"]), lap["Lap"], float(lap["LapTime"]), lap["RunningPos"])
        for drv in lap_json.get("laps", [])
        for lap in drv.get("Laps", [])
        if lap.get("Lap", 0) > 0 and lap.get("LapTime")
    ]
    # Lap *times* are not used: in pack racing everyone runs the same time (the
    # driver/field ratio was 1.000 in every race), so only positions carry signal.
    laps = pd.DataFrame(rows, columns=["car", "lap", "time", "pos"])
    mine = laps[laps["car"] == car]
    if mine.empty:
        return {}
    late_start = laps["lap"].max() * (1 - LATE_RACE_SHARE)
    late = mine[mine["lap"] > late_start]
    return {
        "late_race_avg_pos": float(late["pos"].mean()) if len(late) else np.nan,
        "pct_laps_top5": float((mine["pos"] <= 5).mean()),
        "pct_laps_top10": float((mine["pos"] <= 10).mean()),
        "run_pos_std": float(mine["pos"].std()),
    }


def pit_metrics(pit_json, car):
    # The first block of the feed is the starting grid with zero/-1 placeholders.
    stops = [
        s for s in pit_json
        if str(s.get("vehicle_number")) == car
        and s.get("lap_count", 0) > 0 and s.get("pit_stop_duration", 0) > 0
    ]
    tire_stops = [s["pit_stop_duration"] for s in stops
                  if "WHEEL" in s.get("pit_stop_type", "") and s["pit_stop_duration"] <= MAX_SERVICE_STOP_SEC]
    return {
        "pit_stops": len(stops),
        "median_tire_stop_sec": float(np.median(tire_stops)) if tire_stops else np.nan,
        "pit_road_pos_net": float(sum(s.get("positions_gained_lost", 0) for s in stops)),
    }


def note_metrics(notes_json, car):
    """Keyword-based reading of the race-control notes, so treat the counts as approximate."""
    car_tag = re.compile(rf"#{car}(?!\d)")
    incidents, penalties = set(), set()  # keyed by NoteID; the feed repeats some notes
    for lap, entries in notes_json.get("laps", {}).items():
        for entry in entries:
            note = entry.get("Note", "")
            if not car_tag.search(note):
                continue
            key = (lap, entry.get("NoteID"))
            if "penal" in note.lower():
                penalties.add(key)
            elif INCIDENT_WORDS.search(note) and "pit" not in note.lower():
                incidents.add(key)
    return {"incident_notes": len(incidents), "pit_penalties": len(penalties)}


def feed_metrics(raw):
    """Feed-derived metrics per raw row, matched on season + spring/fall."""
    rows = []
    for season in sorted(raw["season"].unique()):
        for race in talladega_races(int(season)):
            match = raw[(raw["season"] == season) & (raw["season_half"] == race["season_half"])]
            if match.empty:
                continue
            car = str(match["car_number"].iloc[0])
            feeds = fetch_race(int(season), race["race_id"])
            metrics = {"race_date": match["race_date"].iloc[0], "nascar_race_id": race["race_id"]}
            if feeds["lap-times"]:
                metrics.update(lap_metrics(feeds["lap-times"], car))
            if feeds["live-pit-data"]:
                metrics.update(pit_metrics(feeds["live-pit-data"], car))
            if feeds["lap-notes"]:
                metrics.update(note_metrics(feeds["lap-notes"], car))
            rows.append(metrics)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- lagging

def add_lags(df, source_col, prev_col=None, hist_col=None):
    """prev_* = last race's value; hist_* = mean over all earlier races (NaNs skipped)."""
    shifted = df[source_col].shift(1)
    if prev_col:
        df[prev_col] = shifted
    if hist_col:
        df[hist_col] = shifted.expanding().mean()


def build():
    raw = pd.read_csv(DATA / "larson_talladega_raw.csv", dtype={"car_number": str})
    supplied = pd.read_csv(DATA / "larson_talladega_model.csv")

    metrics = raw[["race_date"]].copy()
    metrics["dnf"] = (raw["status"] != "running").astype(float)
    metrics["crash_dnf"] = (raw["status"] == "crash").astype(float)
    feeds = feed_metrics(raw)
    metrics = metrics.merge(feeds, on="race_date", how="left")
    metrics.to_csv(DATA / "larson_talladega_race_metrics.csv", index=False)

    # Raw rows plus the upcoming race, in date order.
    df = pd.concat([raw, pd.DataFrame([NEXT_RACE])], ignore_index=True)
    df = df.merge(metrics, on="race_date", how="left").sort_values("race_date").reset_index(drop=True)

    # Original lagged features, rebuilt from raw.
    add_lags(df, "finish_position", "prev_fin", "hist_mean_fin")
    add_lags(df, "driver_rating", "prev_rating", "hist_mean_rating")
    add_lags(df, "avg_running_position", "prev_avg_run_pos", "hist_mean_avg_run_pos")
    add_lags(df, "laps_led", "prev_laps_led", "hist_mean_laps_led")
    add_lags(df, "dnf", hist_col="hist_dnf_rate")
    df["prior_talladega_starts"] = np.arange(len(df))

    # New lagged features.
    add_lags(df, "crash_dnf", hist_col="hist_crash_rate")
    for col in NEW_METRICS:
        if col in df:
            add_lags(df, col, f"prev_{col}", f"hist_mean_{col}")

    # New pre-race feature: qualifying gap to pole, comparable across Gen6 and Gen7.
    df["qual_gap_to_pole_pct"] = (df["pole_speed_mph"] - df["qualifying_speed_mph"]) / df["pole_speed_mph"] * 100
    df["start_pct_field"] = df["start_position"] / df["field_size"]

    df["target_finish_position"] = df["finish_position"]
    df["target_top10"] = (df["finish_position"] <= 10).astype(float).where(df["finish_position"].notna())
    df["target_dnf"] = df["dnf"].where(df["finish_position"].notna())

    check_against_supplied(df, supplied)

    keep = [c for c in supplied.columns if c in df] + [
        c for c in df.columns
        if c not in supplied.columns
        and (c.startswith(("prev_", "hist_")) or c in ("qual_gap_to_pole_pct", "nascar_race_id"))
    ]
    out = df[keep]
    out.to_csv(DATA / "larson_talladega_model_enriched.csv", index=False)
    return out


def check_against_supplied(df, supplied):
    """Report any rebuilt column that disagrees with the supplied model CSV beyond rounding."""
    hist = df[df["race_date"].isin(supplied["race_date"])].set_index("race_date")
    sup = supplied.set_index("race_date")
    for col in ("prev_fin", "hist_mean_fin", "prev_rating", "hist_mean_rating", "prev_avg_run_pos",
                "hist_mean_avg_run_pos", "prev_laps_led", "hist_mean_laps_led", "hist_dnf_rate",
                "prior_talladega_starts", "start_pct_field", "target_dnf", "target_top10"):
        diff = (hist[col] - sup[col]).abs()
        bad = diff[diff > 0.01]
        print(f"  {col:<24} {'OK' if bad.empty else 'MISMATCH at ' + ', '.join(bad.index)}")


if __name__ == "__main__":
    print("Checking rebuilt features against larson_talladega_model.csv:")
    table = build()
    new_cols = [f"{p}{m}" for m in NEW_METRICS for p in ("prev_", "hist_mean_")]
    new_cols += ["hist_crash_rate", "qual_gap_to_pole_pct"]
    print(f"\nWrote {len(table)} rows ({len(table) - 1} races + the race to predict).")
    print("Non-missing counts for the new columns:")
    print(table[new_cols].notna().sum().to_string())
