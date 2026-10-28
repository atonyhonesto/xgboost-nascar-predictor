"""Build the all-tracks table: every car in every NASCAR Cup points race since 2018.

This is the dataset behind the app. One row per car per race, from NASCAR's public
feeds (results from 2018, loop data from 2019). Everything measured during a race
enters only as history, lagged so a row sees nothing from its own race:

  form    the driver's last 5 races, any track (finish, rating, running position)
  season  the driver's average rating so far this season
  track   the driver's record at this track
  type    the driver's last 10 races at this kind of track
  team    the team's last 5 races

The upcoming race isn't stored here: `next_race()` finds it on NASCAR's schedule and
`upcoming_rows()` attaches everyone's history to its entry list.

Usage:  python race_data.py      -> data/races.csv
"""
from datetime import date

import numpy as np
import pandas as pd

from fetch_nascar import BASE, CACHE, CUP, get_json, loop_stats, race_results, weekend_feed

OUT = CACHE.parent / "races.csv"
FIRST_SEASON = 2018  # earliest season with every car's results in the weekend feed
FORM_RACES = 5       # "recent form" window, any track
TYPE_RACES = 10      # window for races at the same kind of track
TEAM_RACES = 5

ROAD_WORDS = ("Road Course", "Street", "Circuit of The Americas", "Sonoma", "Watkins Glen",
              "Road America", "Autódromo")


# ---------------------------------------------------------------- schedule

def track_type(name, season, miles):
    """Short track, intermediate, superspeedway or road course.

    Ovals are split by length; Atlanta has raced like a superspeedway (pack drafting)
    since its 2022 reconfiguration. Bristol's dirt races count as short-track races.
    """
    if any(w in name for w in ROAD_WORDS):
        return "road"
    if "Talladega" in name or name == "Daytona International Speedway" or ("Atlanta" in name and season >= 2022):
        return "superspeedway"
    if "Dirt" in name or miles < 1.1:
        return "short"
    return "intermediate"


def schedule(season, refresh=False):
    """Points races in a season (exhibitions excluded), in date order, with track details."""
    sch = get_json(f"{BASE}/{season}/race_list_basic.json", CACHE / f"{season}_race_list.json", refresh=refresh)
    races = []
    for r in (sch or {}).get(f"series_{CUP}", []):
        # The 2017 file lists 2018's races, so keep only races dated in the season asked for.
        if r.get("race_type_id") != 1 or not r["race_date"].startswith(str(season)):
            continue
        miles = r["scheduled_distance"] / r["scheduled_laps"] if r.get("scheduled_laps") else np.nan
        races.append({
            "season": season, "race_id": r["race_id"], "race_date": r["race_date"][:10],
            "race_name": r["race_name"], "track_id": r["track_id"], "track": r["track_name"],
            "track_miles": round(miles, 3), "track_type": track_type(r["track_name"], season, miles),
            "scheduled_laps": r["scheduled_laps"], "playoff": int(bool(r.get("playoff_round"))),
        })
    return sorted(races, key=lambda r: r["race_date"])


def next_race(after_date, refresh=True):
    """The first points race scheduled after `after_date` (this season or next), or None."""
    year = int(after_date[:4])
    for season in (year, year + 1):
        for race in schedule(season, refresh=refresh):
            if race["race_date"] > after_date:
                return race
    return None


# ---------------------------------------------------------------- rows

def race_rows(race):
    """One row per classified car (cars that didn't qualify have finishing_position 0)."""
    info = race_results(race["season"], race["race_id"])
    if info is None:
        return []
    loop = {d["driver_id"]: d for d in (loop_stats(race["season"], race["race_id"]) or [])}
    cars = [r for r in info["results"] if r.get("finishing_position")]
    rows = []
    for r in cars:
        lp = loop.get(r["driver_id"], {})
        laps = lp.get("laps") or np.nan
        rows.append({
            # The schedule feed's dates aren't always right (it lists fall 2025 Talladega on
            # Nov 17; it ran Oct 19), and lagged history depends on order, so use the result's date.
            **race, "race_date": info["race_date"][:10], "field_size": len(cars),
            "car_number": str(r["car_number"]), "driver_id": r["driver_id"],
            "driver": r["driver_fullname"].replace("*", "").strip(), "team": r["team_name"],
            "make": r["car_make"], "start_position": r["starting_position"] or np.nan,
            "qualifying_speed": r.get("qualifying_speed") or np.nan,
            "pole_speed": info.get("pole_winner_speed") or np.nan,
            "finish_position": r["finishing_position"], "laps_completed": r["laps_completed"],
            "laps_led": r["laps_led"], "status": r["finishing_status"],
            "rating": lp.get("rating", np.nan), "avg_run_pos": lp.get("avg_ps", np.nan),
            "pct_laps_top15": lp.get("top15_laps", np.nan) / laps,
        })
    return rows


def entry_rows(race):
    """The upcoming race's entry list from NASCAR's weekend feed, with the grid once it's set."""
    feed = weekend_feed(race["season"], race["race_id"])
    info = (feed or {}).get("weekend_race", [None])[0]
    if not info or not info.get("results"):
        return pd.DataFrame()
    pole = info.get("pole_winner_speed") or np.nan
    cars = info["results"]
    return pd.DataFrame([{
        **race, "field_size": len(cars), "car_number": str(r["car_number"]), "driver_id": r["driver_id"],
        "driver": r["driver_fullname"].replace("*", "").strip(), "team": r["team_name"],
        "make": r["car_make"], "start_position": r["starting_position"] or np.nan,
        "qualifying_speed": r.get("qualifying_speed") or np.nan, "pole_speed": pole,
    } for r in cars])


# ---------------------------------------------------------------- history

def _lagged(df, col, by, window=None):
    """Mean of `col` over each group's *earlier* rows (all of them, or the last `window`)."""
    def f(s):
        s = s.shift(1)
        return s.expanding().mean() if window is None else s.rolling(window, min_periods=1).mean()
    return df.groupby(by, sort=False)[col].transform(f)


def add_history(df):
    """Lagged form, track, track-type and team history, plus the pre-race features."""
    df = df.sort_values(["race_date", "race_id", "finish_position"]).reset_index(drop=True)
    done = df["finish_position"].notna()
    df["top10"] = (df["finish_position"] <= 10).astype(float).where(done)
    df["win"] = (df["finish_position"] == 1).astype(float).where(done)
    df["dnf"] = (df["status"] != "Running").astype(float).where(done)

    drv = ["driver_id"]
    df["career_starts"] = df.groupby(drv).cumcount()
    df["form_fin"] = _lagged(df, "finish_position", drv, FORM_RACES)
    df["form_rating"] = _lagged(df, "rating", drv, FORM_RACES)
    df["form_run_pos"] = _lagged(df, "avg_run_pos", drv, FORM_RACES)
    df["form_top10"] = _lagged(df, "top10", drv, FORM_RACES)
    df["form_dnf"] = _lagged(df, "dnf", drv, 2 * FORM_RACES)
    df["season_rating"] = _lagged(df, "rating", ["driver_id", "season"])

    trk = ["driver_id", "track_id"]
    df["trk_starts"] = df.groupby(trk).cumcount()
    df["trk_mean_fin"] = _lagged(df, "finish_position", trk)
    df["trk_mean_rating"] = _lagged(df, "rating", trk)
    df["trk_top10_rate"] = _lagged(df, "top10", trk)

    typ = ["driver_id", "track_type"]
    df["type_mean_fin"] = _lagged(df, "finish_position", typ, TYPE_RACES)
    df["type_mean_rating"] = _lagged(df, "rating", typ, TYPE_RACES)
    df["type_top15_laps"] = _lagged(df, "pct_laps_top15", typ, TYPE_RACES)

    # Team: average across the team's cars in each race, then its last few races.
    per_race = (df.groupby(["team", "race_date", "race_id"], as_index=False)
                .agg(t_fin=("finish_position", "mean"), t_rating=("rating", "mean"))
                .sort_values(["race_date", "race_id"]))
    per_race["team_form_fin"] = _lagged(per_race, "t_fin", ["team"], TEAM_RACES)
    per_race["team_form_rating"] = _lagged(per_race, "t_rating", ["team"], TEAM_RACES)
    df = df.merge(per_race[["team", "race_id", "team_form_fin", "team_form_rating"]],
                  on=["team", "race_id"], how="left")

    df["start_pct_field"] = df["start_position"] / df["field_size"]
    df["qual_gap_to_pole_pct"] = (df["pole_speed"] - df["qualifying_speed"]) / df["pole_speed"] * 100
    return df


def upcoming_rows(history, race, entries):
    """Feature rows for an upcoming race: its entry list with everyone's history attached."""
    raw_cols = [c for c in history.columns if c in entries.columns or c in
                ("finish_position", "status", "rating", "avg_run_pos", "pct_laps_top15", "laps_led")]
    combined = pd.concat([history[raw_cols], entries], ignore_index=True)
    feats = add_history(combined)
    return feats[feats["race_id"] == race["race_id"]].reset_index(drop=True)


def build(last_season=None):
    last_season = last_season or date.today().year
    rows = []
    for season in range(FIRST_SEASON, last_season + 1):
        for race in schedule(season, refresh=season == last_season):
            rows.extend(race_rows(race))
        print(f"  {season}: {sum(r['season'] == season for r in rows)} car entries")
    df = add_history(pd.DataFrame(rows))
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    table = build()
    print(f"Wrote {len(table)} rows: {table['race_id'].nunique()} races at {table['track_id'].nunique()} "
          f"tracks, {table['driver_id'].nunique()} drivers, through {table['race_date'].max()}.")
    print(table.drop_duplicates("race_id")["track_type"].value_counts().to_string())
