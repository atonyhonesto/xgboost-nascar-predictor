"""Build the whole-field table: one row per car per Talladega race, every driver.

This is the v2 dataset, in the spirit of the F1 race-predictor projects. It comes
entirely from NASCAR's public feeds (via fetch_nascar.py):

  results (weekend feed)  every car's start, finish, status, team, make   2018 on
  loopstats               driver rating, average running position          2019 on
  lap-times               share of laps run in the top 10                  2020 on

As in v1, anything measured during a race enters only as history: each driver's
and each team's record in *earlier* Talladega races. The upcoming race is appended
as projected rows (the most recent race's entry list, grid unknown).

Usage:  python field_data.py      -> data/field_talladega.csv
"""
import numpy as np
import pandas as pd

from fetch_nascar import CACHE, fetch_race, loop_stats, race_results, talladega_races

OUT = CACHE.parent / "field_talladega.csv"
FIRST_SEASON = 2018  # earliest season with every car's results in the weekend feed
NEXT_RACE_ID = "next"


def race_rows(race):
    """One row per classified car (cars that didn't qualify have finishing_position 0)."""
    info = race_results(race["season"], race["race_id"])
    if info is None:
        return []
    loop = {d["driver_id"]: d for d in (loop_stats(race["season"], race["race_id"]) or [])}
    laps = fetch_race(race["season"], race["race_id"], feeds=("lap-times",))["lap-times"]
    top10_share = {}
    for car in (laps or {}).get("laps", []):
        pos = [lap["RunningPos"] for lap in car.get("Laps", []) if lap.get("Lap", 0) > 0]
        if pos:
            top10_share[str(car["Number"])] = float(np.mean(np.array(pos) <= 10))

    cars = [r for r in info["results"] if r.get("finishing_position")]
    rows = []
    for r in cars:
        lp = loop.get(r["driver_id"], {})
        rows.append({
            "race_id": race["race_id"], "season": race["season"], "race_date": info["race_date"][:10],
            "season_half": race["season_half"], "race_name": info["race_name"],
            "car_number": str(r["car_number"]), "driver_id": r["driver_id"],
            "driver": r["driver_fullname"].replace("*", "").strip(), "team": r["team_name"],
            "make": r["car_make"], "start_position": r["starting_position"],
            "qualifying_speed": r.get("qualifying_speed") or np.nan,
            "pole_speed": info.get("pole_winner_speed") or np.nan, "field_size": len(cars),
            "finish_position": r["finishing_position"], "laps_completed": r["laps_completed"],
            "laps_led": r["laps_led"], "status": r["finishing_status"],
            "rating": lp.get("rating", np.nan), "avg_run_pos": lp.get("avg_ps", np.nan),
            "pct_laps_top10": top10_share.get(str(r["car_number"]), np.nan),
        })
    return rows


def projected_entry(df):
    """Upcoming-race rows: the last race's entry list, with the grid and results unknown."""
    last = df[df["race_date"] == df["race_date"].max()]
    from enrich import NEXT_RACE  # date / spring-fall of the race being forecast
    proj = last[["car_number", "driver_id", "driver", "team", "make"]].copy()
    proj["race_id"], proj["season"] = NEXT_RACE_ID, NEXT_RACE["season"]
    proj["race_date"], proj["season_half"] = NEXT_RACE["race_date"], NEXT_RACE["season_half"]
    proj["race_name"], proj["field_size"] = "Next Talladega race (projected entry list)", len(proj)
    return proj


def add_history(df):
    """Lagged driver and team history. Each row only sees earlier races."""
    df = df.sort_values(["race_date", "finish_position"]).reset_index(drop=True)
    done = df["finish_position"].notna()
    df["top10"] = (df["finish_position"] <= 10).astype(float).where(done)
    df["win"] = (df["finish_position"] == 1).astype(float).where(done)
    df["dnf"] = (df["status"] != "Running").astype(float).where(done)

    # Driver: previous value and running mean over the driver's earlier Talladega races.
    g = df.groupby("driver_id")
    df["drv_starts"] = g.cumcount()
    for src, prev, mean in (("finish_position", "drv_prev_fin", "drv_mean_fin"),
                            ("rating", "drv_prev_rating", "drv_mean_rating"),
                            ("avg_run_pos", None, "drv_mean_run_pos"),
                            ("pct_laps_top10", None, "drv_mean_pct_laps_top10"),
                            ("laps_led", None, "drv_mean_laps_led"),
                            ("top10", None, "drv_top10_rate"),
                            ("dnf", None, "drv_dnf_rate")):
        shifted = g[src].shift(1)
        if prev:
            df[prev] = shifted
        df[mean] = shifted.groupby(df["driver_id"]).transform(lambda s: s.expanding().mean())

    # Team: average across the team's cars in each race, then a running mean of earlier races.
    per_race = (df.groupby(["team", "race_date"], as_index=False)
                .agg(t_fin=("finish_position", "mean"), t_rating=("rating", "mean"), t_top10=("top10", "mean"))
                .sort_values("race_date"))
    tg = per_race.groupby("team")
    for src, out in (("t_fin", "team_mean_fin"), ("t_rating", "team_mean_rating"), ("t_top10", "team_top10_rate")):
        per_race[out] = tg[src].transform(lambda s: s.shift(1).expanding().mean())
    df = df.merge(per_race[["team", "race_date", "team_mean_fin", "team_mean_rating", "team_top10_rate"]],
                  on=["team", "race_date"], how="left")

    # Pre-race features.
    df["start_pct_field"] = df["start_position"] / df["field_size"]
    df["qual_gap_to_pole_pct"] = (df["pole_speed"] - df["qualifying_speed"]) / df["pole_speed"] * 100
    df["is_playoff"] = (df["season_half"] == "fall").astype(int)  # Talladega's fall race is a playoff race
    df["is_gen7"] = (df["season"] >= 2022).astype(int)
    return df


def build():
    rows = []
    for season in range(FIRST_SEASON, 2027):
        for race in talladega_races(season):
            rows.extend(race_rows(race))
    df = pd.DataFrame(rows)
    df = pd.concat([df, projected_entry(df)], ignore_index=True)
    df = add_history(df)
    df.to_csv(OUT, index=False)
    return df


if __name__ == "__main__":
    table = build()
    done = table[table["race_id"] != NEXT_RACE_ID]
    print(f"Wrote {len(table)} rows: {done['race_id'].nunique()} races, {len(done)} car-race entries, "
          f"{done['driver_id'].nunique()} drivers, plus {len(table) - len(done)} projected entries.")
    print(done.groupby("season")["race_id"].nunique().to_string())
