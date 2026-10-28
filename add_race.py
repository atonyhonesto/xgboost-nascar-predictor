"""Append a finished race to data/larson_talladega_raw.csv from NASCAR's public feeds.

Builds the row from the race's weekend feed (results, stages, qualifying) and
loopstats feed (driver rating, passes, running positions), so new races don't
have to be copied in by hand.

Usage:  python add_race.py --season 2026 --race-id 5631 [--car 5]
Then:   python enrich.py && python train.py
"""
import argparse

import pandas as pd

from fetch_nascar import BASE, CACHE, CUP, get_json, loop_stats, race_results

RAW = CACHE.parent / "larson_talladega_raw.csv"
LAP_MILES = 2.66  # Talladega lap length, used to turn qualifying speed into a lap time


def race_number(season, race_id):
    """Position of the race among that season's points races (exhibitions excluded)."""
    schedule = get_json(f"{BASE}/{season}/race_list_basic.json", CACHE / f"{season}_race_list.json")
    points = sorted((r for r in schedule[f"series_{CUP}"] if r.get("race_type_id") == 1),
                    key=lambda r: r["race_date"])
    return [r["race_id"] for r in points].index(race_id) + 1


def build_row(season, race_id, car, previous):
    race = race_results(season, race_id)
    me = next(r for r in race["results"] if str(r["car_number"]) == car)
    winner = next(r for r in race["results"] if r["finishing_position"] == 1)
    loop = next(d for d in loop_stats(season, race_id) if d["driver_id"] == me["driver_id"])
    stages = {s["stage_number"]: next((x for x in s["results"] if str(x["car_number"]) == car), {})
              for s in race.get("stage_results", [])}
    laps, month = race["actual_laps"], int(race["race_date"][5:7])
    qual = me.get("qualifying_speed") or None
    return {
        "season": season, "race_date": race["race_date"][:10], "race_name": race["race_name"],
        "race_number_in_season": race_number(season, race_id),
        "season_half": "spring" if month <= 6 else "fall",
        "playoff_race": "Y" if race.get("playoff_round") else "N",
        "car_number": car, "team": me["team_name"], "manufacturer": me["car_make"],
        "crew_chief": me["crew_chief_fullname"], "sponsor": me["sponsor"].strip(),
        "start_position": me["starting_position"], "finish_position": me["finishing_position"],
        "laps_completed": me["laps_completed"], "total_race_laps": laps,
        "scheduled_laps": race["scheduled_laps"], "laps_led": me["laps_led"],
        "status": me["finishing_status"].lower(),
        "overtime": "Y" if laps > race["scheduled_laps"] else "N", "rain_shortened": "N",
        "cautions_total": race["number_of_cautions"], "caution_laps": race["number_of_caution_laps"],
        "lead_changes": race["number_of_lead_changes"],
        "stage1_finish": stages.get(1, {}).get("finishing_position"),
        "stage2_finish": stages.get(2, {}).get("finishing_position"),
        "stage1_points": stages.get(1, {}).get("stage_points"),
        "stage2_points": stages.get(2, {}).get("stage_points"),
        "total_points_earned": me["points_earned"], "points_rank_after_race": me["points_position"],
        "driver_rating": round(loop["rating"], 1), "avg_running_position": round(loop["avg_ps"]),
        "mid_race_position": loop["mid_ps"], "high_position": loop["best_ps"],
        "low_position": loop["worst_ps"], "pass_differential": loop["passing_diff"],
        "green_flag_passes": loop["passes_gf"], "green_flag_times_passed": loop["passed_gf"],
        "quality_passes": loop["quality_passes"],
        "pct_quality_passes": round(loop["quality_passes"] / loop["passes_gf"] * 100, 1),
        "fastest_laps_run": loop["fast_laps"], "laps_in_top15": loop["top15_laps"],
        "pct_laps_top15": round(loop["top15_laps"] / loop["laps"] * 100, 1),
        "pct_laps_led": round(loop["lead_laps"] / loop["laps"] * 100, 1),
        "qualifying_rank": me.get("qualifying_position") or None,
        "qualifying_lap_time_sec": round(LAP_MILES / qual * 3600, 3) if qual else None,
        "qualifying_speed_mph": qual, "pole_speed_mph": race.get("pole_winner_speed") or None,
        "race_avg_speed_mph": race["average_speed"],
        # Goodyear codes aren't in the feeds; carry the last race's over until confirmed.
        "tire_left_code": previous["tire_left_code"], "tire_right_code": previous["tire_right_code"],
        "tire_set_limit": previous["tire_set_limit"],
        "tire_code_basis": "inferred (carried over from previous race; not yet confirmed)",
        "field_size": race["number_of_cars_in_field"], "race_winner": winner["driver_fullname"],
        "notes": (f"Margin {race['margin_of_victory']}s"
                  + (f", overtime ({laps} laps)" if laps > race["scheduled_laps"] else "")
                  + ". Built by add_race.py from NASCAR feeds; qualifying lap time derived from speed."),
        "source_urls": f"{BASE}/{season}/{CUP}/{race_id}/weekend-feed.json"
                       f" | https://cf.nascar.com/loopstats/prod/{season}/{CUP}/{race_id}.json",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--race-id", type=int, required=True, help="cf.nascar.com race_id")
    p.add_argument("--car", default="5")
    args = p.parse_args()

    raw = pd.read_csv(RAW, dtype={"car_number": str})
    row = build_row(args.season, args.race_id, args.car, raw.iloc[-1])
    if (raw["race_date"] == row["race_date"]).any():
        raise SystemExit(f"{row['race_date']} is already in {RAW.name}; nothing added.")
    raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)[raw.columns]
    raw.to_csv(RAW, index=False)
    print(f"Added {row['race_date']} {row['race_name']}: started P{row['start_position']}, "
          f"finished P{row['finish_position']} ({row['status']}), rating {row['driver_rating']}.")
    print("Next: python enrich.py, then python train.py")


if __name__ == "__main__":
    main()
