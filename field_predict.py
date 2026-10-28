"""Per-car forecast for a Talladega race, F1-predictor style: every car's chances.

Run field_train.py first (it saves the models and simulation temperature).

  python field_predict.py                                # next race, projected entry list, grid unknown
  python field_predict.py --season 2027 --race-id 5700   # a real race: entry list + grid from NASCAR
  python field_predict.py --sims 20000 --csv out.csv     # more simulations, save the table

With --race-id, the entry list comes from NASCAR's weekend feed. Once qualifying is
done the starting grid is used; before that, random grids are simulated.
"""
import argparse

import numpy as np
import pandas as pd

import field_model as fm
from fetch_nascar import weekend_feed


def entry_rows(season, race_id, history):
    """Rows for a scheduled race: its entry list, with each driver's latest history attached."""
    feed = weekend_feed(season, race_id)
    race = (feed or {}).get("weekend_race", [None])[0]
    if not race or not race.get("results"):
        raise SystemExit("NASCAR hasn't posted an entry list for that race yet.")
    entries = pd.DataFrame([{
        "car_number": str(r["car_number"]), "driver_id": r["driver_id"],
        "driver": r["driver_fullname"].replace("*", "").strip(), "team": r["team_name"], "make": r["car_make"],
        "start_position": r["starting_position"] or np.nan, "qualifying_speed": r.get("qualifying_speed") or np.nan,
    } for r in race["results"]])
    # The projected rows in the table already carry everyone's history through the last race.
    hist_cols = [c for c in fm.FEATURES if c.startswith(("drv_", "team_"))]
    latest = history.drop_duplicates("driver_id", keep="last").set_index("driver_id")[hist_cols]
    rows = entries.join(latest, on="driver_id")
    rows["drv_starts"] = rows["drv_starts"].fillna(0)  # first Talladega start
    rows["field_size"] = len(rows)
    rows["start_pct_field"] = rows["start_position"] / rows["field_size"]
    pole = race.get("pole_winner_speed") or np.nan
    rows["qual_gap_to_pole_pct"] = (pole - rows["qualifying_speed"]) / pole * 100
    month = int(race["race_date"][5:7])
    rows["is_playoff"], rows["is_gen7"] = int(month > 6), int(season >= 2022)
    rows["race_date"], rows["race_name"] = race["race_date"][:10], race["race_name"]
    rows["make"] = pd.Categorical(rows["make"], categories=fm.MAKES)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int)
    p.add_argument("--race-id", type=int)
    p.add_argument("--sims", type=int, default=fm.N_SIMS)
    p.add_argument("--csv", help="also save the full table to this file")
    args = p.parse_args()

    table = fm.load_table()
    models, meta = fm.load()
    _, upcoming = fm.split(table)
    if args.race_id:
        rows = entry_rows(args.season, args.race_id, upcoming)
    else:
        rows = upcoming
    fc = fm.forecast_race(models, rows, meta["temperature"], n_sims=args.sims)

    grid = "starting grid from qualifying" if rows["start_position"].notna().all() else "grid unknown: random grids"
    print(f"{fc['race_name'].iloc[0]}, {fc['race_date'].iloc[0]} · {len(fc)} cars · {grid} · "
          f"{args.sims:,} simulations\n")
    fc = fc.sort_values("exp_finish").reset_index(drop=True)
    out = pd.DataFrame({
        "Rank": fc.index + 1, "Car": "#" + fc["car_number"], "Driver": fc["driver"], "Team": fc["team"],
        "Start": fc["start_position"].map(lambda v: "" if pd.isna(v) else f"P{int(v)}"),
        "Exp. finish": fc["exp_finish"].map("{:.1f}".format),
        "Win": fc["p_win"].map("{:.1%}".format), "Top 3": fc["p_top3"].map("{:.0%}".format),
        "Top 5": fc["p_top5"].map("{:.0%}".format), "Top 10": fc["p_top10"].map("{:.0%}".format),
        "DNF": fc["p_dnf"].map("{:.0%}".format),
    })
    print(out.to_string(index=False))
    print("\nTalladega is close to a lottery: even the favourite rarely tops ~5% to win. "
          "See reports/field_metrics.csv for how this model scores against simple baselines.")
    if args.csv:
        fc.to_csv(args.csv, index=False)
        print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
