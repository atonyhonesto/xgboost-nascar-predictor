"""Backtest and train the app's models, then forecast the next race.

  1. Walk-forward backtest over every race since Jan 2025: for each race, train on all
     earlier races, simulate it, and score every car against simple baselines.
  2. Fit the simulation temperature on those out-of-sample races.
  3. Train the final ranker + DNF models on every completed race and save them.
  4. Forecast the next race on NASCAR's schedule (entry list and grid from its feed).

Outputs:
  models/race/{ranker,dnf}.json, meta.json      saved models + temperature
  reports/race_backtest.csv                     per car, per tested race
  reports/race_metrics.csv                      model vs baseline scorecard
  reports/race_metrics_by_type.csv              the same, by kind of track
  reports/race_calibration.csv                  predicted vs actual top-10 rate
  reports/race_next.csv, race_next.json         the next race's forecast

Usage:  python race_train.py      (run race_data.py first)
"""
import json

import race_model as rm
import simulation as sim
from race_data import entry_rows, next_race, upcoming_rows


def main():
    df = rm.load_table()
    rm.REPORTS.mkdir(exist_ok=True)
    print(f"{len(df):,} car entries in {df['race_id'].nunique()} races, through {df['race_date'].max()}.")
    print(f"Backtesting every race since {rm.BACKTEST_FROM} (this takes a couple of minutes)...")

    bt, oof = rm.backtest(df, progress=lambda i, n, label: print(f"  {i}/{n} {label}") if i % 10 == 0 or i == n else None)
    scores, by_type = rm.score(bt), rm.score(bt, by="track_type")
    bt.to_csv(rm.REPORTS / "race_backtest.csv", index=False)
    scores.to_csv(rm.REPORTS / "race_metrics.csv", index=False)
    by_type.to_csv(rm.REPORTS / "race_metrics_by_type.csv", index=False)
    sim.calibration(bt).to_csv(rm.REPORTS / "race_calibration.csv", index=False)
    print(scores.drop(columns="races").to_string(index=False, float_format="%.3f"))
    print("\nRank correlation by kind of track (model vs starting grid):")
    rc = by_type[by_type["baseline_name"] == "starting grid"]
    rc = rc[rc["metric"].str.startswith("Rank")]
    for _, r in rc.iterrows():
        print(f"  {r['track_type']:<14} {r['model']:.3f} vs {r['baseline']:.3f}  ({r['races']} races)")

    temperature = sim.fit_temperature(oof)
    models = rm.fit(df)
    rm.save(models, temperature, {"trained_races": int(df["race_id"].nunique()),
                                  "trained_through": df["race_date"].max()})
    top = rm.importance(models).head(6)
    print(f"\nSimulation temperature: {temperature}")
    print("Ranker leans most on: " + ", ".join(f"{f} ({v:.0%})" for f, v in top.items()))

    race = next_race(df["race_date"].max())
    if race is None:
        print("\nNo upcoming race on NASCAR's published schedule yet.")
        return
    entries = entry_rows(race)
    if entries.empty:
        print(f"\nNext race: {race['race_name']} at {race['track']} ({race['race_date']}); "
              "entry list not posted yet.")
        return
    rows = upcoming_rows(df, race, entries)
    fc, finish = rm.forecast_race(models, rows, temperature)
    fc = fc.join(rm.finish_distribution(finish))
    fc.to_csv(rm.REPORTS / "race_next.csv", index=False)
    (rm.REPORTS / "race_next.json").write_text(json.dumps(race, indent=2))
    grid = "grid from qualifying" if rows["start_position"].notna().all() else "grid not set: random grids"
    print(f"\n{race['race_name']} at {race['track']}, {race['race_date']} ({len(fc)} cars, {grid}):")
    for _, r in fc.sort_values("exp_finish").head(10).iterrows():
        print(f"  #{r['car_number']:>3} {r['driver']:<24} exp {r['exp_finish']:5.1f}  "
              f"win {r['p_win']:5.1%}  top10 {r['p_top10']:4.0%}  DNF {r['p_dnf']:4.0%}")


if __name__ == "__main__":
    main()
