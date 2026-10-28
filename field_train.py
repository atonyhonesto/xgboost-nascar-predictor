"""Backtest and train the whole-field models, then forecast the next race.

  1. Walk-forward backtest over every race with at least 6 earlier races: train on
     the past, simulate the race 4,000 times, score every car against two baselines.
  2. Fit the simulation temperature on those out-of-sample races.
  3. Train the final ranker + DNF models on every completed race and save them.
  4. Simulate the next race 10,000 times and save each car's chances.

Outputs:
  models/field/{ranker,dnf}.json, meta.json     saved models + temperature
  reports/field_backtest.csv                    per car, per tested race
  reports/field_metrics.csv                     model vs baseline scorecard
  reports/field_calibration.csv                 predicted vs actual top-10 rate
  reports/field_next_race.csv                   the forecast for the next race

Usage:  python field_train.py      (run field_data.py first)
"""
import field_model as fm
from model import REPORTS


def main():
    past, upcoming = fm.split(fm.load_table())
    REPORTS.mkdir(exist_ok=True)
    n_races = past["race_id"].nunique()
    print(f"{len(past)} car-race entries in {n_races} races; "
          f"backtesting the last {n_races - fm.MIN_TRAIN_RACES}.\n")

    bt, oof = fm.backtest(past)
    scores = fm.score(bt)
    bt.to_csv(REPORTS / "field_backtest.csv", index=False)
    scores.to_csv(REPORTS / "field_metrics.csv", index=False)
    fm.calibration(bt).to_csv(REPORTS / "field_calibration.csv", index=False)
    print(scores.to_string(index=False, float_format="%.3f"))

    temperature = fm.fit_temperature(oof)
    models = fm.fit(past)
    fm.save(models, temperature, n_races)
    top = fm.importance(models).head(6)
    print(f"\nSimulation temperature: {temperature} (1.0 = take the ranker at face value)")
    print("Ranker leans most on: " + ", ".join(f"{f} ({v:.0%})" for f, v in top.items()))

    if len(upcoming):
        fc = fm.forecast_race(models, upcoming, temperature)
        fc.to_csv(REPORTS / "field_next_race.csv", index=False)
        grid = "grid known" if upcoming["start_position"].notna().all() else "grid unknown, random grids"
        print(f"\nNext race ({fc['race_date'].iloc[0]}, {grid}), top 10 by expected finish:")
        show = fc.sort_values("exp_finish").head(10)
        for _, r in show.iterrows():
            print(f"  #{r['car_number']:>3} {r['driver']:<24} exp {r['exp_finish']:5.1f}  "
                  f"win {r['p_win']:5.1%}  top10 {r['p_top10']:5.1%}  DNF {r['p_dnf']:5.1%}")
    print("\nSaved models to models/field/ and reports to reports/. "
          "Forecast any race with `python field_predict.py`.")


if __name__ == "__main__":
    main()
