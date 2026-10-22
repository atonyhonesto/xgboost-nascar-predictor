"""Backtest and train the XGBoost models for Larson at Talladega.

For each feature set ("core" and "enriched"):
  1. Walk-forward backtest: train on races before race i, predict race i, compare
     against a plain historical-average baseline.
  2. Train final models on every completed race and save them to models/<set>/.

Outputs:
  models/<set>/{top10,dnf,finish}.json   XGBoost models (+ *_meta.json)
  reports/backtest_<set>.csv             race-by-race backtest predictions
  reports/metrics.csv                    model vs baseline scores for both sets

Usage:  python train.py
"""
import pandas as pd

from model import (FEATURE_SETS, MIN_TRAIN_RACES, REPORTS, TARGETS, backtest, load_table,
                   save_all, score, split, train_all)


def main():
    past, upcoming = split(load_table())
    REPORTS.mkdir(exist_ok=True)
    print(f"{len(past)} completed races; backtesting on the last {len(past) - MIN_TRAIN_RACES}.\n")

    all_scores = []
    for name in FEATURE_SETS:
        bt = backtest(past, name)
        bt.to_csv(REPORTS / f"backtest_{name}.csv", index=False)
        sc = score(bt).assign(feature_set=name)
        all_scores.append(sc)

        print(f"== {name} features ({len(FEATURE_SETS[name])}) - lower is better")
        print(sc[["target", "metric", "model", "baseline", "model_better"]]
              .to_string(index=False, float_format="%.3f"))

        models = train_all(past, name)
        save_all(models, name)
        top = models["top10"].importance().head(6)
        print("  Top-10 model leans most on: " + ", ".join(f"{f} ({v:.0%})" for f, v in top.items()))
        print()

    pd.concat(all_scores).to_csv(REPORTS / "metrics.csv", index=False)
    print(f"Saved models to models/ and reports to {REPORTS.name}/.")
    if len(upcoming):
        print(f"Ready to forecast {upcoming['race_date'].iloc[0]}: run `python predict.py` "
              f"or `streamlit run app.py`.")


if __name__ == "__main__":
    main()
