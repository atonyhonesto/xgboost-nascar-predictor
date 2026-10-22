# train.py: backtest and save the models

```powershell
.venv\Scripts\python train.py
```

For **each feature set** (`core`, then `enriched`) it:

1. Runs the walk-forward backtest (see [model.md](model.md#the-backtest)) on the last
   15 races and prints model vs baseline scores.
2. Trains the final top-10, DNF and finish models on **all** completed races.
3. Prints the features the top-10 model relies on most (share of total gain).

It takes a few seconds.

## Outputs

| Path | Contents |
|---|---|
| `models/core/`, `models/enriched/` | `top10.json`, `dnf.json`, `finish.json` (XGBoost's native format) plus a `*_meta.json` for each, with the feature list and base rate |
| `reports/backtest_core.csv`, `reports/backtest_enriched.csv` | One row per backtested race: actual outcome, model prediction (`pred_*`) and baseline (`base_*`) |
| `reports/metrics.csv` | The score table for both feature sets |

`predict.py` reads `models/`. The dashboard trains its own copy in memory, so it
doesn't depend on these files.

## Example output (Oct 21, 2026)

```
== core features (15) - lower is better
target   metric  model  baseline  model_better
 top10    brier  0.203     0.171         False
 top10 log_loss  0.633     0.537         False
   dnf    brier  0.332     0.288         False
   dnf log_loss  0.898     0.803         False
finish      mae 12.968    12.898         False
  Top-10 model leans most on: hist_mean_rating (64%), prev_rating (19%), qual_gap_to_pole_pct (7%), ...
```

## Reading the result
- **`model_better = False` everywhere** means that, so far, "his average so far"
  predicts as well as XGBoost or better. With 23 races and pack-racing crashes, that
  is expected. It is not a bug.
- The model mostly uses **driver rating history**. Start position barely matters at
  Talladega, which matches how superspeedway racing works.

## When to re-run
- After adding a new race result (run `enrich.py` first).
- After changing features or `XGB_PARAMS` in `model.py`.
