# Training and testing (`train.py`)

```powershell
.venv\Scripts\python train.py
```

This is the script that answers the big question: **is the model any good?** It
takes a few seconds.

For each of the two feature sets (`core`, then `enriched`) it:

1. **Tests the model race by race.** For each of the last 15 races, it trains only
   on the races before it, predicts that race, and compares the result with "his
   average so far." (The details are in [model.md](model.md#how-the-model-is-tested).)
2. **Trains the final models** on every completed race, ready for the upcoming one.
3. **Lists what the top-10 model relies on most**, as each feature's share of the
   total gain.

## What it saves

- **`models/core/` and `models/enriched/`:** the trained models in XGBoost's own
  JSON format (`top10.json`, `dnf.json`, `finish.json`). Each has a small
  `*_meta.json` beside it with the feature list and the fallback average.
- **`reports/backtest_core.csv` and `reports/backtest_enriched.csv`:** one row per
  tested race, showing what really happened, what the model said (`pred_*`) and what
  the baseline said (`base_*`).
- **`reports/metrics.csv`:** the scorecard for both feature sets.

`predict.py` uses the saved models. The dashboard trains its own copy each time it
starts, so it doesn't need these files.

## What you'll see

Here's the output from Oct 28, 2026, with the Oct 25 race included:

```
== core features (15) - lower is better
target   metric  model  baseline  model_better
 top10    brier  0.204     0.198         False
 top10 log_loss  0.634     0.599         False
   dnf    brier  0.322     0.276         False
   dnf log_loss  0.874     0.776         False
finish      mae 12.848    13.405          True
  Top-10 model leans most on: hist_mean_rating (62%), qual_gap_to_pole_pct (21%), prev_rating (15%), ...
```

## How to read it

**The finishing-position model now beats the simple average,** missing by 12.8
positions on average against 13.4. **The top-10 and DNF models still don't.** With
24 races at a track where crashes decide so much, that's the expected answer, not a
bug. It's useful to know, and it's the kind of thing plenty of projects never check.
It's also why version 2 learns from every car in the field instead
(see [field_train.md](field_train.md)).

**It leans mostly on driver rating history,** and hardly at all on start position.
That matches what anyone who watches superspeedway racing would tell you: starting
spot barely matters when the whole field runs in a pack.

## When to run it again

- After you add a new race result (run `enrich.py` first).
- After you change the features or `XGB_PARAMS` in `model.py`.
