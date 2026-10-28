# Training the whole-field model (`field_train.py`)

```powershell
.venv\Scripts\python field_train.py
```

Run this after `field_data.py`. It takes about 10 seconds and does four things:

1. **Backtests** the whole-field model on the last 12 races and prints the scorecard.
2. **Fits the simulation temperature** on those out-of-sample races.
3. **Trains the final models** on all 18 races and saves them to `models/field/`.
4. **Forecasts the next race** with 10,000 simulations.

## What it saves

- `models/field/ranker.json`, `dnf.json`: the two XGBoost models.
- `models/field/meta.json`: the feature list, the fitted temperature and how many races were used.
- `reports/field_backtest.csv`: every car's chances in every tested race, next to what happened.
- `reports/field_metrics.csv`: the scorecard.
- `reports/field_calibration.csv`: predicted vs actual top-10 rates.
- `reports/field_next_race.csv`: the forecast for the next race.

The dashboard's two Whole field tabs read these files.

## The scorecard (Oct 28, 2026)

```
                                             metric  model  baseline           baseline_name  model_better
Rank correlation with the result (higher is better)  0.093     0.091           starting grid          True
Rank correlation with the result (higher is better)  0.093     0.040 driver's average finish          True
      Predicted top 10 that finished top 10 (of 10)  2.917     2.750           starting grid          True
                  Top-10 log loss (lower is better)  0.557     0.607  driver's top-10 record          True
               Top-10 Brier score (lower is better)  0.186     0.204  driver's top-10 record          True
                  Top-10 log loss (lower is better)  0.557     0.569          everyone equal          True
               Top-10 Brier score (lower is better)  0.186     0.190          everyone equal          True
                     Win log loss (lower is better)  0.120     0.119          everyone equal         False
                  DNF Brier score (lower is better)  0.170     0.163          field DNF rate         False
```

## How to read it

**The whole-field model beats every simple baseline on finishing order and top-10
chances.** That's something the single-car model never managed. More data, from every
car, made the difference.

**But the margins are small, and that's Talladega.** A rank correlation of 0.09 means
the predicted order matches the real one only slightly better than chance. Of the 10
cars it picks for the top 10, about 3 make it, against 2.75 for the top 10 on the grid.

**It can't predict winners or crashes better than "everyone's equal".** Win chances
and DNF chances are no better than giving every car the same odds. At a track where
the winner is often whoever survives the last-lap wreck, that's an honest result.

**The chances are well calibrated.** Cars given about a 25% top-10 chance finished top
10 about 26% of the time, so the percentages can be read at face value.

**What the ranker relies on:** start position, the team's average finish and rating,
the driver's last rating, and laps led. Team strength matters more than in version 1,
because it covers drivers with little Talladega history of their own.

## When to run it again

After each new race: `field_data.py`, then `field_train.py`.
