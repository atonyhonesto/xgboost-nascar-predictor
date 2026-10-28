# Training and testing (`race_train.py`)

```powershell
.venv\Scripts\python race_train.py
```

Run this after `race_data.py`, ideally after every race. It takes about three
minutes and does four things:

1. **Backtests** every race since January 2025, each forecast using only earlier races.
2. **Fits the simulation temperature** on those out-of-sample races.
3. **Trains the final models** on every completed race.
4. **Forecasts the next race** on NASCAR's schedule from its entry list, with the
   real grid if qualifying is done.

## What it saves

- `models/race/ranker.json`, `dnf.json`, `meta.json`: the models, the temperature
  and how many races they were trained on.
- `reports/race_backtest.csv`: every car's forecast in every tested race, next to the result.
- `reports/race_metrics.csv` and `race_metrics_by_type.csv`: the scorecard, overall and by kind of track.
- `reports/race_calibration.csv`: predicted vs actual top-10 rates.
- `reports/race_next.csv` and `race_next.json`: the next race's forecast, including
  each car's chance of every finishing position.

The app reads all of these. Its **Refresh from NASCAR** button redoes step 4 live.

## What you'll see (Oct 28, 2026)

```
                                             metric  model  baseline                 baseline_name  model_better
Rank correlation with the result (higher is better)  0.422     0.369                 starting grid          True
Rank correlation with the result (higher is better)  0.422     0.309 recent form (last 5 finishes)          True
      Predicted top 10 that finished top 10 (of 10)  4.957     4.543                 starting grid          True
      Predicted top 10 that finished top 10 (of 10)  4.957     4.286 recent form (last 5 finishes)          True
                  Top-10 log loss (lower is better)  0.512     0.552   driver's recent top-10 rate          True
               Top-10 Brier score (lower is better)  0.170     0.184   driver's recent top-10 rate          True
                     Win log loss (lower is better)  0.112     0.122                everyone equal          True
                  DNF Brier score (lower is better)  0.112     0.114                field DNF rate          True

Rank correlation by kind of track (model vs starting grid):
  short          0.551 vs 0.432  (19 races)
  intermediate   0.458 vs 0.406  (29 races)
  superspeedway  0.093 vs 0.121  (12 races)
  road           0.464 vs 0.434  (10 races)

Ranker leans most on: start_pct_field (14%), type_mean_rating (13%), type_mean_fin (11%),
team_form_fin (10%), season_rating (8%), team_form_rating (6%)

Xfinity 500 at Martinsville Speedway, 2026-11-01 (37 cars, grid not set: random grids):
  # 12 Ryan Blaney              exp  10.8  win  6.8%  top10  58%  DNF   4%
  # 11 Denny Hamlin             exp  13.1  win  4.9%  top10  47%  DNF   5%
  ...
```

## How to read it

**Every score beats its baseline,** including win and DNF chances, which the
Talladega-only models never managed. The jump came from data: 12,000 entries from
every kind of track, instead of a few hundred from one.

**Short tracks are where it shines.** A rank correlation of 0.55 means the predicted
order and the real one line up well. That fits how short-track racing works: car speed
and driver skill show over 500 laps at Martinsville.

**Superspeedways are still a lottery.** At Daytona and Talladega the model is a touch
*worse* than just using the starting grid. That's an honest result and it's left in on
purpose.

**What it relies on:** starting spot, how the driver has rated at similar tracks, team
form and this season's rating. Form at the *same kind of track* matters more than
form at any track, which is why the track types exist.

## When to run it again

After each race, run `race_data.py` and then `race_train.py`. In race week, the app's
refresh button handles new entry lists and grids without retraining.
