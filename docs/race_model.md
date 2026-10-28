# The models (`race_model.py`)

This is the heart of the app. It holds the two XGBoost models, the forecast for a
race, and the backtest that checks it all honestly.

## The two models

**The ranker: `XGBRanker` with `rank:pairwise`.** Instead of predicting each car on its
own, it learns from pairs inside each race: did car A finish ahead of car B? Every
race is one "group" (`qid`). The result is a score for each car, and a higher score
means further forward.

It only learns from **cars still running at the finish**. A car taken out in a wreck
is classified near the back by bad luck, not pace, and the DNF model and the
simulation handle that part separately.

**The DNF model: `XGBClassifier`** gives each car's chance of not finishing, learned
from every car in every race.

Both use XGBoost's **native categorical support** for the kind of track and the
manufacturer. That's how a single model can learn that recent form means a lot at a
short track and very little at a superspeedway.

### Settings

```python
ranker: n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=10,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0
dnf:    n_estimators=200, max_depth=3 (otherwise the same)
```

With 12,000 rows there's room for slightly deeper trees than the Talladega study
used, but the settings stay cautious: slow learning, L2 regularisation, and random
row and column sampling. One fit takes under two seconds.

## Forecasting a race

`forecast_race(models, rows, temperature)` gives every car's chances from 10,000
simulated races (see [simulation.md](simulation.md)).

If the starting grid isn't known yet (before qualifying), it draws **50 random grids**
and runs 200 simulations for each, so the uncertainty about where everyone starts is
part of the forecast. Once the grid is set, it uses the real one.

`finish_distribution(finish)` turns the simulations into each car's chance of
finishing in every position. The app's driver view charts this.

## The backtest

`backtest(df)` walks forward through **every race since January 2025** (70 races):

1. Train both models on every race *before* this one.
2. Fit the simulation temperature on the earlier backtest races.
3. Simulate this race 2,000 times with its real starting grid.
4. Record every car's chances next to what really happened.

It takes about two minutes.

`score(bt)` compares the model with simple baselines, and `score(bt, by="track_type")`
does the same for each kind of track:

| Score | The question | Baselines |
|---|---|---|
| Rank correlation | Does the predicted order match the real one? (1 = perfectly, 0 = random) | starting grid; recent form |
| Top-10 hits | Of the model's top 10, how many finished top 10? | top 10 on the grid; recent form |
| Top-10 log loss and Brier | Are the top-10 chances accurate? | the driver's recent top-10 rate |
| Win log loss | Are the win chances accurate? | everyone equal |
| DNF Brier | Are the DNF chances accurate? | the field's DNF rate |

## Explaining a forecast

- **`contributions(models, rows)`** gives SHAP values for the ranker score: how much
  each factor moves a car up or down the order. The app's driver view uses it.
- **`importance(models)`** gives each feature's share of the ranker's total gain.

## The building blocks

- **`prepare()` / `load_table()`** read `data/races.csv` and set the categorical columns.
- **`fit()`, `predict()`** train and score.
- **`save()` / `load()`** write and read `models/race/`.
- **`forecast_race()`, `finish_distribution()`** produce the forecast.
- **`backtest()`, `score()`** are the honesty checks.
