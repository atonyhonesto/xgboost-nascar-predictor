# The models (`model.py`)

This is the heart of the project. `train.py`, `predict.py` and the dashboard all
import from here, so they always agree on which features to use, how to train and
how to score. If you only read one file, read this one.

## What the model gets to see

The golden rule is that **everything must be known before the green flag drops.**
Anything measured during a race, such as driver rating, laps led or running
position, only comes in as history from *earlier* races.

There are two sets of features, so you can compare them.

### The `core` set (15 features, history back to 2014)

- **Where it starts:** `start_pct_field` (start position ÷ field size, so 0.05
  means near the front) and `qual_gap_to_pole_pct` (how far its qualifying speed was
  below pole, in %). The qualifying gap is blank when there were no time trials.
- **Context:** whether it's a playoff race (`is_playoff`), whether it's the current
  Gen7 car (`is_gen7`, 2022 on), whether he's driving for Hendrick (`is_hendrick`),
  and how many times he's raced here before (`prior_talladega_starts`).
- **How he's done here before:**
  - last finish (`prev_fin`) and average finish (`hist_mean_fin`)
  - last and average driver rating (`prev_rating`, `hist_mean_rating`)
  - last and average running position (`prev_avg_run_pos`, `hist_mean_avg_run_pos`)
  - average laps led (`hist_mean_laps_led`)
- **How often things went wrong:** how often he failed to finish (`hist_dnf_rate`)
  and how often that was a crash (`hist_crash_rate`).

### The `enriched` set (core plus 8 more)

These come from NASCAR's lap-by-lap, pit stop and race-control feeds, which only go
back to 2021:
- **Running near the front:** the share of laps in the top 10 and top 5
  (`prev_pct_laps_top10`, `hist_mean_pct_laps_top10`, `prev_pct_laps_top5`)
- **The race itself:** where he ran late in the race (`hist_mean_late_race_avg_pos`)
  and how much his position swung (`hist_mean_run_pos_std`)
- **Pit road:** how quick the pit crew is (`hist_mean_median_tire_stop_sec`) and
  how many places he gains or loses on pit road (`hist_mean_pit_road_pos_net`)
- **Trouble:** how often race control mentions the car in an incident
  (`hist_mean_incident_notes`)

[enrich.md](enrich.md) explains how each one is measured.

The earlier races don't have these numbers, and that's fine: XGBoost treats blanks
as "missing" and learns which way to send them at each split. Nothing has to be
filled in by hand.

## What it predicts

There are three separate models:
- **`top10`:** an `XGBClassifier` that gives the chance of finishing 1st to 10th.
- **`dnf`:** an `XGBClassifier` that gives the chance of not running at the finish.
- **`finish`:** an `XGBRegressor` that gives the expected finishing position,
  kept between 1 and the field size.

## Why the settings are so cautious

```python
n_estimators=80, learning_rate=0.05, max_depth=2, min_child_weight=1.0,
subsample=1.0, colsample_bytree=0.8, reg_lambda=5.0, random_state=42
```

Twenty-three races is very little to learn from. Give XGBoost deep trees and it
will happily memorise every quirk of the past, like "he crashed in 2019, so
probably always crash in the fall," and then fall apart on the next race. So the
trees are only two levels deep, it learns slowly, a strong L2 penalty keeps
predictions modest, and each tree sees a random 80% of the features.

These settings were deliberately **not** tuned to make the backtest look good.
With only 15 test races, tuning would just teach the model the test.

A small side effect: the random 80% column sampling works slightly differently on
Windows and Linux, even with the same seed. The live app, which runs on Linux, can
show numbers a point or so off from what you get on a Windows PC.

## The building blocks

These are the main pieces, in roughly the order you'd use them:

- **`load_table()`** reads the model table and adds the simple flags
  (`is_playoff`, `is_gen7`, `is_hendrick`) and a friendly label like "2025 fall".
- **`split(df)`** separates finished races from the upcoming one, the race whose
  result is still blank.
- **`set_race_inputs(row, start, qual, pole)`** fills in what you learn on
  qualifying day.
- **`TargetModel`** wraps one XGBoost model. It can `fit`, `predict`, explain itself
  (`contributions` for SHAP values and `importance` for gain), and `save` and `load`.
  If the training races happen to contain only one outcome, for example no DNFs at
  all, it can't learn anything, so it falls back to the plain average.
- **`train_all()`** trains all three models in one go. **`save_all()`** and
  **`load_all()`** write them to and read them from `models/<feature set>/`.
- **`forecast()`** gives you `{"top10": …, "dnf": …, "finish": …}` for one race.
- **`sweep_start_positions()`** runs the forecast for every grid spot.
- **`backtest()`** and **`score()`** are the honesty check, explained next.

## How the model is tested

Picture it being used for real, race by race:

1. Before race 9, train on races 1 to 8 only, and predict race 9.
2. Before race 10, train on races 1 to 9, and predict race 10.
3. Keep going to the most recent race.

At each step there's also a **baseline**: the plain average of the same earlier
races, such as "he's finished top 10 in 2 of his 8 races so far, so 25%."

Then three scores compare them. Lower is better for all three:
- **Brier score** is the average squared miss on a probability. Saying "50%" every
  time scores 0.25.
- **Log loss** is similar but punishes confident wrong calls much harder.
  Probabilities are kept between 1% and 99% so one bad call can't blow it up.
- **Mean absolute error** is how many positions off the finish prediction was, on average.

If the model can't beat the baseline, it isn't adding anything. Right now it
doesn't, and the README shows the numbers.

## Adding a feature of your own

1. Build the column in `enrich.py`. If it's measured during a race, make sure it's
   lagged, so it only uses earlier races.
2. Add its name to `CORE_FEATURES` or `ENRICHED_FEATURES`.
3. Give it a friendly name in `FEATURE_NAMES` in `app.py`.
4. Run `train.py` and see whether the backtest gets better. Be honest with yourself
   if it doesn't.
