# model.py: shared modelling code

Everything the three entry points (`train.py`, `predict.py`, `app.py`) have in common
lives here, so they can never disagree about features or settings.

## Feature sets

All features are known **before the green flag**. In-race statistics appear only as
history from earlier races.

**`core` (15 features, available back to 2014)**

| Feature | Meaning |
|---|---|
| `start_pct_field` | Start position ÷ field size (0.05 = near the front) |
| `qual_gap_to_pole_pct` | How far his qualifying speed was below pole, in %. Blank when there were no time trials |
| `is_playoff`, `is_gen7`, `is_hendrick` | Fall playoff race; Gen7 car (2022+); driving for Hendrick |
| `prior_talladega_starts` | Experience count |
| `prev_fin`, `hist_mean_fin` | Last Talladega finish; average of all earlier ones |
| `prev_rating`, `hist_mean_rating` | NASCAR driver rating, last and average |
| `prev_avg_run_pos`, `hist_mean_avg_run_pos` | Average running position, last and average |
| `hist_mean_laps_led` | Average laps led |
| `hist_dnf_rate`, `hist_crash_rate` | Share of earlier races not finished / ended in a crash |

**`enriched` (core + 8 features, from NASCAR's lap, pit and notes feeds, 2021 on)**

`prev_pct_laps_top10`, `hist_mean_pct_laps_top10`, `prev_pct_laps_top5`,
`hist_mean_late_race_avg_pos`, `hist_mean_run_pos_std`,
`hist_mean_median_tire_stop_sec`, `hist_mean_pit_road_pos_net`,
`hist_mean_incident_notes`. See [enrich.md](enrich.md) for how each is measured.
XGBoost treats the blanks before 2021 as "missing" and learns a default direction for them.

## Targets

| Key | Column | Model |
|---|---|---|
| `top10` | `target_top10` | `XGBClassifier`, probability of finishing 1st–10th |
| `dnf` | `target_dnf` | `XGBClassifier`, probability of not running at the finish |
| `finish` | `target_finish_position` | `XGBRegressor`, expected position (clipped to 1–field size) |

## XGBoost settings (`XGB_PARAMS`)

```python
n_estimators=80, learning_rate=0.05, max_depth=2, min_child_weight=1.0,
subsample=1.0, colsample_bytree=0.8, reg_lambda=5.0, random_state=42
```

These are deliberately conservative: shallow trees, a slow learning rate and strong L2
regularisation. With 23 rows, anything more flexible memorises the noise. They were
**not** tuned against the backtest, because tuning on 15 test races would overfit the
test itself.

## Main pieces

| Name | What it does |
|---|---|
| `load_table()` | Reads `data/larson_talladega_model_enriched.csv` and adds the `is_*` flags and a `race_label` like "2025 fall" |
| `split(df)` | Returns *(completed races, upcoming race)*, split on whether the finish is known |
| `set_race_inputs(row, start, qual, pole)` | Fills in the qualifying-day inputs for the upcoming race |
| `TargetModel` | Wraps one XGBoost model: `fit`, `predict`, `contributions` (SHAP), `importance` (gain), `save`/`load`. If the training rows contain only one outcome it falls back to the base rate |
| `train_all(past, feature_set)` | Fits all three targets |
| `save_all` / `load_all` | Write / read `models/<feature_set>/` |
| `forecast(models, row)` | `{"top10": p, "dnf": p, "finish": x}` for one race |
| `sweep_start_positions(models, row)` | The forecast at every start position |
| `backtest(past, feature_set)` | Walk-forward evaluation (below) |
| `score(bt)` | Brier, log loss and MAE for model vs baseline |

## The backtest

For each race *i* from the 9th onwards (`MIN_TRAIN_RACES = 8`):

1. Train on races 1 … *i*−1 only.
2. Predict race *i*.
3. Record the **baseline** as the plain average of the target over the same training
   races: his top-10 rate so far, his DNF rate so far, his average finish so far.

Scores (lower is better):
- **Brier score**: mean squared error of a probability. Always predicting 50% scores 0.25.
- **Log loss**: punishes confident wrong calls hard. Probabilities are clipped to 1–99%.
- **MAE**: mean absolute error in finishing positions.

A model is only worth using if it beats the baseline. It doesn't yet: see the
README's results table.

## Adding a feature
1. Build the column in `enrich.py`, lagged if it is measured during a race.
2. Add its name to `CORE_FEATURES` or `ENRICHED_FEATURES`.
3. Add a friendly name to `FEATURE_NAMES` in `app.py`.
4. Run `train.py` and check whether the backtest improved.
