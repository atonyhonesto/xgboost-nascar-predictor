# XGBoost Talladega Predictor: 5 Car

**An end-to-end XGBoost project on real, messy, very small data.** It forecasts how
the #5 Hendrick Motorsports car (Kyle Larson) will run in the next NASCAR Cup race
at Talladega, then explains every forecast and tests itself honestly against a
simple baseline. It comes with an interactive Streamlit + Plotly dashboard and a
command-line version.

Three XGBoost models share one pipeline:

| Model | XGBoost class | Predicts |
|---|---|---|
| Top 10 | `XGBClassifier` (`binary:logistic`) | Probability of finishing 1st–10th |
| DNF | `XGBClassifier` (`binary:logistic`) | Probability of not running at the finish |
| Finish | `XGBRegressor` (`reg:squarederror`) | Expected finishing position |

The idea comes from the F1 race-predictor repos on GitHub, narrowed to one car at one track.

## What this project demonstrates

| Technique | Where | Why it matters here |
|---|---|---|
| **Leak-free feature engineering** | [enrich.py](enrich.py) | Stats measured during a race (driver rating, laps led, lap-by-lap positions) enter only as *history*: the previous race's value or the average over earlier races. A model that sees a race's own stats would look brilliant and be useless |
| **Native missing-value handling** | [model.py](model.py) | Lap, pit and race-note features exist only from 2021 on, and some races had no qualifying. XGBoost learns a default branch for missing values, so there's no imputation and no rows are dropped |
| **Regularisation for tiny data** | `XGB_PARAMS` in [model.py](model.py) | 23 races. Depth-2 trees, a 0.05 learning rate, L2 `reg_lambda=5` and column subsampling stop the trees from memorising noise |
| **Walk-forward backtesting** | `backtest()` in [model.py](model.py) | Train on races 1…*i*−1, predict race *i*, repeat. No shuffled cross-validation, because that would let the model learn from the future |
| **Baseline comparison** | `score()` in [model.py](model.py) | Every score (Brier, log loss, MAE) is reported next to "the average so far". A model only earns its place by beating that |
| **SHAP explanations, no extra library** | `TargetModel.contributions()` | `booster.predict(..., pred_contribs=True)` returns exact TreeSHAP values. The dashboard shows which features pushed each forecast up or down |
| **Feature importance by gain** | `TargetModel.importance()` | `get_score(importance_type="total_gain")`, printed by `train.py` |
| **Feature-set comparison** | `FEATURE_SETS` in [model.py](model.py) | `core` (15 features, 2014 on) vs `enriched` (23 features), backtested side by side |
| **Model persistence** | `save()` / `load()` | XGBoost's native JSON format plus a small metadata file per model |
| **Live inference inputs** | [fetch_nascar.py](fetch_nascar.py) | After qualifying, the start position and speeds are pulled from NASCAR's public feed and fed straight into the models |

> **On the results:** with 23 races of pack racing, where a crash can end any race,
> the models do **not** yet beat the simple baseline. The project reports that
> openly, in the app and in `train.py`. See [Current results](#current-results-backtest-on-the-last-15-races).
> Showing *how* to find out is the point. The [Roadmap](#roadmap) covers how more
> data could change the answer.

---

## Quick start

```powershell
git clone https://github.com/atonyhonesto/xgboost-talladega-5-car.git
cd xgboost-talladega-5-car
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

(On macOS or Linux: `python3 -m venv .venv`, then `.venv/bin/pip` and `.venv/bin/python`.)

| Want to... | Run |
|---|---|
| Open the dashboard | double-click **`run_app.bat`**, or `.venv\Scripts\streamlit run app.py` |
| Retrain and re-run the backtest | `.venv\Scripts\python train.py` |
| Forecast from the command line | `.venv\Scripts\python predict.py` |
| Rebuild the data table | `.venv\Scripts\python enrich.py` |

The dashboard opens at <http://localhost:8501>. Close it with `Ctrl+C` in the terminal.

## Hosting on Streamlit Community Cloud

1. Fork or push this repo to GitHub. `.venv/` and `__pycache__/` stay out
   (see `.gitignore`), and `data/` including `data/cache/` goes in.
2. On [share.streamlit.io](https://share.streamlit.io), choose **Create app**, pick the
   repo and branch `main`, and set the main file to `app.py`.
3. Under **Advanced settings**, choose **Python 3.12**. Dependencies install from
   `requirements.txt`.

The hosted app trains its models on startup and reads NASCAR's public feeds, so it
needs no secrets. On the free tier it sleeps after a few days without visitors and
wakes on the next visit.

## Race-weekend workflow (Oct 24–25, 2026)

1. **Before qualifying:** open the dashboard. With start position unknown, the
   *Chances by start position* chart shows the forecast across the whole grid.
2. **After qualifying (Sat Oct 24):** click **Pull lineup from NASCAR** in the sidebar.
   It fills in the 5 Car's start position and, if there were time trials, its qualifying
   speed and the pole speed. You can also type them in by hand.
   Command-line equivalent: `python predict.py --auto`.
3. **After the race:** add the result to `data/larson_talladega_raw.csv`, set
   `NEXT_RACE` in `enrich.py` to the next Talladega race, and run `enrich.py`
   then `train.py`.

## How it works

```
data/larson_talladega_raw.csv ──┐
NASCAR feeds (fetch_nascar.py) ─┴─> enrich.py ──> data/larson_talladega_model_enriched.csv
                                                        │
                                       model.py (features, XGBoost, backtest)
                                         │              │               │
                                     train.py       predict.py        app.py
                                  (saves models/,  (command-line     (Streamlit
                                   reports/)        forecast)         dashboard)
```

The data has **one row per race**: 23 completed races from spring 2014 to spring 2026,
plus a row for the race being predicted.

## Project files

| File | What it does | Docs |
|---|---|---|
| `app.py` | Streamlit dashboard: forecast, race history, backtest, data | [docs/app.md](docs/app.md) |
| `model.py` | Shared features, XGBoost models, backtest and forecasting | [docs/model.md](docs/model.md) |
| `train.py` | Backtests both feature sets, saves models and reports | [docs/train.md](docs/train.md) |
| `predict.py` | Command-line forecast, with auto-fill from NASCAR's lineup | [docs/predict.md](docs/predict.md) |
| `enrich.py` | Builds the model table from the raw CSV and NASCAR feeds | [docs/enrich.md](docs/enrich.md) |
| `fetch_nascar.py` | Downloads and caches NASCAR's public data feeds | [docs/fetch_nascar.md](docs/fetch_nascar.md) |
| `data/` | Input CSVs, the built model table and the feed cache | [docs/data.md](docs/data.md) |
| `models/`, `reports/` | Outputs written by `train.py` | [docs/train.md](docs/train.md) |
| `requirements.txt` | Python libraries (see below) | |
| `run_app.bat` | Double-click launcher for the dashboard | |

The CSVs and `File Notes.txt` in the project root are the original hand-built inputs
and notes. The scripts read the copies in `data/`.

## Libraries

| Library | Used for |
|---|---|
| `xgboost` | The three prediction models, plus SHAP contributions that explain each forecast |
| `pandas`, `numpy` | Data tables and feature building |
| `scikit-learn` | Required by XGBoost's `XGBClassifier` / `XGBRegressor` interface |
| `requests` | Downloading NASCAR's public JSON feeds |
| `streamlit` | The web dashboard: sidebar inputs, tabs, metric tiles, tables |
| `plotly` | Interactive charts with hover tooltips |

## Current results (backtest on the last 15 races)

Lower is better. The baseline is "the average so far" over the same training races.

| Target | Score | Model (core) | Baseline |
|---|---|---|---|
| Top 10 | Brier score | 0.203 | **0.171** |
| DNF | Brier score | 0.332 | **0.288** |
| Finish | Mean abs. error | 12.97 | **12.90** |

The enriched finish model edges the baseline (12.72), but the gap is within noise. The full
table is in `reports/metrics.csv` and on the dashboard's **Backtest** tab.

## Roadmap

**Version 1: one driver, one track**
- [x] Step 1: data pipeline, XGBoost models, backtest, CLI and dashboard *(this release)*
- [ ] Confirm the Oct 25 tire codes and the 2024/2025 penalty counts (see [docs/enrich.md](docs/enrich.md))
- [ ] Add Larson's Daytona and Atlanta superspeedway races to roughly triple the rows
- [ ] Compare against a prediction-market benchmark (Kalshi lists top-10, top-5, top-3 and win markets for this race)

**Version 2: every current driver at the upcoming track**
- One row per driver per race (about 900+ rows for Talladega alone), as the F1 repos do
- Rank the whole field; Larson becomes one driver among many
- The same `fetch_nascar.py` feeds already include every car, so most of the plumbing is in place
