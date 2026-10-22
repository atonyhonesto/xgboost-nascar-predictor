# Larson Talladega Predictor

An XGBoost app that forecasts how Kyle Larson (#5, Hendrick Motorsports) will run in
the next Cup race at Talladega: his chance of a **top-10 finish**, his chance of
**not finishing (DNF)**, and his **expected finishing position**. It has a visual
dashboard (Streamlit + Plotly) and a command-line version.

The project is modelled on the F1 race-predictor repos on GitHub, but narrowed to one
driver at one track.

> **Status: Version 1, Step 1.** The full pipeline works end to end: data, features,
> training, backtest, forecast and dashboard. With only 23 past races, the models
> **do not yet beat** Larson's plain historical averages. The app says so on screen.
> See [Roadmap](#roadmap) for how to improve that.

---

## Quick start

Everything is already installed in `.venv` (Python 3.12).

| Want to... | Run |
|---|---|
| Open the dashboard | double-click **`run_app.bat`**, or `.venv\Scripts\streamlit run app.py` |
| Retrain and re-run the backtest | `.venv\Scripts\python train.py` |
| Forecast from the command line | `.venv\Scripts\python predict.py` |
| Rebuild the data table | `.venv\Scripts\python enrich.py` |

The dashboard opens at <http://localhost:8501>. Close it with `Ctrl+C` in the terminal.

To set the project up on another machine:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Hosting on Streamlit Community Cloud

1. Put the project in a public GitHub repo. `.venv/` and `__pycache__/` stay out
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
   It fills in Larson's start position and, if there were time trials, his qualifying
   and pole speeds. You can also type them in by hand.
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

- **One row per race**, 23 completed races from spring 2014 to spring 2026, plus a
  row for the race being predicted.
- **No leakage:** anything measured during a race (driver rating, laps led, lap
  positions, pit stops) enters the model only as *history*: the previous race's value
  or the average over earlier races.
- **Three XGBoost models:** top-10 (classifier), DNF (classifier), finish position
  (regressor). The trees are kept shallow and heavily regularised because the data
  is tiny.
- **Two feature sets:** `core` (15 features, back to 2014) and `enriched` (23 features,
  adding lap, pit and race-note history that exists only from 2021 on).
- **Honest evaluation:** a walk-forward backtest trains on the races before each race
  and scores the prediction against "his average so far".

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

The CSVs and `File Notes.txt` in the project root are your original copies. The
scripts read the copies in `data/`.

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

Lower is better. The baseline is "his average so far".

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
- [ ] Compare against a betting-odds or field-average benchmark

**Version 2: every current driver at the upcoming track**
- One row per driver per race (about 900+ rows for Talladega alone), as the F1 repos do
- Rank the whole field; Larson becomes one driver among many
- The same `fetch_nascar.py` feeds already include every car, so most of the plumbing is in place
