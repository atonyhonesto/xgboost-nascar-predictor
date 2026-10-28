# XGBoost NASCAR Predictor

[![XGBoost](https://img.shields.io/badge/XGBoost-3.4-EB6834?style=for-the-badge)](https://xgboost.readthedocs.io/)
[![XGBRanker](https://img.shields.io/badge/XGBRanker-rank%3Apairwise-EB6834?style=for-the-badge)](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)
[![SHAP](https://img.shields.io/badge/SHAP-pred__contribs-2A78D6?style=for-the-badge)](https://xgboost.readthedocs.io/en/stable/prediction.html#prediction-options)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-7-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/python/)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://xgboost-nascar-predictor.streamlit.app)

**▶ Try it live: [xgboost-nascar-predictor.streamlit.app](https://xgboost-nascar-predictor.streamlit.app)**

Before every NASCAR Cup race, this app forecasts **every car's chances**: win, top 3,
top 5, top 10, failing to finish (DNF), and an expected finishing position. It's in
the spirit of the F1 race-predictor projects on GitHub, built to show what **XGBoost**
can do with real, messy sports data.

It finds the next race on NASCAR's schedule by itself and reads the entry list from
NASCAR's public feeds. After qualifying, one click brings in the real starting grid.
Right now that's the **Xfinity 500 at Martinsville, Sunday Nov 1, 2026**.

Under the hood:

- **12,224 car entries** from every Cup points race since 2018: 322 races at 36 tracks, 151 drivers
- An **`XGBRanker`** that learns who finishes ahead of whom, one race at a time
- An **`XGBClassifier`** for each car's chance of failing to finish
- A **simulation** that plays the race out 10,000 times to turn that into chances
- An honest **walk-forward backtest** over 70 races against simple baselines

## How good is it?

Each of the 70 races since January 2025 was forecast using only the races before it,
with the real starting grid, then compared with simple baselines:

| Score | XGBoost | Baseline |
|---|---|---|
| Rank correlation with the real finishing order (higher is better) | **0.422** | 0.369 (starting grid) · 0.309 (recent form) |
| Of the model's top 10, how many finished top 10 | **4.96** | 4.54 (top 10 on the grid) · 4.29 (recent form) |
| Top-10 log loss (lower is better) | **0.512** | 0.552 (driver's recent top-10 rate) |
| Win log loss (lower is better) | **0.112** | 0.122 (everyone equal) |
| DNF Brier score (lower is better) | **0.112** | 0.114 (field DNF rate) |

**It beats every baseline,** and it's best where skill shows most:

| Kind of track | Rank correlation: XGBoost | Starting grid |
|---|---|---|
| Short tracks (Martinsville, Bristol, Richmond…) | **0.551** | 0.432 |
| Road courses | **0.464** | 0.434 |
| Intermediate ovals | **0.458** | 0.406 |
| Superspeedways (Daytona, Talladega, Atlanta) | 0.093 | **0.121** |

Superspeedways are the honest exception. Pack racing there is close to a lottery, and
the model can't beat the starting grid. That's where this project started (see
[The Talladega study](#the-talladega-study)).

One thing to improve: the top-10 chances are a little **under-confident** at the
extremes. Cars given around 50% finished top 10 about two times in three. The
calibration chart in the app shows it.

## What you can learn from it

The race is the fun part. The real point is the XGBoost techniques, which carry over
to any ranking problem on tabular data.

**Learning to rank.** `XGBRanker` with `rank:pairwise` is trained one race at a time
on *pairs* of cars: did A beat B? That's the natural question for any race, league or
leaderboard. → [race_model.py](race_model.py)

**From scores to chances, with a simulation.** XGBoost gives each car a score, and
NumPy plays the race out 10,000 times. With `rank:pairwise`, adding Gumbel noise to
the scores reproduces exactly the head-to-head chances the ranker learned, so the
simulation stays faithful to the model. The amount of noise is fitted on races the
model hadn't seen. → [simulation.py](simulation.py)

**One model for every kind of track.** The kind of track (short track, intermediate,
superspeedway, road course) is a categorical feature, using XGBoost's native
`enable_categorical`. So one model learns that form carries over well at Martinsville
and hardly at all at Talladega.

**Don't let the model see the answer.** Driver rating, laps led and running positions
are only known *after* a race. Every one of them comes in only as history: the last
5 races, this track's record, the last 10 races at similar tracks, the team's last
5 races. → [race_data.py](race_data.py)

**Train on the signal, not the luck.** The ranker only learns from cars still running
at the finish. A car wrecked on lap 50 finishes near the back by bad luck, not because
it was slow, and the simulation handles DNFs separately.

**Test the way you'll use it.** Shuffled cross-validation would let the model train on
2026 to predict 2025, which is time travel. The backtest walks forward: train on
every earlier race, forecast the next one, repeat.

**Always have a baseline, and check calibration.** Every score is shown next to simple
baselines, and a calibration chart checks that "25%" really means 25%.

**Explain every prediction.** `pred_contribs=True` gives exact SHAP values from XGBoost
itself. The app's driver view shows what lifts or sinks each car.

## Running it yourself

You'll need Python 3.12.

```powershell
git clone https://github.com/atonyhonesto/xgboost-nascar-predictor.git
cd xgboost-nascar-predictor
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\streamlit run app.py
```

On macOS or Linux, use `python3 -m venv .venv`, then `.venv/bin/pip` and
`.venv/bin/streamlit`. On Windows you can also double-click `run_app.bat`. The app
opens at <http://localhost:8501>.

The repo includes the built table (`data/races.csv`), the trained models and the
reports, so the app works straight away.

## After each race

```powershell
.venv\Scripts\python race_data.py    # add the new race (downloads only what's missing)
.venv\Scripts\python race_train.py   # backtest, retrain, and forecast the next race
```

`race_data.py` needs NASCAR's raw feeds, which it saves in `data/cache/`. That folder
isn't in the repo because it holds about 650 files, so the first run on a new machine
downloads them, which takes a few minutes.

During race week, the app's **Refresh from NASCAR** button re-reads the entry list and,
after qualifying, the starting grid, and re-simulates the race live.

## How the pieces fit together

```
NASCAR's public feeds ── fetch_nascar.py
                              │
                         race_data.py ──> data/races.csv (every car, every race, with history)
                              │
                         race_model.py (XGBRanker + DNF model) ── simulation.py (10,000 races)
                              │
                         race_train.py ──> models/race/, reports/race_*.csv
                              │
                           app.py (Race forecast · Driver view · How good is it? · Data)
```

| File | In one line | Guide |
|---|---|---|
| `app.py` | The interactive app | [docs/app.md](docs/app.md) |
| `race_data.py` | Every car in every race since 2018, with lagged history | [docs/race_data.md](docs/race_data.md) |
| `race_model.py` | The ranker, the DNF model, forecasting and the backtest | [docs/race_model.md](docs/race_model.md) |
| `simulation.py` | Turns XGBoost scores into every car's chances | [docs/simulation.md](docs/simulation.md) |
| `race_train.py` | Backtests, trains and forecasts the next race | [docs/race_train.md](docs/race_train.md) |
| `fetch_nascar.py` | Talks to NASCAR's public data feeds | [docs/fetch_nascar.md](docs/fetch_nascar.md) |
| `data/` | The race data | [docs/data.md](docs/data.md) |

**Built with:** `xgboost` for the models and their explanations, `numpy` for the
simulation, `pandas` for the data, `scikit-learn` (which XGBoost's model classes
need), `requests` for NASCAR's feeds, `streamlit` for the app and `plotly` for the
interactive charts.

## Hosting it

The live app runs free on [Streamlit Community Cloud](https://share.streamlit.io). To
host your own, fork this repo, create an app there pointing at `app.py` on the `main`
branch, and choose Python 3.12 under Advanced settings. No passwords or API keys are
needed.

Free apps fall asleep after about 12 hours without visitors. A small GitHub Action,
[`.github/workflows/keep-awake.yml`](.github/workflows/keep-awake.yml), visits every 6
hours and wakes the app if it has dozed off.

## The Talladega study

The project started much smaller: **one car (the #5 of Kyle Larson) at one track
(Talladega)**, then every car at Talladega. Those versions are still in the repo, and
they're a good lesson in why the data, not the algorithm, sets the ceiling:

- **Version 1: one car at Talladega,** 24 races. Its three XGBoost models mostly
  couldn't beat the driver's own averages. Files: `enrich.py`, `model.py`, `train.py`,
  `predict.py`, `add_race.py`. Guides: [model](docs/model.md), [enrich](docs/enrich.md),
  [train](docs/train.md), [predict](docs/predict.md), [add_race](docs/add_race.md).
- **Version 2: every car at Talladega,** 708 entries. It beat the baselines by a hair
  on finishing order, but not on winners or crashes. Before Oct 25, 2026 it ranked the
  eventual winner, the 5 Car, 33rd of 40. Files: `field_data.py`, `field_model.py`,
  `field_train.py`, `field_predict.py`. Guides: [field_data](docs/field_data.md),
  [field_model](docs/field_model.md), [field_train](docs/field_train.md),
  [field_predict](docs/field_predict.md).

Going to every track (this version) is what moved the model from "barely better than
a coin flip" to clearly better than the baselines.

## What's next

- Fix the under-confidence in the top-10 chances (a calibration step after the simulation)
- Put the forecasts next to prediction markets: Kalshi lists win, top-3, top-5 and top-10 markets
- Model *when* a car's race ends with XGBoost's survival objective (`survival:aft`)
- Practice speeds, once they can be pulled from NASCAR's feeds reliably
