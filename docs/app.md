# The app (`app.py`)

The web page everyone sees: every car's chances in the next NASCAR Cup race, a
closer look at any car, and an honest account of how good the model is.

To open it, run `.venv\Scripts\streamlit run app.py` (or double-click `run_app.bat`
on Windows). It opens at <http://localhost:8501>. The live copy is at
[xgboost-nascar-predictor.streamlit.app](https://xgboost-nascar-predictor.streamlit.app).

## How it decides what to show

1. **It finds the next race** on NASCAR's published schedule: the first points race
   after the last one in `data/races.csv`.
2. **It shows the saved forecast** from `race_train.py` if there is one for that race.
   That's quick, and every visitor sees the same numbers.
3. **Refresh from NASCAR** (top right) re-reads the entry list and, once qualifying is
   done, the real starting grid, then simulates the race 10,000 times live. Use it in
   race week, especially after qualifying.

If NASCAR hasn't posted an entry list yet, the page says so. If there's no race on the
schedule at all (between seasons), it says that too.

The line under the race name says how many cars are entered, whether the grid is set,
and whether the forecast is the saved one or a live refresh. **Until the grid is set,
each batch of simulations draws a random grid,** so the uncertainty about where
everyone starts is part of the forecast.

## The four tabs

### Race forecast

- **Three tiles:** the favourite to win, the car most likely to finish top 10, and
  "how open is it?": the favourite's win chance against an even split of the field.
- **Where each car is likely to finish:** one line per car, sorted by expected finish.
  The dot is the expected finish and the pale bar covers the middle 80% of simulated
  finishes. The car you pick in *Driver view* is shown in orange.
- **Every car's chances:** start, expected finish, and the chance of a win, top 3,
  top 5, top 10 and DNF. The progress bars are scaled per column so the differences
  are easy to see, and the numbers are the real chances.

### Driver view

Pick any car in the race.

- **Five tiles:** win, top 5, top 10, expected finish, DNF.
- **Where they finished in 10,000 simulations:** the full spread of outcomes, position
  by position. Finishes at the very back include simulated DNFs.
- **What the model sees:** SHAP values from the ranker, showing which factors push this
  car up the order (blue) or back (red), with each factor's value. Before the grid is
  set, the car is explained from a mid-field start, because "unknown" isn't something
  the model has learned from.
- **Recent form:** the last 10 races, with DNFs marked by a red **×**.
- **At this track:** every start here since 2018, with a quick summary.

### How good is it?

- **The scorecard:** the 70-race walk-forward backtest against simple baselines (the
  starting grid, recent form, the driver's recent top-10 rate, everyone equal), saying
  plainly whether the model won each one.
- **By kind of track:** rank correlation for XGBoost and the starting grid on short
  tracks, intermediates, superspeedways and road courses.
- **Are the top-10 chances honest?** is a calibration chart. Points on the dashed line
  mean "a 30% chance" really happens about 30% of the time.
- **What the ranker relies on:** feature importance (gain).
- **Look back at a past race:** pick any of the 70 races to see the forecast as it would
  have been made beforehand, next to the real result.

### Data

The full table of every car in every race since 2018 (the latest 500 rows on screen,
all of them in the download), and the exact model inputs for this race's entry list.

## Colours

From a colour-blind-safe palette, and colour is never the only clue: every series has
a legend, and DNFs get their own **×** marker.

- **Blue** (`#2a78d6`): the model's forecasts and the main bars
- **Orange** (`#eb6834`): the car picked in Driver view
- **Black diamond:** a car's real finish in a past race
- **Grey** (`#898781`): baselines and reference lines
- **Red** (`#d03b3b`): a DNF (always an **×**), and factors that push a car back

## Changing things

- **Feature names** shown in the charts live in `FEATURE_NAMES` at the top of `app.py`.
- **Chart look:** `style()` sets the shared fonts, margins, gridlines and legend.
- **Two shared charts:** `finish_range_chart()` and `chances_table()` are used for both
  the next race and past races.
