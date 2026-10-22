# app.py: the dashboard

On screen the app refers to the car, not the driver: **"5 Car"**, built from
`NEXT_RACE["car_number"]` in `enrich.py` (the `CAR` constant at the top of `app.py`).
The history is still the driver's: his pre-2021 races were in the #42 Chip Ganassi
car, and the Race history tab says so.

A Streamlit web app for exploring the forecast. Start it with `run_app.bat` or
`.venv\Scripts\streamlit run app.py`. It opens at <http://localhost:8501>.

The app trains its models in memory each time it starts, using the same code as
`train.py` (see [model.md](model.md)). Training takes about a second, so it doesn't
need `train.py` to have run first. Results are cached while the app stays open.

## Sidebar: race inputs

| Control | What it does |
|---|---|
| **Pull lineup from NASCAR** | Reads Larson's start position, qualifying speed and pole speed from NASCAR's weekend feed. Before qualifying it reports *"Lineup not posted yet"*. |
| **Start position** | The grid slot to forecast for (1 to field size). |
| **Include qualifying speeds** | When ticked, enter Larson's qualifying speed and the pole speed. The app turns them into *qualifying gap to pole (%)*. Leave it unticked if there were no time trials. |
| **Feature set** | `core` (15 features, 2014 on) or `enriched` (23 features, adding 2021-on lap/pit/notes history). |

## Banner

A blue note under the title says how many of the five backtest scores the model wins.
It stays visible because with 23 races the honest answer is currently **0 of 5**.

## Tabs

### Forecast
- **Three tiles:** top-10 chance, DNF chance and expected finish. The small badge
  compares each to the plain track history ("vs track history"). Green means better for Larson,
  red means worse. A higher DNF chance or a higher (worse) finish number shows red.
- **Chances by start position:** the top-10 and DNF probabilities across every grid
  slot, holding everything else fixed. The dotted line marks the slot chosen in the
  sidebar. Flat stretches mean the model found no reliable link to start position.
- **What moves the top-10 chance:** SHAP contributions from XGBoost. Each bar shows
  how much one feature pushed *this* forecast up (blue) or down (red), in log-odds.
  The label shows the feature's value for this race; "not set" means missing.

### Race history
- **Start and finish, every Talladega race:** a grey dot for the start, a blue dot for
  the finish, and a red **×** when he didn't finish. Position 1 is at the top. The
  shaded band is the top 10.
- **Share of laps inside the top 10 (2021 on):** from NASCAR's lap feed. Hover a bar
  to see his finish in that race.
- A table of start, finish, status, laps led and driver rating.

### Backtest
- **Model vs baseline table:** Brier score and log loss for the two probability
  models, and mean absolute error for finish position. Lower is better.
- **Chart:** pick a target. The blue line is the model's prediction made *before*
  each race, and the dashed grey line is the baseline. For top 10 and DNF, stars mark
  the races where it happened. A useful model puts the stars above the baseline line.
- An expander holds the raw race-by-race numbers.

### Data
The full model table, with a **Download CSV** button. The last row is the race being
predicted.

## Chart colours

Colours come from a validated, colour-blind-safe reference palette:

| Use | Colour |
|---|---|
| Top-10 / model / finish (running) | blue `#2a78d6` |
| DNF chance / actual outcomes | orange `#eb6834` |
| Start position / baseline / reference lines | muted grey `#898781` |
| DNF finish marker | critical red `#d03b3b`, always drawn as an **×** with a legend label |
| Contributions | blue (raises) / red `#e34948` (lowers) |

Colour never carries meaning alone: every series has a legend or direct label, and
DNFs use a distinct marker shape.

## Editing tips
- Friendly feature names live in `FEATURE_NAMES` at the top of the file. Add an entry
  when you add a feature in `model.py`.
- `style()` applies the shared chart layout (fonts, margins, gridlines, legend).
- The race being forecast comes from `NEXT_RACE` / `NEXT_RACE_ID` in `enrich.py`.
