# The whole-field model (`field_model.py`)

This is version 2's engine. It answers the F1-predictor question: **for every car in
the race, what are its chances?** That means a chance of winning, of a top 3, top 5
and top 10, of failing to finish, and an expected finishing position.

It works in two stages. **XGBoost** learns how cars compare, and a **simulation**
plays the race out thousands of times to turn that into chances.

## Stage 1: two XGBoost models

**The ranker (`XGBRanker`, `rank:pairwise`)** learns *who finishes ahead of whom*.
Instead of predicting each car on its own, it's trained one race at a time (each race
is a "group") on pairs of cars: did car A beat car B? The result is a score for each
car, and a higher score means further forward.

One important detail: **the ranker only learns from cars that were still running at
the finish.** A car wrecked on lap 50 is classified near the back by bad luck, not
because it was slow. Training on those finishes taught the ranker noise, and leaving
them out roughly tripled the rank correlation in the backtest (0.03 to 0.09). It's the
only change made after looking at the backtest, and it was made on principle (the
simulation handles DNFs separately), not to chase a number.

**The DNF model (`XGBClassifier`)** gives each car's chance of *not* finishing.

Both use XGBoost's **native categorical support** (`enable_categorical=True`) for the
manufacturer, so Chevrolet, Ford and Toyota don't need to be turned into dummy columns.

## Stage 2: simulating the race

For each simulated race:
1. Each car rolls the dice on its DNF chance. Cars that "fail" are placed at the back.
2. The cars still running are ordered by their ranker score **plus random noise**.

The noise follows a Gumbel distribution, and that's not an arbitrary choice. With
`rank:pairwise`, the ranker's scores mean "the chance A finishes ahead of B is
sigmoid(score A − score B)". Adding Gumbel noise to each score reproduces exactly that
chance. So the simulation is faithful to what the ranker learned.

Doing this 10,000 times and counting gives every car's chances. It takes well under a
second.

### The temperature

One knob controls how much the scores matter compared with luck: the **temperature**,
a multiplier on the noise. A temperature of 1.0 takes the ranker at face value.
Higher means more luck, lower means the scores count for more.

Rather than guess, the temperature is **fitted on out-of-sample races**: it's set to
the value that best explains the real finishing orders of races the models hadn't
seen. The fit uses the Plackett–Luce likelihood, the standard way to score a
predicted ranking. In the backtest, each race uses a temperature fitted only on the
races before it.

### When the grid isn't known

Before qualifying, nobody knows the starting grid. So `forecast_race()` draws 50
random grids and runs 200 simulations for each. That way the uncertainty about the
grid is part of the forecast.

## How it's tested

`backtest()` walks forward through time, like version 1. For each race from the 7th
on, it trains on the earlier races only, simulates the race 4,000 times and records
every car's chances next to what really happened. That gives **469 car entries in 12
races**, against 16 data points for the single-car model.

`score()` compares the model with simple baselines:

| Score | What it asks | Baselines |
|---|---|---|
| Rank correlation | Does the predicted order match the real one? (1 = perfectly, 0 = no better than random) | the starting grid; each driver's average finish |
| Top-10 hits | Of the 10 cars predicted to finish top 10, how many did? | the top 10 on the grid |
| Top-10 log loss and Brier | Are the top-10 chances accurate? | each driver's own top-10 record; everyone equal |
| Win log loss | Are the win chances accurate? | everyone equal |
| DNF Brier | Are the DNF chances accurate? | the field's overall DNF rate |

`calibration()` checks whether the chances are honest: of all the cars given about
a 25% top-10 chance, did about 25% finish top 10?

## Settings

The ranker uses trees three levels deep with 150 rounds, and the DNF model uses trees
two levels deep with 100 rounds. Both use a slow learning rate, strong L2
regularisation and 80% row and column sampling. There's more data than in version 1,
but at Talladega it's still noisy, so the settings stay cautious.

## The building blocks

- **`load_table()` / `split()`** read the table and separate finished races from the upcoming one.
- **`fit()`** trains both models. **`predict()`** gives each car's score and DNF chance.
- **`contributions()`** gives SHAP values of the ranker score for each car, and **`importance()`** gives gain importance.
- **`simulate()`** runs the race and **`summarise()`** turns the simulations into
  chances. Both now live in `simulation.py`, shared with the all-tracks app model.
- **`forecast_race()`** does the whole thing for one race, including random grids when needed.
- **`fit_temperature()`** fits the noise level, and **`plackett_luce_loglik()`** scores a
  ranking (also in `simulation.py`).
- **`backtest()`, `score()`, `calibration()`** are the honesty checks.
- **`save()` / `load()`** write and read `models/field/`.
