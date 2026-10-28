# A forecast in your terminal (`predict.py`)

If you'd rather not open the dashboard, this gives you the same forecast as text.
It uses the models saved by `train.py`, so run that once first.

## Using it

**Before qualifying,** when nobody knows the start position yet:
```powershell
.venv\Scripts\python predict.py
```
You get a table of the forecast for every spot on the grid.

**After qualifying,** let it look up the lineup for you:
```powershell
.venv\Scripts\python predict.py --auto
```
If NASCAR hasn't posted the lineup yet, it says so and shows the full table instead.

**Or type the numbers in yourself:**
```powershell
.venv\Scripts\python predict.py --start 12 --qual-speed 181.2 --pole-speed 182.4
```
`--start` is the starting position. The two speeds (the car's and the pole
winner's, in mph) are used together to work out how far off pole it was. If you
give only one, it's ignored.

**To try the enriched model** instead of the default `core` one, add
`--features enriched`.

## What you'll see

```
Kyle Larson, Talladega, 2026-10-25  [core features]

  Top-10 chance:     53%   (his base rate: 22%)
  DNF chance:        41%   (his base rate: 30%)
  Expected finish:   12.0  (his average: 22.0)
  Biggest influences on the top-10 chance:
    + hist_mean_rating = 76.6
    + prev_rating = 52
    + qual_gap_to_pole_pct = 0.658
    ...

Backtest: these models do not yet beat his plain averages. See reports/metrics.csv.
```

The numbers in brackets are his plain track history, so you can see how far the
model has moved away from it. In the list of influences, **+** means that factor
nudged the top-10 chance up and **−** means it pulled it down. "not set" means that
input was missing.

That last line is a reminder that the model hasn't yet proven it beats the simple
average.

## Which race it forecasts

The race details come from `NEXT_RACE` and `NEXT_RACE_ID` in `enrich.py`: the date,
team, field size and NASCAR's race ID. It's currently spring 2027, which has no race ID
until NASCAR publishes the 2027 schedule, so `--auto` will say the lineup isn't
available yet. Update both
when you move on to the next Talladega race.
