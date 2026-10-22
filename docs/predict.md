# predict.py: command-line forecast

The terminal version of the dashboard's Forecast tab. Run `train.py` once first,
because this script loads the saved models from `models/`.

## Usage

```powershell
# Before qualifying: a table of the forecast at every start position
.venv\Scripts\python predict.py

# After qualifying: pull Larson's start and speeds from NASCAR's feed
.venv\Scripts\python predict.py --auto

# Or type them in
.venv\Scripts\python predict.py --start 12 --qual-speed 181.2 --pole-speed 182.4

# Use the enriched model set instead of core
.venv\Scripts\python predict.py --start 12 --features enriched
```

| Option | Meaning |
|---|---|
| `--start N` | Starting position |
| `--qual-speed MPH` | Larson's qualifying speed |
| `--pole-speed MPH` | The pole winner's speed. Both speeds are needed to compute the qualifying gap |
| `--auto` | Read the lineup from NASCAR's weekend feed. If it isn't posted yet, the script falls back to the full start-position table |
| `--features core\|enriched` | Which saved model set to use (default `core`) |

## Example

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

`+` means the feature pushed the top-10 chance up, `-` means down (SHAP values from XGBoost).

## Where the race details come from
`NEXT_RACE` and `NEXT_RACE_ID` in `enrich.py` define the race being forecast (date,
team, field size, NASCAR race ID 5631). Update both for the next Talladega race.
