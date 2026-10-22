# data/: input and generated files

| File | Made by | What it is |
|---|---|---|
| `larson_talladega_raw.csv` | You (hand-built) | 56 columns, one row per race: start, finish, status, laps led, stage results, loop data, qualifying, cautions, crew chief, tire codes, source URLs. **The source of truth; edit this to add races** |
| `larson_talladega_model.csv` | You (hand-built) | The original model-ready table. `enrich.py` checks its rebuilt features against this file |
| `larson_talladega_model_enriched.csv` | `enrich.py` | **What the models train on.** 46 columns: the original features, the new lagged features and the targets. The last row is the upcoming race, with blank targets |
| `larson_talladega_race_metrics.csv` | `enrich.py` | Per-race in-race metrics from NASCAR's feeds, *not* lagged. For reference and the dashboard's history chart only |
| `cache/` | `fetch_nascar.py` | Saved copies of NASCAR's JSON feeds (schedules plus lap, pit and notes feeds for 2020–2026) |

## Coverage

- **23 completed races:** spring 2014 to spring 2026. He did not race either 2020 event.
- **Lap, pit and notes data:** 11 races, from spring 2021 on.
- **Qualifying speed:** blank for 4 races without time trials.
- **Tire codes:** partly inferred (see `tire_code_basis` in the raw file). Spring 2014 has none.

## Key columns in the model table

| Column | Notes |
|---|---|
| `race_date`, `season`, `season_half` | Race identity. `season_half` is spring or fall |
| `start_position`, `field_size`, `start_pct_field` | Grid slot. Blank for the upcoming race until qualifying |
| `qualifying_speed_mph`, `qual_gap_to_pole_pct` | Qualifying |
| `prev_*` | The value from his **previous** Talladega race |
| `hist_mean_*`, `hist_*_rate` | The average over **all earlier** Talladega races |
| `target_finish_position`, `target_top10`, `target_dnf` | What the models predict. Blank for the upcoming race |
| `nascar_race_id` | NASCAR's ID, used to look up the feeds |

## Why "lagged"?
Driver rating, laps led and lap positions are only known *after* a race. Using a
race's own values to predict that race would leak the answer. So each row only sees
earlier races: `prev_rating` on the fall 2025 row is his rating from spring 2025.

## The project root copies
`larson_talladega_raw.csv`, `larson_talladega_model.csv` and `File Notes.txt` in the
project root are your original files, kept untouched. The scripts use the copies in
`data/`, which are identical.
