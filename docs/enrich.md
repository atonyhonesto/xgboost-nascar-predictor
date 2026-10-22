# enrich.py: build the model table

```powershell
.venv\Scripts\python enrich.py
```

Turns the raw race-by-race CSV plus NASCAR's feeds into the leak-free table that
`model.py` trains on.

## Steps

1. **Rebuild the original lagged features** (`prev_fin`, `hist_mean_rating`, ...) from
   `data/larson_talladega_raw.csv` and check each one against the supplied
   `data/larson_talladega_model.csv`. Every column prints `OK`, which confirms the
   original data work was correct.
2. **Measure new per-race metrics** from NASCAR's feeds (2021 on) via `fetch_nascar.py`.
3. **Lag them:** `prev_<x>` is the previous race's value and `hist_mean_<x>` is the
   average over all earlier races, so no race ever sees its own result.
4. **Append the upcoming race** (`NEXT_RACE`) as the last row, with its targets blank.

## Outputs

| File | Use |
|---|---|
| `data/larson_talladega_model_enriched.csv` | **The model table.** 23 races + the race to predict |
| `data/larson_talladega_race_metrics.csv` | The new per-race numbers *before* lagging. Reference only, because they would leak results into the model |

## New metrics

| Metric | Source | How it's measured |
|---|---|---|
| `pct_laps_top5`, `pct_laps_top10` | lap feed | Share of his laps run in the top 5 / top 10 |
| `late_race_avg_pos` | lap feed | Average position over the final 20% of laps (`LATE_RACE_SHARE`). Blank if he was out before then |
| `run_pos_std` | lap feed | How much his position swung during the race |
| `median_tire_stop_sec` | pit feed | Median tire-change stop time. Stops over 30 s (`MAX_SERVICE_STOP_SEC`) count as damage repairs and are excluded |
| `pit_road_pos_net` | pit feed | Positions gained or lost on pit road. Weak signal: it depends on when the field pits |
| `incident_notes`, `pit_penalties` | race notes | Notes naming #5 with words like *wreck, spin, damage*, or *penalty*. **Approximate**, because it is keyword matching |
| `crash_dnf` | raw CSV | DNF caused by a crash, separate from mechanical failures |
| `qual_gap_to_pole_pct` | raw CSV | `(pole − his speed) / pole × 100`. Replaces raw speed, which can't be compared across Gen6 and Gen7 cars |

**Dropped:** lap-time pace. In Talladega pack racing everyone runs the same lap time,
so his pace relative to the field was 1.000 in every race.

## Configuration (top of the file)

| Name | Meaning |
|---|---|
| `NEXT_RACE` | Details of the race to forecast: date, team, crew chief, field size, tire codes |
| `NEXT_RACE_ID` | NASCAR's race ID for it (5631 = YellaWood 500, Oct 25 2026). Used to pull the lineup |
| `LATE_RACE_SHARE` | Fraction of the race counted as "late" (0.20) |
| `MAX_SERVICE_STOP_SEC` | Cut-off between a service stop and a damage repair (30 s) |

## Open checks
- **Oct 25 tire codes** are assumed to match spring 2026 (D-5142 / D-5218). Confirm
  them against Goodyear's race-week notes.
- **Two penalties each in spring 2024 and spring 2025** haven't been checked by hand.
  One in 2024 is probably his pre-race penalty for unapproved adjustments.

## After each new race
1. Add a row for the race to `data/larson_talladega_raw.csv`.
2. Point `NEXT_RACE` / `NEXT_RACE_ID` at the next Talladega race.
3. Run `enrich.py`, then `train.py`.
