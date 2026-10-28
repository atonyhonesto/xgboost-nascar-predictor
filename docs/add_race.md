# Adding a finished race (`add_race.py`)

After each Talladega race, the 5 Car's result needs a new row in
`data/larson_talladega_raw.csv`. This script builds that row straight from NASCAR's
feeds, so you don't have to copy 56 columns in by hand.

```powershell
.venv\Scripts\python add_race.py --season 2026 --race-id 5631
```

`--car` defaults to 5. The race ID is NASCAR's (5631 is the Oct 25, 2026 YellaWood
500). You'll find it in the season's schedule feed, and `fetch_nascar.py` prints it for
every Talladega race.

## Where the numbers come from

- **The weekend feed:** start, finish, status, laps, laps led, stage finishes and
  points, qualifying, cautions, lead changes, the winner, team, crew chief and sponsor.
- **The loopstats feed:** driver rating, running positions (mid-race, best, worst,
  average), green-flag passes, quality passes, fastest laps and laps in the top 15.
- **Worked out from those:** percentages such as quality passes and laps led, the
  qualifying lap time (from the speed and Talladega's 2.66-mile lap), the race number
  in the season, overtime, and spring or fall.

## What it can't fill in

**Goodyear tire codes aren't in NASCAR's feeds.** The script copies the previous
race's codes and marks the row's `tire_code_basis` as *"inferred … not yet confirmed"*.
Check Goodyear's race-week notes and correct them if they changed.

## After adding a race

```powershell
.venv\Scripts\python enrich.py
.venv\Scripts\python train.py
.venv\Scripts\python field_data.py
.venv\Scripts\python field_train.py
```

The first two update the 5 Car model and the last two update the whole-field model.
If the race was the one `NEXT_RACE` in `enrich.py` pointed at, move `NEXT_RACE` on to
the next Talladega race first.

The script refuses to add a race date that's already in the file, so running it twice
is harmless.
