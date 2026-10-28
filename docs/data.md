# The data (`data/`)

Everything the models learn from lives in this folder. Here's what each file is
and how they connect.

## The app's data

**`races.csv`** is what the app learns from: **every car in every NASCAR Cup points
race since 2018**, 12,224 rows from 322 races at 36 tracks. It's made by
`race_data.py` (see [race_data.md](race_data.md)). The raw columns say what happened
(start, finish, status, rating). The `form_`, `trk_`, `type_`, `team_`, `season_` and
`career_` columns are history from *earlier* races only.

**`cache/`** holds NASCAR's raw feeds: about 650 files, 27 MB. It isn't in the GitHub
repo (see `.gitignore`), because the app doesn't need it. The build scripts download
anything that's missing.

## The Talladega study's data

The rest of this folder belongs to the earlier single-track versions (see the README's
*Talladega study*).

**`larson_talladega_raw.csv`** is the source of truth. It's hand-built, with one row
for each of Larson's Talladega races and 56 columns: start and finish, whether he
was still running at the end, laps led, stage results, NASCAR's loop data such as
driver rating, qualifying, cautions, the crew chief, tire codes, and a link to the
source for every row. **New races are added here,** most easily with `add_race.py`,
which builds the row from NASCAR's feeds (see [add_race.md](add_race.md)).

**`larson_talladega_model.csv`** is the original model-ready table, also hand-built.
`enrich.py` checks its own work against it.

**`larson_talladega_model_enriched.csv`** is what the models actually train on. It's
made by `enrich.py` and has 46 columns: the original features, the new history
features and the three things being predicted. The last row is the upcoming race, so
its results are blank.

**`larson_talladega_race_metrics.csv`** holds the per-race numbers from NASCAR's
feeds, such as share of laps in the top 10 and pit stop times, *before* they're
turned into history. The dashboard charts them, but they never go into the model
directly.

**`field_talladega.csv`** is the version 2 table: **every car in every Talladega race
since 2018**, one row per car per race, with each driver's and team's history. It's
made by `field_data.py` (see [field_data.md](field_data.md)). The rows with no result
at the end are the projected entry list for the next race.

## How much data there is

**For the 5 Car (version 1):**
- **24 completed races,** from spring 2014 to fall 2026. Larson didn't race either
  Talladega event in 2020. The latest, Oct 25, 2026, was his first Talladega win.
- **Lap-by-lap, pit and race-note data for 12 of them,** from spring 2021 on. NASCAR
  doesn't publish it for earlier years.
- **Qualifying speeds are missing for 4 races** that had no time trials.
- **Tire codes are partly educated guesses.** Goodyear rarely changes them, so some
  were carried over from a neighbouring race. The `tire_code_basis` column says which
  are confirmed. Spring 2014 has none, and fall 2026 is carried over until confirmed.

**For the whole field (version 2):**
- **708 car entries in 18 races,** spring 2018 to fall 2026, from 88 different drivers.
- **Loop data from 2019, lap data from 2020.** The earlier gaps stay blank.

## Reading the 5 Car's model table

Most column names follow a simple pattern:

- **`prev_something`** is the value from his *previous* Talladega race.
- **`hist_mean_something`** or **`hist_something_rate`** is the average over *all
  his earlier* Talladega races.
- **`target_finish_position`, `target_top10`, `target_dnf`** are what the models
  try to predict. They're blank for the upcoming race.

The rest are mostly self-explanatory: `race_date`, `season`, `season_half` (spring or
fall), `start_position` and `field_size`, `qualifying_speed_mph` and
`qual_gap_to_pole_pct`, and `nascar_race_id`, which links each row to NASCAR's feeds.

## Why so much "previous" and "average so far"?

Because of one rule: **a race can't be predicted using numbers from that same race.**
Driver rating, laps led and running position are only known once the checkered flag
falls. If the model could see them, it would look brilliant and be useless.

So each row only knows about the past. On the fall 2025 row, for example,
`prev_rating` is his rating from the spring 2025 race, never from fall 2025 itself.

## The copies in the top folder

`larson_talladega_raw.csv`, `larson_talladega_model.csv` and `File Notes.txt` in the
project's top folder are the original files, kept untouched. The scripts use the
identical copies here in `data/`.
