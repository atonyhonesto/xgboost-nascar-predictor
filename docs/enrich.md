# Building the features (`enrich.py`)

```powershell
.venv\Scripts\python enrich.py
```

This script takes the raw race records and NASCAR's detailed race data and turns
them into the table the models learn from. Most of the careful thinking in the
project happens here, because this is where it would be easiest to cheat by
accident.

## What it does, step by step

**1. Rebuilds the original features and double-checks them.** The starting point
was a hand-built model table. The script recreates every history feature in it
(last finish, average rating and so on) from the raw data, then compares the two.
Every column comes out `OK`, so the original work was right.

**2. Measures new things from NASCAR's feeds.** For races from 2021 on, it reads the
lap-by-lap, pit stop and race-control data (through `fetch_nascar.py`) and measures
how each race actually went.

**3. Turns them into history.** This is the key step. A race's own stats are only
known after it ends, so they can't be used to predict it. Instead each one becomes
two features: `prev_<x>`, what happened in the *previous* race, and `hist_mean_<x>`,
the average over *all earlier* races. No race ever sees its own result.

**4. Adds the upcoming race** (from `NEXT_RACE`) as the last row, with its results left blank.

## What it saves

- **`data/larson_talladega_model_enriched.csv`** is the table the models train on:
  the 23 past races plus the upcoming one.
- **`data/larson_talladega_race_metrics.csv`** holds the new per-race numbers
  *before* they're turned into history. They're useful to look at and the dashboard
  charts them, but they must never go into the model directly, because they'd give
  away the answer.

## The new measurements

From the **lap-by-lap data:**
- **Share of laps in the top 5 and top 10:** how much of the race he spent near the
  front. This turned out to be the most telling new number. In his spring 2022 4th
  place he ran 93% of the laps inside the top 10.
- **Average position late in the race,** over the final 20% of laps, which is where
  Talladega is usually decided. It's blank if he was already out by then.
- **How much his position swung:** a calm race and a chaotic one look very different.

From the **pit stop data:**
- **Typical tire-stop time,** a measure of how quick the crew is. Stops longer than
  30 seconds are left out, because those are damage repairs, not normal service.
- **Places gained or lost on pit road.** This is a weak signal, because it depends
  a lot on when everyone else pits.

From the **race-control notes:**
- **Incidents and penalties** that mention the #5. These are found by looking for
  words like *wreck*, *spin*, *damage* and *penalty*, so treat the counts as
  approximate.

From the **raw race records:**
- **Crash rate,** kept separate from mechanical failures, because a blown engine and
  a wreck are different stories.
- **Qualifying gap to pole, in %.** Raw qualifying speed can't be compared between
  the older Gen6 cars and today's Gen7, but "how far off pole" can.

**What got dropped:** lap-time pace. At Talladega the whole field runs nose to tail,
so his lap times matched the field's in every single race. There's nothing to learn
from it here, so positions tell the story instead.

## Settings at the top of the file

- **`NEXT_RACE`:** details of the race to forecast, including the date, team, crew
  chief and field size. It's currently the spring 2027 race.
- **`NEXT_RACE_DATE_CONFIRMED`:** `False` while the date is an estimate. NASCAR hadn't
  published its 2027 schedule feed yet, so April 25, 2027 is a guess based on when the
  spring race usually runs. The app shows a note while this is `False`.
- **`NEXT_RACE_ID`:** NASCAR's ID for that race, which the lineup lookup uses. It's
  `None` until the 2027 schedule is out. Then copy the ID from `race_list_basic.json`
  (or run `fetch_nascar.py`) and set the confirmed date.
- **`LATE_RACE_SHARE`:** how much of the race counts as "late" (20%).
- **`MAX_SERVICE_STOP_SEC`:** the cut-off between a normal pit stop and a repair (30 seconds).

## Still worth double-checking

- **The Oct 25, 2026 tire codes** are assumed to be the same as spring 2026 (D-5142 /
  D-5218). Goodyear's race-week notes will confirm them.
- **Two penalties each in spring 2024 and spring 2025** haven't been checked by hand.
  One in 2024 is probably his pre-race penalty for unapproved adjustments, which sent
  him to the back.

## After each new race

1. Add the race with `python add_race.py --season <year> --race-id <id>` (see [add_race.md](add_race.md)).
2. Point `NEXT_RACE` and `NEXT_RACE_ID` at the next Talladega race.
3. Run `enrich.py`, then `train.py`. For the whole-field model, also run
   `field_data.py` and `field_train.py`.
