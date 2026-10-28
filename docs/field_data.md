# The whole-field data (`field_data.py`)

```powershell
.venv\Scripts\python field_data.py
```

Version 1 looked at one car. Version 2 looks at **every car in every race**, the way
the F1 predictor projects do. This script builds that table, entirely from NASCAR's
public feeds. It saves it as `data/field_talladega.csv`.

## What's in it

One row per car per race: **708 entries across 18 Talladega races (2018 to 2026),
covering 88 different drivers.** On top of that there are 40 *projected* rows for the
next race (more on those below).

Each row has three kinds of information.

**What happened**, used as targets and to build history: start and finish position,
whether the car was still running at the end (`status`), laps completed and led,
NASCAR's driver rating and average running position, and the share of laps the car
spent in the top 10.

**History, all lagged** so a row only knows about *earlier* races:
- **The driver's Talladega record:** how many starts (`drv_starts`), last and
  average finish, top-10 rate, DNF rate, last and average driver rating, average
  running position, share of laps in the top 10, and average laps led.
- **The team's Talladega record:** average finish, average driver rating and top-10
  rate across all of the team's cars in earlier races. This helps a lot with drivers
  who are new to the track but drive for a strong team.

**Known before the race:** start position as a share of the field, qualifying gap
to pole (%), manufacturer, whether it's the fall playoff race, and whether it's the
current Gen7 car.

## Where each piece comes from

| Feed | What it gives | From |
|---|---|---|
| Results (weekend feed) | Every car's start, finish, status, team, make, qualifying | 2018 |
| Loopstats | Driver rating, average running position | 2019 |
| Lap times | Share of laps in the top 10 | 2020 |

Older races aren't in NASCAR's public feeds, which is why the table starts in 2018.
The gaps (no loop data in 2018, no lap data before 2020) stay as blanks, and XGBoost
handles them as "missing".

## The projected rows

The next Talladega race hasn't happened, and NASCAR doesn't post an entry list until
race week. So the table assumes the **same 40 cars as the most recent race**, with the
starting grid left blank. Each driver's history runs right up to the last race.

When the real entry list is out, `field_predict.py --race-id` uses that instead.

## Why so many columns start with `drv_` and `team_`

It's the same rule as version 1: a race can't be predicted from numbers measured
during that race. The `drv_` and `team_` columns are always averages of *earlier*
Talladega races, so they're safe to use.

## When to run it

After each new Talladega race. It downloads only what isn't already saved in
`data/cache/`, then rebuilds the whole table. Then run `field_train.py`.
