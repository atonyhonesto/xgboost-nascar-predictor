# The data (`race_data.py`)

```powershell
.venv\Scripts\python race_data.py
```

This builds the table the app learns from: **every car in every NASCAR Cup points
race since 2018.** That's 12,224 car entries from 322 races at 36 tracks and 151
drivers. It's saved as `data/races.csv`.

## Where it comes from

All of it comes from NASCAR's public feeds, through `fetch_nascar.py`:

- **The schedule** gives every race and its track: name, length, laps, and whether
  it's a playoff race.
- **The results** (weekend feed) give every car's start, finish, status, laps, laps
  led, team, make and qualifying speed. They go back to 2018.
- **The loop data** gives every driver's rating, average running position and laps
  in the top 15. It goes back to 2019.

The first run downloads about 650 files into `data/cache/` and takes a few minutes.
After that, only new races are downloaded.

## Kinds of track

Every track gets one of four types, because form carries over very differently
between them:

- **Short track:** ovals under 1.1 miles, such as Martinsville, Bristol, Richmond,
  Phoenix, Dover and New Hampshire. Bristol's dirt races count too.
- **Intermediate:** longer ovals that don't race in a draft, such as Charlotte,
  Kansas, Las Vegas, Darlington, Michigan and Pocono.
- **Superspeedway:** Daytona, Talladega, and Atlanta since its 2022 rebuild turned it
  into a drafting track.
- **Road course:** anything with "Road Course" or "Street" in the name, plus COTA,
  Sonoma, Watkins Glen, Road America and Mexico City.

## What each row knows

Every in-race number comes in only as **history**, so a row never sees its own race:

- **Recent form (`form_`):** the driver's last 5 races at any track, covering average
  finish, driver rating, running position and top-10 rate, plus the DNF rate over the
  last 10.
- **This season (`season_rating`):** the driver's average rating so far this year.
- **This track (`trk_`):** starts here, plus average finish, rating and top-10 rate here.
- **Similar tracks (`type_`):** the last 10 races at the same kind of track, covering
  average finish, rating and share of laps in the top 15.
- **Team (`team_`):** the team's average finish and rating over its last 5 races,
  across all its cars.
- **Experience (`career_starts`):** Cup starts since 2018.

Then there's what's known before the green flag: starting spot (as a share of the
field), qualifying gap to pole (%), the kind of track, its length, the manufacturer
and whether it's a playoff race.

## The upcoming race

The upcoming race isn't stored in the table. Instead:

- **`next_race(after_date)`** finds the first points race on NASCAR's schedule after
  the last one in the table.
- **`entry_rows(race)`** reads its entry list from the weekend feed, including the
  starting grid once qualifying is done.
- **`upcoming_rows(history, race, entries)`** attaches everyone's history to the entry list.

The app and `race_train.py` both use these, so the forecast always follows the real
schedule.

## A data quirk worth knowing

NASCAR's schedule feed doesn't always have the right date: it lists the fall 2025
Talladega race on Nov 17, but it ran on Oct 19. Because history depends on the order
of races, each row takes its date from the race's own **results** feed instead.

## When to run it

After every race, then run `race_train.py`.
