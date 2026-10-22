# fetch_nascar.py: NASCAR data feeds

Downloads and caches NASCAR's public JSON feeds at `https://cf.nascar.com/cacher/`.
These are the same feeds the [pynascar](https://github.com/ab5525/pynascar) library
wraps. No API key is needed.

```powershell
.venv\Scripts\python fetch_nascar.py     # prints which feeds exist for each Talladega race
```

## Feeds used

| Feed | URL pattern | Contents | Seasons |
|---|---|---|---|
| Schedule | `/{year}/race_list_basic.json` | Every race that year, with `race_id` | 2015 on |
| Lap times | `/{year}/1/{race_id}/lap-times.json` | Every car's time and running position on every lap | 2020 on |
| Pit stops | `/{year}/1/{race_id}/live-pit-data.json` | Each stop: duration, type, positions gained or lost | 2020 on |
| Race notes | `/{year}/1/{race_id}/lap-notes.json` | Race-control notes by lap (cautions, incidents, penalties) | 2020 on |
| Weekend | `/{year}/1/{race_id}/weekend-feed.json` | Entry list, then the lineup after qualifying, then results | current race |

Older races return HTTP 403, which the code treats as "no file". The `1` in the URL
is the Cup series. Talladega's `track_id` is 82.

## Functions

| Function | Returns |
|---|---|
| `get_json(url, cache_path, refresh=False)` | Parsed JSON, read from `data/cache/` when present. `None` if the feed doesn't exist |
| `talladega_races(year)` | Talladega Cup races that year: `race_id`, name, date, spring/fall |
| `fetch_race(year, race_id)` | `{"lap-times": ..., "live-pit-data": ..., "lap-notes": ...}` |
| `weekend_feed(year, race_id)` | The weekend feed, always re-downloaded because it changes all weekend |
| `starting_info(year, race_id, car)` | `{"start_position", "qualifying_speed", "pole_speed", "field_size"}` for one car once the lineup is posted, otherwise `None` |

## Cache
Finished races never change, so their feeds are saved in `data/cache/` and reused.
Delete a file there to force a fresh download. The weekend feed is never cached.

## Quirks found while building this
- The 2017 schedule file serves 2018's races, so races are filtered by date.
- Before qualifying, the weekend feed lists the entry with `starting_position = 0`.
- Races without time trials (e.g. spring 2026) have a lineup but `pole_winner_speed = 0`.
  The qualifying gap is then left blank.
- The first block of the pit feed is the starting grid, with zero or −1 placeholders.
  These rows are skipped.
