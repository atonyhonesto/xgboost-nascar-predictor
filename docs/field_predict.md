# Every car's chances in your terminal (`field_predict.py`)

The F1-predictor-style output: a table of every car in the race, ordered by expected
finish, with each car's chances. Run `field_train.py` once first, because this script
uses the saved models.

## Using it

**The next race, before anything is known:**
```powershell
.venv\Scripts\python field_predict.py
```
This uses the projected entry list (the last race's 40 cars) and draws random starting
grids.

**A real race, once NASCAR has posted it:**
```powershell
.venv\Scripts\python field_predict.py --season 2027 --race-id 5700
```
This reads the entry list from NASCAR's weekend feed. Before qualifying it draws
random grids. After qualifying it uses the real grid and qualifying speeds. (5700 is
only an example; the real race ID is in NASCAR's 2027 schedule once it's published.)

**More simulations, or save the table:**
```powershell
.venv\Scripts\python field_predict.py --sims 20000 --csv next_race.csv
```

## What you'll see

```
Next Talladega race (projected entry list), 2027-04-25 · 40 cars · grid unknown: random grids · 10,000 simulations

 Rank  Car            Driver               Team Start Exp. finish  Win Top 3 Top 5 Top 10 DNF
    1  #11      Denny Hamlin   Joe Gibbs Racing              17.5 2.9%    9%   15%    31% 12%
    2  #77    Carson Hocevar  Spire Motorsports              18.1 3.1%    9%   16%    30% 13%
    3  #20  Christopher Bell   Joe Gibbs Racing              18.9 2.8%    9%   14%    28% 14%
  ...
```

Notice how close everyone is. The favourite's win chance is about 3%, and an evenly
matched field of 40 would give everyone 2.5%. That's not the model being timid. It's
the most honest reading of how random Talladega is, and the backtest backs it up.

## Drivers new to Talladega

A driver with no Talladega starts has no driver history, so the model leans on their
**team's** record, the manufacturer and (once known) where they start. Their
`drv_starts` is set to 0.
