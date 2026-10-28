"""Forecast Larson's next Talladega race from the saved models.

Run train.py first. Then, before qualifying:
  python predict.py                        # forecast at every start position
After qualifying:
  python predict.py --auto                 # pull start position + speeds from NASCAR's feed
  python predict.py --start 12 --qual-speed 181.2 --pole-speed 182.4
Options:
  --features core|enriched                 # which saved model set to use (default: core)
"""
import argparse

from enrich import NEXT_RACE, NEXT_RACE_ID
from fetch_nascar import starting_info
from model import (DEFAULT_FEATURE_SET, FEATURE_SETS, TARGETS, forecast, load_all, load_table,
                   set_race_inputs, split, sweep_start_positions)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, help="starting position")
    p.add_argument("--qual-speed", type=float, help="Larson's qualifying speed (mph)")
    p.add_argument("--pole-speed", type=float, help="pole winner's speed (mph)")
    p.add_argument("--auto", action="store_true", help="read the lineup from NASCAR's weekend feed")
    p.add_argument("--features", choices=FEATURE_SETS, default=DEFAULT_FEATURE_SET)
    return p.parse_args()


def print_forecast(pred, models, row):
    print(f"  Top-10 chance:     {pred['top10']:.0%}   (his base rate: {models['top10'].fallback:.0%})")
    print(f"  DNF chance:        {pred['dnf']:.0%}   (his base rate: {models['dnf'].fallback:.0%})")
    print(f"  Expected finish:   {pred['finish']:.1f}  (his average: {models['finish'].fallback:.1f})")
    contrib = models["top10"].contributions(row)
    contrib = contrib[contrib.abs() > 0.005].sort_values(key=abs, ascending=False).head(5)
    if len(contrib):
        print("  Biggest influences on the top-10 chance:")
        for feat, val in contrib.items():
            value = row[feat].iloc[0]
            shown = "not set" if value != value else f"{value:.3g}"  # NaN != NaN
            print(f"    {'+' if val > 0 else '-'} {feat} = {shown}")


def main():
    args = parse_args()
    _, upcoming = split(load_table())
    if upcoming.empty:
        raise SystemExit("No upcoming race row in the table. Update NEXT_RACE in enrich.py.")
    models = load_all(args.features)
    row = upcoming.iloc[[0]]
    print(f"Kyle Larson, Talladega, {row['race_date'].iloc[0]}  [{args.features} features]\n")

    start, qual, pole = args.start, args.qual_speed, args.pole_speed
    if args.auto:
        info = starting_info(NEXT_RACE["season"], NEXT_RACE_ID, NEXT_RACE["car_number"])
        if info is None:
            print("Lineup not posted yet; showing every start position instead.\n")
        else:
            start, qual, pole = info["start_position"], info["qualifying_speed"], info["pole_speed"]
            print(f"From NASCAR's feed: starts P{start}, qualified {qual} mph, pole {pole} mph.\n")

    if start is None:
        table = sweep_start_positions(models, row)
        table["top10"] = table["top10"].map("{:.0%}".format)
        table["dnf"] = table["dnf"].map("{:.0%}".format)
        table["finish"] = table["finish"].map("{:.1f}".format)
        print(table.rename(columns={t: TARGETS[t]["label"] for t in TARGETS}).to_string(index=False))
        return

    row = set_race_inputs(row, start, qual, pole)
    print_forecast(forecast(models, row), models, row)
    print("\nBacktest: these models do not yet beat his plain averages. See reports/metrics.csv.")


if __name__ == "__main__":
    main()
