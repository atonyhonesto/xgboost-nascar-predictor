"""Whole-field models: XGBoost ranks the field, NumPy simulates the race.

Two XGBoost models, trained on every car in every race:
  ranker  XGBRanker (rank:pairwise), grouped by race: learns who finishes ahead of whom
  dnf     XGBClassifier: each car's chance of not running at the finish

The simulation (Monte Carlo) turns them into per-car chances:
  1. each car fails to finish with its DNF probability;
  2. the cars still running are ordered by ranker score + Gumbel noise.
With rank:pairwise, P(i ahead of j) = sigmoid(score_i - score_j), and adding Gumbel
noise to the scores gives exactly that, so the noise scale ("temperature") has a
natural value of 1. It is then fitted on out-of-sample races to fix any over- or
under-confidence.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from model import MODELS, REPORTS
from simulation import calibration, fit_temperature, plackett_luce_loglik, simulate, summarise  # noqa: F401
from simulation import logloss as _logloss

TABLE = MODELS.parent / "data" / "field_talladega.csv"
FOLDER = MODELS / "field"

FEATURES = [
    "start_pct_field", "qual_gap_to_pole_pct", "is_playoff", "is_gen7", "make",
    "drv_starts", "drv_prev_fin", "drv_mean_fin", "drv_top10_rate", "drv_dnf_rate",
    "drv_prev_rating", "drv_mean_rating", "drv_mean_run_pos", "drv_mean_pct_laps_top10",
    "drv_mean_laps_led", "team_mean_fin", "team_mean_rating", "team_top10_rate",
]
MAKES = ["Chevrolet", "Ford", "Toyota"]

RANKER_PARAMS = dict(objective="rank:pairwise", n_estimators=150, learning_rate=0.05, max_depth=3,
                     min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                     tree_method="hist", enable_categorical=True, random_state=42, n_jobs=1)
DNF_PARAMS = dict(objective="binary:logistic", n_estimators=100, learning_rate=0.05, max_depth=2,
                  min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                  tree_method="hist", enable_categorical=True, random_state=42, n_jobs=1)

MIN_TRAIN_RACES = 6        # the backtest starts once this many earlier races exist
N_SIMS = 10_000
N_GRIDS = 50               # random starting grids drawn when the lineup isn't known


# ---------------------------------------------------------------- data

def load_table(path=TABLE):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run `python field_data.py` first.")
    df = pd.read_csv(path, dtype={"car_number": str, "race_id": str})
    df["make"] = pd.Categorical(df["make"], categories=MAKES)
    df["race_label"] = df["season"].astype(str) + " " + df["season_half"]
    return df


def split(df):
    """(completed races, upcoming-race rows), each sorted by race date."""
    done = df["finish_position"].notna()
    return (df[done].sort_values(["race_date", "race_id"]).reset_index(drop=True),
            df[~done].reset_index(drop=True))


def race_ids(df):
    return list(df.drop_duplicates("race_id").sort_values("race_date")["race_id"])


# ---------------------------------------------------------------- models

def fit(train):
    """Train the ranker and DNF model on the given races.

    The ranker only learns the order of cars still running at the finish: a car wrecked
    on lap 50 is classified near the back by bad luck, not pace, and the simulation
    handles DNFs separately anyway.
    """
    train = train.sort_values(["race_date", "race_id"])
    running = train[train["dnf"] == 0]
    qid = pd.factorize(running["race_id"])[0]
    relevance = running["field_size"] - running["finish_position"]  # higher = finished further forward
    ranker = xgb.XGBRanker(**RANKER_PARAMS).fit(running[FEATURES], relevance, qid=qid)
    dnf = xgb.XGBClassifier(**DNF_PARAMS).fit(train[FEATURES], train["dnf"])
    return {"ranker": ranker, "dnf": dnf}


def predict(models, rows):
    """(ranker score, DNF probability) per row."""
    return (models["ranker"].predict(rows[FEATURES]),
            models["dnf"].predict_proba(rows[FEATURES])[:, 1])


def contributions(models, rows):
    """SHAP values of the ranker score per car (rows x features)."""
    dm = xgb.DMatrix(rows[FEATURES], enable_categorical=True)
    values = models["ranker"].get_booster().predict(dm, pred_contribs=True)[:, :-1]
    return pd.DataFrame(values, columns=FEATURES, index=rows.index)


def importance(models, which="ranker"):
    gain = pd.Series(models[which].get_booster().get_score(importance_type="total_gain"))
    return (gain / gain.sum()).reindex(FEATURES).fillna(0).sort_values(ascending=False)


def save(models, temperature, trained_races):
    FOLDER.mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        m.save_model(FOLDER / f"{name}.json")
    meta = {"features": FEATURES, "temperature": temperature, "trained_races": trained_races}
    (FOLDER / "meta.json").write_text(json.dumps(meta, indent=2))


def load():
    if not (FOLDER / "meta.json").exists():
        raise FileNotFoundError(f"No saved field models in {FOLDER}. Run `python field_train.py` first.")
    meta = json.loads((FOLDER / "meta.json").read_text())
    ranker, dnf = xgb.XGBRanker(), xgb.XGBClassifier()
    ranker.load_model(FOLDER / "ranker.json")
    dnf.load_model(FOLDER / "dnf.json")
    return {"ranker": ranker, "dnf": dnf}, meta


# ---------------------------------------------------------------- simulation
# simulate(), summarise(), fit_temperature() and calibration() live in simulation.py.

def forecast_race(models, rows, temperature, n_sims=N_SIMS, seed=0):
    """Per-car chances for one race. If the grid is unknown, random grids are drawn too."""
    rng = np.random.default_rng(seed)
    rows = rows.reset_index(drop=True)
    if rows["start_position"].notna().all():
        scores, p_dnf = predict(models, rows)
        finish, out = simulate(scores, p_dnf, n_sims, temperature, rng)
    else:
        n, per_grid = len(rows), n_sims // N_GRIDS
        finish, out = [], []
        for _ in range(N_GRIDS):
            grid = rows.copy()
            grid["start_position"] = rng.permutation(n) + 1
            grid["start_pct_field"] = grid["start_position"] / n
            scores, p_dnf = predict(models, grid)
            f, o = simulate(scores, p_dnf, per_grid, temperature, rng)
            finish.append(f)
            out.append(o)
        finish, out = np.vstack(finish), np.vstack(out)
    return pd.concat([rows, summarise(finish, out)], axis=1)


# ---------------------------------------------------------------- backtest

def backtest(past, n_sims=4000):
    """Walk forward: for race i, train on races before it, then forecast race i.

    Returns one row per car per tested race with the model's chances, two baselines
    and what actually happened.
    """
    ids = race_ids(past)
    oof, results = [], []
    for i in range(MIN_TRAIN_RACES, len(ids)):
        train = past[past["race_id"].isin(ids[:i])]
        test = past[past["race_id"] == ids[i]].reset_index(drop=True)
        models = fit(train)
        temperature = fit_temperature(pd.concat(oof) if oof else pd.DataFrame())
        scores, _ = predict(models, test)
        oof.append(test.assign(score=scores))
        fc = forecast_race(models, test, temperature, n_sims=n_sims, seed=i)
        fc["temperature"], fc["train_races"] = temperature, i
        # Baseline 1: the grid. Baseline 2: each driver's own Talladega top-10 record,
        # shrunk towards the field average (10 of ~40) so rookies aren't 0% or 100%.
        prior = 10 / fc["field_size"]
        fc["base_top10"] = ((fc["drv_top10_rate"].fillna(0) * fc["drv_starts"] + 2 * prior)
                            / (fc["drv_starts"] + 2))
        fc["base_uniform_top10"] = prior
        fc["base_dnf"] = train["dnf"].mean()
        results.append(fc)
    return pd.concat(results, ignore_index=True), pd.concat(oof, ignore_index=True)


def score(bt):
    """Model vs baselines. Rank correlation and top-10 hits: higher is better; the rest: lower."""
    races = bt.groupby("race_id")
    rows = []

    def spearman(col):
        return float(races.apply(lambda g: g[col].rank().corr(g["finish_position"].rank())).mean())

    hist_fin = bt["drv_mean_fin"].fillna(bt["drv_mean_fin"].median())
    rows += [
        {"metric": "Rank correlation with the result (higher is better)", "model": spearman("exp_finish"),
         "baseline": spearman("start_position"), "baseline_name": "starting grid"},
        {"metric": "Rank correlation with the result (higher is better)", "model": spearman("exp_finish"),
         "baseline": float(bt.assign(h=hist_fin).groupby("race_id").apply(
             lambda g: g["h"].rank().corr(g["finish_position"].rank())).mean()),
         "baseline_name": "driver's average finish"},
    ]

    def hits(col):
        return float(races.apply(lambda g: g.nsmallest(10, col)["top10"].sum()).mean())

    rows.append({"metric": "Predicted top 10 that finished top 10 (of 10)", "model": hits("exp_finish"),
                 "baseline": hits("start_position"), "baseline_name": "starting grid"})
    for base, name in (("base_top10", "driver's top-10 record"), ("base_uniform_top10", "everyone equal")):
        rows.append({"metric": "Top-10 log loss (lower is better)", "model": _logloss(bt["p_top10"], bt["top10"]),
                     "baseline": _logloss(bt[base], bt["top10"]), "baseline_name": name})
        rows.append({"metric": "Top-10 Brier score (lower is better)",
                     "model": float(((bt["p_top10"] - bt["top10"]) ** 2).mean()),
                     "baseline": float(((bt[base] - bt["top10"]) ** 2).mean()), "baseline_name": name})
    rows.append({"metric": "Win log loss (lower is better)", "model": _logloss(bt["p_win"], bt["win"]),
                 "baseline": _logloss(1 / bt["field_size"], bt["win"]), "baseline_name": "everyone equal"})
    rows.append({"metric": "DNF Brier score (lower is better)",
                 "model": float(((bt["p_dnf"] - bt["dnf"]) ** 2).mean()),
                 "baseline": float(((bt["base_dnf"] - bt["dnf"]) ** 2).mean()), "baseline_name": "field DNF rate"})
    res = pd.DataFrame(rows)
    higher = res["metric"].str.contains("higher|of 10")
    res["model_better"] = np.where(higher, res["model"] > res["baseline"], res["model"] < res["baseline"])
    return res
