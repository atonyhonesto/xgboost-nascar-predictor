"""The app's models: every car, any track. XGBoost ranks the field, simulation.py plays it out.

Two XGBoost models, trained on every car in every Cup points race since 2018:
  ranker  XGBRanker (rank:pairwise), one group per race, trained on cars still running
          at the finish: learns who is quicker than whom on this kind of track
  dnf     XGBClassifier: each car's chance of not running at the finish

The features (built in race_data.py) are all known before the race: recent form,
this track's history, this kind of track's history, team form and qualifying.
Track type is a categorical feature, so one model learns how form carries over to
short tracks, intermediates, superspeedways and road courses.
"""
import json

import numpy as np
import pandas as pd
import xgboost as xgb

import simulation as sim
from race_data import OUT as TABLE

ROOT = TABLE.parent.parent
FOLDER = ROOT / "models" / "race"
REPORTS = ROOT / "reports"

FEATURES = [
    "start_pct_field", "qual_gap_to_pole_pct", "track_type", "track_miles", "make", "playoff",
    "career_starts", "form_fin", "form_rating", "form_run_pos", "form_top10", "form_dnf",
    "season_rating", "trk_starts", "trk_mean_fin", "trk_mean_rating", "trk_top10_rate",
    "type_mean_fin", "type_mean_rating", "type_top15_laps", "team_form_fin", "team_form_rating",
]
CATEGORIES = {"track_type": ["short", "intermediate", "superspeedway", "road"],
              "make": ["Chevrolet", "Ford", "Toyota"]}

RANKER_PARAMS = dict(objective="rank:pairwise", n_estimators=300, learning_rate=0.05, max_depth=4,
                     min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                     tree_method="hist", enable_categorical=True, random_state=42, n_jobs=-1)
DNF_PARAMS = dict(objective="binary:logistic", n_estimators=200, learning_rate=0.05, max_depth=3,
                  min_child_weight=10, subsample=0.8, colsample_bytree=0.8, reg_lambda=5.0,
                  tree_method="hist", enable_categorical=True, random_state=42, n_jobs=-1)

BACKTEST_FROM = "2025-01-01"  # every race from here on is forecast with only earlier races
N_SIMS = 10_000
N_GRIDS = 50                  # random starting grids drawn when the lineup isn't known


# ---------------------------------------------------------------- data

def prepare(df):
    """Set the categorical columns XGBoost reads natively."""
    df = df.copy()
    for col, cats in CATEGORIES.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    df["race_label"] = df["race_date"].astype(str) + " · " + df["track"].astype(str)
    return df


def load_table(path=TABLE):
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run `python race_data.py` first.")
    return prepare(pd.read_csv(path, dtype={"car_number": str}))


# ---------------------------------------------------------------- models

def fit(train):
    """Train the ranker (on cars still running) and the DNF model (on everyone)."""
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
    """SHAP values of the ranker score (rows x features): what lifts or sinks each car."""
    dm = xgb.DMatrix(rows[FEATURES], enable_categorical=True)
    values = models["ranker"].get_booster().predict(dm, pred_contribs=True)[:, :-1]
    return pd.DataFrame(values, columns=FEATURES, index=rows.index)


def importance(models, which="ranker"):
    gain = pd.Series(models[which].get_booster().get_score(importance_type="total_gain"))
    return (gain / gain.sum()).reindex(FEATURES).fillna(0).sort_values(ascending=False)


def save(models, temperature, info):
    FOLDER.mkdir(parents=True, exist_ok=True)
    for name, m in models.items():
        m.save_model(FOLDER / f"{name}.json")
    meta = {"features": FEATURES, "temperature": temperature, **info}
    (FOLDER / "meta.json").write_text(json.dumps(meta, indent=2))


def load():
    if not (FOLDER / "meta.json").exists():
        raise FileNotFoundError(f"No saved models in {FOLDER}. Run `python race_train.py` first.")
    meta = json.loads((FOLDER / "meta.json").read_text())
    ranker, dnf = xgb.XGBRanker(), xgb.XGBClassifier()
    ranker.load_model(FOLDER / "ranker.json")
    dnf.load_model(FOLDER / "dnf.json")
    return {"ranker": ranker, "dnf": dnf}, meta


# ---------------------------------------------------------------- forecasting

def forecast_race(models, rows, temperature, n_sims=N_SIMS, seed=0):
    """Every car's chances in one race. If the grid isn't set, random grids are simulated too."""
    rng = np.random.default_rng(seed)
    rows = prepare(rows.reset_index(drop=True)) if rows["make"].dtype != "category" else rows.reset_index(drop=True)
    if rows["start_position"].notna().all():
        scores, p_dnf = predict(models, rows)
        finish, out = sim.simulate(scores, p_dnf, n_sims, temperature, rng)
    else:
        n, per_grid = len(rows), n_sims // N_GRIDS
        finish, out = [], []
        for _ in range(N_GRIDS):
            grid = rows.copy()
            grid["start_position"] = rng.permutation(n) + 1
            grid["start_pct_field"] = grid["start_position"] / n
            scores, p_dnf = predict(models, grid)
            f, o = sim.simulate(scores, p_dnf, per_grid, temperature, rng)
            finish.append(f)
            out.append(o)
        finish, out = np.vstack(finish), np.vstack(out)
    return pd.concat([rows, sim.summarise(finish, out)], axis=1), finish


def finish_distribution(finish):
    """Share of simulations each car finished in each position (cars x positions)."""
    n = finish.shape[1]
    counts = np.stack([np.bincount(finish[:, i], minlength=n + 1)[1:] for i in range(n)]) / len(finish)
    return pd.DataFrame(counts, columns=[f"pos_{k}" for k in range(1, n + 1)])


# ---------------------------------------------------------------- evaluation

def backtest(df, start=BACKTEST_FROM, n_sims=2000, progress=None):
    """Walk forward through every race from `start`: train on all earlier races, forecast it.

    Returns (per-car results, out-of-sample ranker scores for fitting the temperature).
    """
    races = df.drop_duplicates("race_id").sort_values("race_date")
    test_ids = list(races.loc[races["race_date"] >= start, "race_id"])
    oof, results = [], []
    for i, race_id in enumerate(test_ids):
        test = df[df["race_id"] == race_id].reset_index(drop=True)
        train = df[df["race_date"] < test["race_date"].iloc[0]]
        models = fit(train)
        temperature = sim.fit_temperature(pd.concat(oof) if oof else pd.DataFrame())
        scores, _ = predict(models, test)
        oof.append(test.assign(score=scores))
        fc, _ = forecast_race(models, test, temperature, n_sims=n_sims, seed=i)
        fc["temperature"] = temperature
        # Baseline for top-10 chances: the driver's recent top-10 rate, shrunk towards
        # the field average so a driver with few races isn't 0% or 100%.
        prior = 10 / fc["field_size"]
        n_recent = fc["career_starts"].clip(upper=5)
        fc["base_top10"] = (fc["form_top10"].fillna(0) * n_recent + 2 * prior) / (n_recent + 2)
        fc["base_dnf"] = train["dnf"].mean()
        results.append(fc)
        if progress:
            progress(i + 1, len(test_ids), fc["race_label"].iloc[0])
    return pd.concat(results, ignore_index=True), pd.concat(oof, ignore_index=True)


def _spearman(g, col):
    return g[col].rank().corr(g["finish_position"].rank())


def score(bt, by=None):
    """Model vs baselines. Rank correlation and top-10 hits: higher is better; the rest: lower."""
    def block(b):
        races = b.groupby("race_id")
        form = b.assign(form_fin=b["form_fin"].fillna(b["form_fin"].median()))
        hits = lambda d, col: float(d.groupby("race_id").apply(lambda g: g.nsmallest(10, col)["top10"].sum()).mean())
        rows = [
            ("Rank correlation with the result (higher is better)",
             races.apply(_spearman, "exp_finish").mean(), races.apply(_spearman, "start_position").mean(), "starting grid"),
            ("Rank correlation with the result (higher is better)",
             races.apply(_spearman, "exp_finish").mean(),
             form.groupby("race_id").apply(_spearman, "form_fin").mean(), "recent form (last 5 finishes)"),
            ("Predicted top 10 that finished top 10 (of 10)", hits(b, "exp_finish"), hits(b, "start_position"),
             "starting grid"),
            ("Predicted top 10 that finished top 10 (of 10)", hits(b, "exp_finish"), hits(form, "form_fin"),
             "recent form (last 5 finishes)"),
            ("Top-10 log loss (lower is better)", sim.logloss(b["p_top10"], b["top10"]),
             sim.logloss(b["base_top10"], b["top10"]), "driver's recent top-10 rate"),
            ("Top-10 Brier score (lower is better)", ((b["p_top10"] - b["top10"]) ** 2).mean(),
             ((b["base_top10"] - b["top10"]) ** 2).mean(), "driver's recent top-10 rate"),
            ("Win log loss (lower is better)", sim.logloss(b["p_win"], b["win"]),
             sim.logloss(1 / b["field_size"], b["win"]), "everyone equal"),
            ("DNF Brier score (lower is better)", ((b["p_dnf"] - b["dnf"]) ** 2).mean(),
             ((b["base_dnf"] - b["dnf"]) ** 2).mean(), "field DNF rate"),
        ]
        res = pd.DataFrame(rows, columns=["metric", "model", "baseline", "baseline_name"])
        res[["model", "baseline"]] = res[["model", "baseline"]].astype(float)
        higher = res["metric"].str.contains("higher|of 10")
        res["model_better"] = np.where(higher, res["model"] > res["baseline"], res["model"] < res["baseline"])
        res["races"] = b["race_id"].nunique()
        return res

    if by is None:
        return block(bt)
    return pd.concat([block(g).assign(**{by: k}) for k, g in bt.groupby(by, observed=True)], ignore_index=True)
