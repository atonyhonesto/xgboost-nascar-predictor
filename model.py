"""Shared modelling code: feature sets, XGBoost models, walk-forward backtest, forecasting.

train.py, predict.py and app.py all import from here, so every entry point uses the
same features, settings and preprocessing.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
TABLE = DATA / "larson_talladega_model_enriched.csv"

# Everything here is known before the green flag (history features are lagged).
CORE_FEATURES = [
    "start_pct_field", "qual_gap_to_pole_pct", "is_playoff", "is_gen7", "is_hendrick",
    "prior_talladega_starts", "prev_fin", "hist_mean_fin", "prev_rating", "hist_mean_rating",
    "prev_avg_run_pos", "hist_mean_avg_run_pos", "hist_mean_laps_led", "hist_dnf_rate",
    "hist_crash_rate",
]
# Lap/pit/notes features exist only from 2021 on; XGBoost treats the earlier blanks as missing.
ENRICHED_FEATURES = CORE_FEATURES + [
    "prev_pct_laps_top10", "hist_mean_pct_laps_top10", "prev_pct_laps_top5",
    "hist_mean_late_race_avg_pos", "hist_mean_run_pos_std", "hist_mean_median_tire_stop_sec",
    "hist_mean_pit_road_pos_net", "hist_mean_incident_notes",
]
FEATURE_SETS = {"core": CORE_FEATURES, "enriched": ENRICHED_FEATURES}
DEFAULT_FEATURE_SET = "core"

TARGETS = {
    "top10": {"column": "target_top10", "kind": "binary", "label": "Top-10 finish"},
    "dnf": {"column": "target_dnf", "kind": "binary", "label": "Did not finish (DNF)"},
    "finish": {"column": "target_finish_position", "kind": "regression", "label": "Finish position"},
}

# Shallow, heavily regularised trees: there are only 23 races to learn from.
XGB_PARAMS = dict(
    n_estimators=80, learning_rate=0.05, max_depth=2, min_child_weight=1.0,
    subsample=1.0, colsample_bytree=0.8, reg_lambda=5.0, random_state=42, n_jobs=1,
)
MIN_TRAIN_RACES = 8  # the backtest starts once this many earlier races exist
PROB_CLIP = 0.01     # keeps log loss finite when a probability hits 0 or 1


# ---------------------------------------------------------------- data

def load_table(path=TABLE):
    """Enriched model table plus a few derived flags, in date order."""
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found. Run `python enrich.py` first.")
    df = pd.read_csv(path).sort_values("race_date").reset_index(drop=True)
    df["is_playoff"] = (df["playoff_race"] == "Y").astype(int)
    df["is_gen7"] = (df["season"] >= 2022).astype(int)
    df["is_hendrick"] = (df["team"] == "Hendrick Motorsports").astype(int)
    df["race_label"] = df["season"].astype(str) + " " + df["season_half"]
    return df


def split(df):
    """(past races with results, the upcoming race row(s) without results)."""
    done = df["target_finish_position"].notna()
    return df[done].reset_index(drop=True), df[~done].reset_index(drop=True)


def set_race_inputs(row, start_position=None, qual_speed=None, pole_speed=None):
    """Copy of the upcoming-race row with the qualifying-day inputs filled in."""
    row = row.copy()
    if start_position is not None:
        row["start_position"] = start_position
        row["start_pct_field"] = start_position / row["field_size"].iloc[0]
    if qual_speed and pole_speed:
        row["qualifying_speed_mph"] = qual_speed
        row["qual_gap_to_pole_pct"] = (pole_speed - qual_speed) / pole_speed * 100
    return row


# ---------------------------------------------------------------- models

class TargetModel:
    """One XGBoost model for one target, falling back to the base rate when it can't fit."""

    def __init__(self, target, features):
        self.target, self.features = target, list(features)
        self.spec = TARGETS[target]
        self.model, self.fallback = None, np.nan

    def fit(self, train):
        y = train[self.spec["column"]]
        self.fallback = float(y.mean())
        if self.spec["kind"] == "binary":
            # A classifier needs both outcomes in the training rows.
            if y.nunique() < 2:
                self.model = None
                return self
            self.model = xgb.XGBClassifier(objective="binary:logistic", **XGB_PARAMS)
        else:
            self.model = xgb.XGBRegressor(objective="reg:squarederror", **XGB_PARAMS)
        self.model.fit(train[self.features], y)
        return self

    def predict(self, rows):
        if self.model is None:
            return np.full(len(rows), self.fallback)
        if self.spec["kind"] == "binary":
            return self.model.predict_proba(rows[self.features])[:, 1]
        return np.clip(self.model.predict(rows[self.features]), 1, rows["field_size"].to_numpy())

    def contributions(self, row):
        """Per-feature push on this one prediction (SHAP values; log-odds for binary targets)."""
        if self.model is None:
            return pd.Series(0.0, index=self.features)
        dm = xgb.DMatrix(row[self.features])
        values = self.model.get_booster().predict(dm, pred_contribs=True)[0]
        return pd.Series(values[:-1], index=self.features)

    def importance(self):
        """Total gain per feature, normalised to sum to 1."""
        if self.model is None:
            return pd.Series(dtype=float)
        gain = pd.Series(self.model.get_booster().get_score(importance_type="total_gain"))
        return (gain / gain.sum()).reindex(self.features).fillna(0).sort_values(ascending=False)

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_model(folder / f"{self.target}.json")
        meta = {"features": self.features, "fallback": self.fallback, "fitted": self.model is not None}
        (folder / f"{self.target}_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, folder, target):
        meta = json.loads((folder / f"{target}_meta.json").read_text())
        tm = cls(target, meta["features"])
        tm.fallback = meta["fallback"]
        if meta["fitted"]:
            tm.model = (xgb.XGBClassifier() if tm.spec["kind"] == "binary" else xgb.XGBRegressor())
            tm.model.load_model(folder / f"{target}.json")
        return tm


def train_all(past, feature_set=DEFAULT_FEATURE_SET):
    features = FEATURE_SETS[feature_set]
    return {t: TargetModel(t, features).fit(past) for t in TARGETS}


def save_all(models, feature_set):
    for tm in models.values():
        tm.save(MODELS / feature_set)


def load_all(feature_set=DEFAULT_FEATURE_SET):
    folder = MODELS / feature_set
    if not folder.exists():
        raise FileNotFoundError(f"No saved models in {folder}. Run `python train.py` first.")
    return {t: TargetModel.load(folder, t) for t in TARGETS}


# ---------------------------------------------------------------- forecasting

def forecast(models, row):
    """{target: prediction} for a single race row."""
    return {t: float(m.predict(row)[0]) for t, m in models.items()}


def sweep_start_positions(models, row):
    """Forecast at every possible start position, holding everything else fixed."""
    out = []
    for pos in range(1, int(row["field_size"].iloc[0]) + 1):
        pred = forecast(models, set_race_inputs(row, start_position=pos))
        out.append({"start_position": pos, **pred})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- evaluation

def backtest(past, feature_set=DEFAULT_FEATURE_SET):
    """Walk forward in time: train on races before race i, predict race i.

    The baseline for each target is the plain average of that target over the same
    training races (e.g. "his top-10 rate so far"). A model is only useful if it beats it.
    """
    rows = []
    for i in range(MIN_TRAIN_RACES, len(past)):
        train, test = past.iloc[:i], past.iloc[[i]]
        models = train_all(train, feature_set)
        row = {
            "race_date": test["race_date"].iloc[0], "race_label": test["race_label"].iloc[0],
            "start_position": test["start_position"].iloc[0], "train_races": i,
            "actual_finish": test["target_finish_position"].iloc[0],
            "actual_top10": test["target_top10"].iloc[0], "actual_dnf": test["target_dnf"].iloc[0],
        }
        for target, tm in models.items():
            row[f"pred_{target}"] = float(tm.predict(test)[0])
            row[f"base_{target}"] = tm.fallback
        rows.append(row)
    return pd.DataFrame(rows)


def score(bt):
    """Model vs baseline: Brier score and log loss for the probabilities, MAE for finish.

    Lower is better for every metric.
    """
    out = []
    for target in ("top10", "dnf"):
        actual = bt[f"actual_{target}"]
        for metric in ("brier", "log_loss"):
            vals = {}
            for who in ("pred", "base"):
                p = bt[f"{who}_{target}"].clip(PROB_CLIP, 1 - PROB_CLIP)
                vals[who] = (((p - actual) ** 2).mean() if metric == "brier"
                             else -(actual * np.log(p) + (1 - actual) * np.log(1 - p)).mean())
            out.append({"target": target, "metric": metric, "model": vals["pred"], "baseline": vals["base"]})
    out.append({
        "target": "finish", "metric": "mae",
        "model": (bt["pred_finish"] - bt["actual_finish"]).abs().mean(),
        "baseline": (bt["base_finish"] - bt["actual_finish"]).abs().mean(),
    })
    res = pd.DataFrame(out)
    res["model_better"] = res["model"] < res["baseline"]
    return res
