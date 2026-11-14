"""
methods.py - the methods we compare, and the one scoring function they all share.

Plain words: this file is the "rule book". The practice exam (05_train_eval.py)
and the real exam (05b_test.py) both import from here, so every method is
built and scored in exactly the same way in both places.

- Majority:  always answers the most common class.
- NdviDrop:  "damaged if greenness dropped by more than X since the same month in 2022".
- TreeModel: a RandomForest (200 decision trees that vote) or LightGBM, on a chosen list of features.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ROOT, PROCESSED, SEED, DATES, CLASSES,  # noqa: E402
                    FEATURES_NDVI, FEATURES_SPECTRAL, FEATURES_ALL)
from common import load_table  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score  # noqa: E402

LABELLED = [d for d, w in DATES.items() if w["labelled"]]   # ["2024-12", "2025-07"]
TIME_FIT, TIME_CHECK = "2024-12", "2025-07"                 # time test: learn in Dec 2024, check in Jul 2025
MODELS_DIR = ROOT / "models"
FEATURE_SETS = {"rf_ndvi": FEATURES_NDVI,
                "rf_spectral": FEATURES_NDVI + FEATURES_SPECTRAL,
                "rf_all": FEATURES_ALL,
                "lgbm_all": FEATURES_ALL}
ORDER = ["majority", "ndvi_drop", "rf_ndvi", "rf_spectral", "rf_all", "lgbm_all"]
MAIN_MODEL, MAIN_BASELINE = "rf_all", "ndvi_drop"           # fixed from cross-validation, before the test was opened
EXAMS = [f"same-date {d}" for d in LABELLED] + [f"time {TIME_FIT}->{TIME_CHECK}"]

# The v2 report compares five methods on the same pixels (see 05c_train_v2.py and 05b_test.py):
#   majority, ndvi_drop (v1 rule), rule_war (new, stronger rule), rf_all (the v1 model), v2 (the new model)
#   v2_2m = a declared side result: the same v2 features, but learned from the two exam months only
V2_ORDER = ["majority", "ndvi_drop", "rule_war", "rf_all", "v2_2m", "v2"]
V2_SIDE = {"hist_2m": "v2_2m"}             # candidate name in 05c_train_v2.py -> method name in the report
V2_MAIN, V2_BASELINE = "v2", "rule_war"
V2_GAPS = [("v2", "rule_war"), ("v2", "rf_all"), ("v2", "ndvi_drop")]     # v2 against each rival, with paired intervals


# ------------------------------------------------------------------ data
def stacked_table(part: str, unlock_test: bool = False) -> pd.DataFrame:
    """Pixels of one part ("train" or "test") for both label dates, stacked: one row per pixel per date."""
    parts = []
    for d in LABELLED:
        t = load_table(d, part, unlock_test=unlock_test)
        t["date"] = d
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    # guard: every row really is on land of that part (split raster: 1 = train, 2 = test)
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        assert (src.read(1)[t["row"], t["col"]] == {"train": 1, "test": 2}[part]).all()
    return t


# ------------------------------------------------------------------ scoring
def macro_f1_fast(y, pred):
    """Macro F1 with plain numpy. y = UN label (0/1), pred = True where we say "damaged"."""
    tp1 = np.sum(pred & (y == 1)); fp1 = np.sum(pred & (y == 0)); fn1 = np.sum(~pred & (y == 1))
    tp0 = np.sum(~pred & (y == 0)); fp0 = fn1; fn0 = fp1
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)  # noqa: E731
    return (f1(tp0, fp0, fn0) + f1(tp1, fp1, fn1)) / 2


def scores_from_cm(tn, fp, fn, tp) -> dict:
    """All reported numbers from the four confusion-matrix counts.
    tn = intact and called intact, fp = intact but called damaged,
    fn = damaged but called intact, tp = damaged and called damaged."""
    div = lambda a, b: float(a / b) if b else 0.0  # noqa: E731
    p0, r0 = div(tn, tn + fn), div(tn, tn + fp)      # "intact":  precision, recall
    p1, r1 = div(tp, tp + fp), div(tp, tp + fn)      # "damaged": precision, recall
    f0, f1 = div(2 * p0 * r0, p0 + r0), div(2 * p1 * r1, p1 + r1)
    return {"n": int(tn + fp + fn + tp), "accuracy": div(tn + tp, tn + fp + fn + tp),
            "balanced_accuracy": (r0 + r1) / 2, "macro_f1": (f0 + f1) / 2,
            f"precision_{CLASSES[0]}": p0, f"recall_{CLASSES[0]}": r0, f"f1_{CLASSES[0]}": f0,
            f"precision_{CLASSES[1]}": p1, f"recall_{CLASSES[1]}": r1, f"f1_{CLASSES[1]}": f1,
            "cm": [[int(tn), int(fp)], [int(fn), int(tp)]]}   # rows = UN label, columns = prediction


def scores(y, pred) -> dict:
    """Every number we report for one set of predictions (uses scikit-learn's metric functions)."""
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = {"n": int(len(y)), "accuracy": float((y == pred).mean()),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "macro_f1": float(f.mean())}
    for i, c in enumerate(CLASSES):
        out[f"precision_{c}"], out[f"recall_{c}"], out[f"f1_{c}"] = float(p[i]), float(r[i]), float(f[i])
    out["cm"] = cm.tolist()    # rows = UN label (intact, damaged), columns = prediction
    return out


# ------------------------------------------------------------------ the methods
def best_threshold(x, y) -> float:
    """The EXACT best cut for "damaged if x < cut": tries every place between two different
    values of x (plus "everything damaged" and "nothing damaged") and keeps the best macro F1.
    Sorting once and counting with cumulative sums makes all ~250,000 tries take a second."""
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o].astype(np.int64)
    n, n1 = len(xs), int(ys.sum())
    n0 = n - n1
    i = np.arange(n + 1)                               # i = how many of the lowest values are called damaged
    tp1 = np.concatenate([[0], np.cumsum(ys)])         # damaged and called damaged
    fp1 = i - tp1                                      # intact but called damaged
    fn1 = n1 - tp1                                     # damaged but called intact
    tp0 = n0 - fp1                                     # intact and called intact
    with np.errstate(invalid="ignore", divide="ignore"):
        f1_dam = np.where(2 * tp1 + fp1 + fn1 > 0, 2 * tp1 / (2 * tp1 + fp1 + fn1), 0.0)
        f1_int = np.where(2 * tp0 + fn1 + fp1 > 0, 2 * tp0 / (2 * tp0 + fn1 + fp1), 0.0)
    score = (f1_dam + f1_int) / 2
    valid = np.ones(n + 1, bool)                       # a cut must fall between two DIFFERENT values
    valid[1:n] = xs[1:] > xs[:-1]
    score[~valid] = -1
    b = int(np.argmax(score))
    if b == 0:
        return float(xs[0] - 1.0)                      # nothing called damaged
    if b == n:
        return float(xs[-1] + 1.0)                     # everything called damaged
    return float((xs[b - 1] + xs[b]) / 2)


class Majority:
    """Always answers the most common class in the data it was fitted on."""
    def fit(self, t):
        self.value = int(t["label"].mean() >= 0.5)
        return self

    def predict(self, t):
        return np.full(len(t), self.value, np.int8)


class NdviDrop:
    """Damaged if d_ndvi (greenness now minus same month 2022) is below a threshold.
    One threshold per date, each chosen to give the best macro F1 on the rows it is fitted on.
    A date it was not fitted on uses the threshold of the fitted date(s) (time test)."""
    def fit(self, t):
        self.threshold = {}
        for d, g in t.groupby("date"):
            self.threshold[d] = best_threshold(g["d_ndvi"].to_numpy(), g["label"].to_numpy())
        return self

    def predict(self, t):
        fallback = float(np.mean(list(self.threshold.values())))
        thr = t["date"].map(self.threshold).fillna(fallback).to_numpy()
        return (t["d_ndvi"].to_numpy() < thr).astype(np.int8)


class TreeModel:
    """RandomForest or LightGBM on one feature set. The model only ever sees those columns."""
    def __init__(self, kind, features):
        self.kind, self.features = kind, list(features)

    def _new(self):
        if self.kind == "rf":
            return RandomForestClassifier(
                n_estimators=200,            # 200 trees vote
                min_samples_leaf=20,         # a final answer needs >= 20 pixels behind it (less memorising)
                max_features="sqrt",         # each question picks from a random few features
                max_samples=0.25,            # each tree learns from a random 25% of the pixels
                class_weight="balanced_subsample",   # intact pixels are rarer, so they count more
                n_jobs=-1, random_state=SEED)
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
            subsample=0.5, subsample_freq=1, colsample_bytree=0.8, class_weight="balanced",
            random_state=SEED, n_jobs=-1, verbose=-1)

    def fit(self, t):
        self.model = self._new().fit(t[self.features].to_numpy(), t["label"].to_numpy())
        return self

    def proba(self, t):
        """The model's "damaged" score per pixel, from 0 (surely intact) to 1 (surely damaged)."""
        return self.model.predict_proba(t[self.features].to_numpy())[:, 1]

    def predict(self, t):
        return (self.proba(t) >= 0.5).astype(np.int8)


def make(name):
    if name == "majority":
        return Majority()
    if name == "ndvi_drop":
        return NdviDrop()
    return TreeModel("lgbm" if name.startswith("lgbm") else "rf", FEATURE_SETS[name])
