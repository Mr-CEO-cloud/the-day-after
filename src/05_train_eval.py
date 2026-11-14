"""
05_train_eval.py - Phase 3: two simple opponents ("baselines") and the AI models,
scored with cross-validation on TRAINING land only.

Plain words:
- The question for every pixel: did the UN mark it "damaged" (1) or
  "non-affected" (0)? Each method below answers that from satellite data.
- Opponent 1, MAJORITY: always answer "damaged" (the most common answer). It
  shows how good a score you get by not looking at the image at all.
- Opponent 2, NDVI-DROP: "damaged if greenness dropped by more than X since the
  same month in 2022". X is picked separately for each date, using training
  pixels only. This is close to how the UN made its map, so it is a strong rival.
- The AI: a RandomForest, a "forest" of 200 decision trees. Each tree learns a
  chain of yes/no questions ("is moisture lower than before?", "was it an
  orchard?") from a random part of the training pixels, and the trees vote.
  We train three versions to see which information helps:
    rf_ndvi      greenness only (the same information the NDVI-drop rule sees)
    rf_spectral  + all 6 bands, moisture (NDMI) and bare-soil (BSI) indices
    rf_all       + pre-war context (productivity, seasonality, WorldCover)
  plus lgbm_all, LightGBM (trees built one after another, each fixing the last
  one's mistakes) on the same features as rf_all, as a second opinion.
- Cross-validation: training blocks are dealt into 5 folds. Five times, we fit
  on 4 folds (minus a 100 m buffer) and check on the 5th. Every pixel is
  checked exactly once, by a model that never saw its block. The locked test
  blocks are NOT used here at all.
- Two practice exams:
    same-date:  fit on both label dates, check on each date
    time test:  fit on Dec 2024 only, check on Jul 2025 (can it handle a new season?)
- Score = macro F1: the F1 score (balance of precision and recall) for "intact"
  and for "damaged", averaged. Plain accuracy would reward always saying
  "damaged", because 66-80% of training pixels are damaged.

Run:
  .venv/Scripts/python.exe -I src/05_train_eval.py cv     # cross-validation scores (results/cv_*)
  .venv/Scripts/python.exe -I src/05_train_eval.py fit    # final models on all training land (models/)
  .venv/Scripts/python.exe -I src/05_train_eval.py figure # redraw results/figures/cv_scores.png from cv_scores.csv
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ROOT, PROCESSED, RESULTS, SEED, DATES, CLASSES,  # noqa: E402
                    FEATURES_NDVI, FEATURES_SPECTRAL, FEATURES_ALL)
from common import load_table, cv_splits  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, balanced_accuracy_score  # noqa: E402

LABELLED = [d for d, w in DATES.items() if w["labelled"]]   # ["2024-12", "2025-07"]
TIME_FIT, TIME_CHECK = "2024-12", "2025-07"
MODELS_DIR = ROOT / "models"
FEATURE_SETS = {"rf_ndvi": FEATURES_NDVI,
                "rf_spectral": FEATURES_NDVI + FEATURES_SPECTRAL,
                "rf_all": FEATURES_ALL,
                "lgbm_all": FEATURES_ALL}
ORDER = ["majority", "ndvi_drop", "rf_ndvi", "rf_spectral", "rf_all", "lgbm_all"]


# ------------------------------------------------------------------ data
def train_table() -> pd.DataFrame:
    """Training pixels for both label dates, stacked (one row per pixel per date)."""
    parts = []
    for d in LABELLED:
        t = load_table(d, "train")
        t["date"] = d
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    # guard: every row really is on training land (the split raster says 1 = train)
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        assert (src.read(1)[t["row"], t["col"]] == 1).all(), "non-train pixel in the training table"
    return t


# ------------------------------------------------------------------ scoring
def macro_f1_fast(y, pred):
    """Macro F1 with plain numpy (used inside the threshold search, which tries ~200 values)."""
    tp1 = np.sum(pred & (y == 1)); fp1 = np.sum(pred & (y == 0)); fn1 = np.sum(~pred & (y == 1))
    tp0 = np.sum(~pred & (y == 0)); fp0 = fn1; fn0 = fp1
    f1 = lambda tp, fp, fn: 2 * tp / max(2 * tp + fp + fn, 1)  # noqa: E731
    return (f1(tp0, fp0, fn0) + f1(tp1, fp1, fn1)) / 2


def scores(y, pred) -> dict:
    """Every number we report for one set of predictions."""
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
        return self.model.predict_proba(t[self.features].to_numpy())[:, 1]

    def predict(self, t):
        return (self.proba(t) >= 0.5).astype(np.int8)


def make(name):
    if name == "majority":
        return Majority()
    if name == "ndvi_drop":
        return NdviDrop()
    return TreeModel("lgbm" if name.startswith("lgbm") else "rf", FEATURE_SETS[name])


# ------------------------------------------------------------------ cross-validation
def run_cv():
    t = train_table()
    print(f"training table: {len(t):,} rows ({t['date'].value_counts().to_dict()})")
    rows, oof = [], {}
    folds = list(cv_splits(t))
    counts = np.zeros(len(t), int)
    for _, _, check_idx in folds:
        counts[check_idx] += 1
    assert (counts == 1).all(), "every training row must be in exactly one check fold"
    for name in ORDER:
        t0 = time.time()
        oof[name] = np.full(len(t), -1, np.int8)            # out-of-fold predictions, same-date exam
        oof[name + "@time"] = np.full(len(t), -1, np.int8)  # time-test exam (only Jul rows filled)
        for k, fit_idx, check_idx in folds:
            assert not set(fit_idx) & set(check_idx)
            fit_t, check_t = t.iloc[fit_idx], t.iloc[check_idx]
            # exam 1: same dates
            m = make(name).fit(fit_t)
            pred = m.predict(check_t)
            oof[name][check_idx] = pred
            for d in LABELLED:
                sel = (check_t["date"] == d).to_numpy()
                rows.append({"model": name, "exam": f"same-date {d}", "fold": k,
                             **scores(check_t["label"].to_numpy()[sel], pred[sel])})
            # exam 2: time test (fit Dec only -> check Jul)
            fit_dec = fit_t[fit_t["date"] == TIME_FIT]
            chk_jul_idx = check_idx[(check_t["date"] == TIME_CHECK).to_numpy()]
            chk_jul = t.iloc[chk_jul_idx]
            m2 = make(name).fit(fit_dec)
            pred2 = m2.predict(chk_jul)
            oof[name + "@time"][chk_jul_idx] = pred2
            rows.append({"model": name, "exam": f"time {TIME_FIT}->{TIME_CHECK}", "fold": k,
                         **scores(chk_jul["label"].to_numpy(), pred2)})
        # every training row checked exactly once (same-date); every Jul row exactly once (time test)
        assert (oof[name] >= 0).all(), f"{name}: some rows never checked"
        assert ((oof[name + "@time"] >= 0) == (t["date"] == TIME_CHECK).to_numpy()).all()
        print(f"  {name}: {time.time() - t0:.0f}s")
    return t, pd.DataFrame(rows), oof


def summarise(t, per_fold, oof):
    """Mean +/- spread over the 5 folds, plus pooled out-of-fold confusion matrices."""
    metrics = ["macro_f1", "balanced_accuracy", "accuracy", "precision_intact", "recall_intact", "f1_intact",
               "precision_damaged", "recall_damaged", "f1_damaged"]
    agg = per_fold.groupby(["exam", "model"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["model"] = pd.Categorical(agg["model"], ORDER)
    agg = agg.sort_values(["exam", "model"]).round(4)
    pooled = {}
    y = t["label"].to_numpy()
    for name in ORDER:
        for d in LABELLED:
            sel = (t["date"] == d).to_numpy()
            pooled[f"{name} | same-date {d}"] = scores(y[sel], oof[name][sel])
        sel = oof[name + "@time"] >= 0
        pooled[f"{name} | time {TIME_FIT}->{TIME_CHECK}"] = scores(y[sel], oof[name + "@time"][sel])
    return agg, pooled


def paired_gaps(per_fold, model="rf_all", baseline="ndvi_drop"):
    """Model minus baseline, fold by fold (same check pixels), so we can say 'better in every fold' or not."""
    a = per_fold.pivot_table(index=["exam", "fold"], columns="model", values="macro_f1")
    g = (a[model] - a[baseline]).rename("gap").reset_index()
    return g.groupby("exam")["gap"].agg(["mean", "min", "max", lambda v: int((v > 0).sum())])             .rename(columns={"<lambda_0>": "folds_model_better"}).reset_index().round(4)


def figure(agg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    exams = list(dict.fromkeys(agg["exam"]))
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3 (validated palette)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    h = 0.8 / len(exams)
    for i, ex in enumerate(exams):
        a = agg[agg["exam"] == ex].set_index("model").loc[ORDER]
        y = np.arange(len(ORDER)) + i * h
        ax.barh(y, a["macro_f1_mean"], height=h * 0.9, xerr=a["macro_f1_std"], color=colors[i],
                label=ex, error_kw={"elinewidth": 1, "ecolor": "#52514e", "capsize": 0})
        for yy, v, sd in zip(y, a["macro_f1_mean"], a["macro_f1_std"]):
            ax.text(v + sd + 0.012, yy, f"{v:.2f}", va="center", fontsize=8, color="#52514e")
    ax.set_yticks(np.arange(len(ORDER)) + h * (len(exams) - 1) / 2)
    ax.set_yticklabels(ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro F1 (mean of 5 cross-validation folds; line = spread)")
    ax.set_title("Training land only: how well each method matches the UN damage map", loc="left", fontsize=11)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=len(exams), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "cv_scores.png", dpi=140)


def cmd_cv():
    t, per_fold, oof = run_cv()
    agg, pooled = summarise(t, per_fold, oof)
    per_fold.drop(columns="cm").to_csv(RESULTS / "cv_scores_per_fold.csv", index=False)
    agg.to_csv(RESULTS / "cv_scores.csv", index=False)
    (RESULTS / "cv_pooled.json").write_text(json.dumps(pooled, indent=1))
    (PROCESSED / "cv").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROCESSED / "cv" / "oof.npz", row=t["row"].to_numpy(), col=t["col"].to_numpy(),
                        date=t["date"].to_numpy(), label=t["label"].to_numpy(), **oof)
    figure(agg)
    gaps = paired_gaps(per_fold)
    gaps.to_csv(RESULTS / "cv_paired_gaps.csv", index=False)
    print(gaps.to_string(index=False))
    cols = ["exam", "model", "macro_f1_mean", "macro_f1_std", "recall_intact_mean", "precision_intact_mean",
            "recall_damaged_mean", "precision_damaged_mean"]
    print(agg[cols].to_string(index=False))


# ------------------------------------------------------------------ final fit (all training land)
def cmd_fit():
    import joblib
    t = train_table()
    MODELS_DIR.mkdir(exist_ok=True)
    info = {}
    for name in ORDER:
        for tag, data in (("", t), ("@time", t[t["date"] == TIME_FIT])):
            t0 = time.time()
            m = make(name).fit(data)
            joblib.dump(m, MODELS_DIR / f"{name}{tag}.joblib", compress=3)
            info[name + tag] = {"fitted_on": sorted(data["date"].unique().tolist()), "n_rows": int(len(data)),
                                "seconds": round(time.time() - t0, 1)}
            if name == "ndvi_drop":
                info[name + tag]["thresholds"] = m.threshold
            print(f"  {name}{tag}: fitted on {len(data):,} rows in {time.time() - t0:.0f}s")
    # which features the main forest leaned on (impurity importance, from training land only)
    rf = joblib.load(MODELS_DIR / "rf_all.joblib")
    imp = pd.Series(rf.model.feature_importances_, index=rf.features).sort_values(ascending=False).round(4)
    imp.rename("importance").to_csv(RESULTS / "feature_importance_rf_all.csv", index_label="feature")
    (RESULTS / "final_models.json").write_text(json.dumps(info, indent=2))
    print(imp.head(10).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cv", "fit", "figure"])
    {"cv": cmd_cv, "fit": cmd_fit,
     "figure": lambda: figure(pd.read_csv(RESULTS / "cv_scores.csv"))}[ap.parse_args().step]()
