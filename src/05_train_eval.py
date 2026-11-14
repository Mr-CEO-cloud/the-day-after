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
            x, y = g["d_ndvi"].to_numpy(), g["label"].to_numpy()
            cands = np.unique(np.quantile(x, np.linspace(0.005, 0.995, 199)))
            best = max(cands, key=lambda c: macro_f1_fast(y, x < c))
            self.threshold[d] = float(best)
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
        for yy, v in zip(y, a["macro_f1_mean"]):
            ax.text(v + 0.012, yy, f"{v:.2f}", va="center", fontsize=8, color="#52514e")
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
    ax.legend(loc="lower right", fontsize=8, frameon=False)
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
    ap.add_argument("step", choices=["cv", "fit"])
    {"cv": cmd_cv, "fit": cmd_fit}[ap.parse_args().step]()
