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
  I train three versions to see which information helps:
    rf_ndvi      greenness only (the same information the NDVI-drop rule sees)
    rf_spectral  + all 6 bands, moisture (NDMI) and bare-soil (BSI) indices
    rf_all       + pre-war context (productivity, seasonality, WorldCover)
  plus lgbm_all, LightGBM (trees built one after another, each fixing the last
  one's mistakes) on the same features as rf_all, as a second opinion.
- Cross-validation: training blocks are dealt into 5 folds. Five times, I fit
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
  .venv/Scripts/python.exe -I src/05_train_eval.py calibrate  # calibration curve + rehearsal of the exam analysis (results/cv_report/)
  .venv/Scripts/python.exe -I src/05_train_eval.py figure # redraw results/figures/cv_scores.png from cv_scores.csv
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED, RESULTS  # noqa: E402
from common import cv_splits  # noqa: E402
from methods import (LABELLED, TIME_FIT, TIME_CHECK, MODELS_DIR, ORDER, MAIN_MODEL, MAIN_BASELINE,  # noqa: E402
                     stacked_table, scores, make)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ------------------------------------------------------------------ cross-validation
def run_cv():
    t = stacked_table("train")
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
            if hasattr(m, "proba"):       # tree models: keep the 0-1 "damaged" score too
                p = m.proba(check_t)
                oof.setdefault(name + "#p", np.full(len(t), np.nan, np.float32))[check_idx] = p
                pred = (p >= 0.5).astype(np.int8)
            else:
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


def paired_gaps(per_fold, model=MAIN_MODEL, baseline=MAIN_BASELINE):
    """Model minus baseline, fold by fold (same check pixels), so I can say 'better in every fold' or not."""
    a = per_fold.pivot_table(index=["exam", "fold"], columns="model", values="macro_f1")
    g = (a[model] - a[baseline]).rename("gap").reset_index().groupby("exam")["gap"]
    out = pd.DataFrame({"mean": g.mean(), "min": g.min(), "max": g.max(),
                        "folds_model_better": g.apply(lambda v: int((v > 0).sum()))})
    return out.reset_index().round(4)


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
    t = stacked_table("train")
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
    rf = joblib.load(MODELS_DIR / f"{MAIN_MODEL}.joblib")
    imp = pd.Series(rf.model.feature_importances_, index=rf.features).sort_values(ascending=False).round(4)
    imp.rename("importance").to_csv(RESULTS / "feature_importance_rf_all.csv", index_label="feature")
    (RESULTS / "final_models.json").write_text(json.dumps(info, indent=2))
    print(imp.head(10).to_string())


# ------------------------------------------------------------------ calibration + rehearsal of the exam analysis
KEEP = ["post_ndvi", "pre_ndvi", "d_ndvi", "d_bsi", "prewar_peak_mean"]   # features the error analysis describes


def cmd_calibrate():
    """1. Calibration: the forest's raw score is tilted towards "intact" (I made rare intact pixels count
          more). A calibration curve fixes that: for each raw score it looks up what share of training
          pixels with that score the UN really marked damaged. It is learned from out-of-fold scores only.
       2. Rehearsal: run the complete exam analysis (evaluation.py) on training land, using out-of-fold
          predictions, so the analysis code is finished BEFORE the test set is opened."""
    import joblib
    import rasterio
    from sklearn.isotonic import IsotonicRegression
    from evaluation import full_report
    t = stacked_table("train")
    z = np.load(PROCESSED / "cv" / "oof.npz", allow_pickle=True)
    assert (z["row"] == t["row"]).all() and (z["col"] == t["col"]).all() and (z["date"] == t["date"]).all()
    P = t[["row", "col", "date", "label", "block_id", "plot_id", "cv_fold"] + KEEP].copy()
    for m in ORDER:
        P["pred__" + m], P["time__" + m] = z[m], z[m + "@time"]
    P["score"] = z[MAIN_MODEL + "#p"]
    new_curve = lambda: IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")  # noqa: E731  (a rising step curve)
    # rehearsal: each fold is calibrated by a curve learned on the OTHER folds
    P["cal"] = np.nan
    for k in sorted(P["cv_fold"].unique()):
        other, this = P["cv_fold"] != k, P["cv_fold"] == k
        P.loc[this, "cal"] = new_curve().fit(P.loc[other, "score"], P.loc[other, "label"]).predict(P.loc[this, "score"])
    # the final curve, learned on all out-of-fold scores, is the one used on the test set and in the app
    curve = new_curve().fit(P["score"], P["label"])
    joblib.dump(curve, MODELS_DIR / f"{MAIN_MODEL}_calibrator.joblib")
    grid = np.linspace(0, 1, 21)
    pd.DataFrame({"raw_score": grid, "calibrated_probability": curve.predict(grid).round(4)})       .to_csv(RESULTS / "calibration_curve.csv", index=False)
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        shape = src.shape
    table, gaps = full_report(P, RESULTS / "cv_report", "Training land (cross-validation)", shape)
    print(table[["exam", "method", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high"]].round(3).to_string(index=False))
    print(json.dumps(gaps, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cv", "fit", "calibrate", "figure"])
    {"cv": cmd_cv, "fit": cmd_fit, "calibrate": cmd_calibrate,
     "figure": lambda: figure(pd.read_csv(RESULTS / "cv_scores.csv"))}[ap.parse_args().step]()
