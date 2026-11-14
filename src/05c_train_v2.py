"""
05c_train_v2.py - model v2, step 4: build the improved model on TRAINING land only.

Plain words:
- The test blocks were opened once for v1. So v2 is designed, compared and chosen with
  cross-validation on training land only (the same 5 buffered folds as v1). The test
  blocks are not read anywhere in this file.
- We add one change at a time, so we can say what each change bought (an "ablation"):
    hist_2m        season-history features, trained on the same 2 months as v1
    hist_18m       the same, trained on all 18 UN monthly maps
    hist_18m_w     + clean pixels count more in training
    hist_18m_w_nb  + neighbourhood averages
    rf_18m_w       RandomForest in place of LightGBM
    lgbm_small / lgbm_big / lgbm_slow   three other LightGBM settings
- Rivals scored on the same pixels:
    rule_now   "damaged if this month's greenness is far below normal" (like v1's rule)
    rule_war   "damaged if the WORST month since Oct 2023 was far below normal" (close to the UN method)
    majority   always the most common answer
- Exams:
    every month   each of the 18 months, in the held-out fold
    headline      Dec 2024 and Jul 2025 (the v1 exams, same pixels)
    forward       learn from maps up to Dec 2024, check 2025 months (new place AND later time)
    unseen season learn without any March-May map, check April 2025
- The selection rule was fixed in the plan before any of this ran: highest mean macro F1
  over Dec 2024 + Jul 2025; within 0.005 the simpler candidate wins; v2 must beat v1 by
  more than v1's fold-to-fold spread or v1 stays.

Run:
  .venv/Scripts/python.exe -I src/05c_train_v2.py cv        # all candidates          -> results/v2/cv_*.csv
  .venv/Scripts/python.exe -I src/05c_train_v2.py choose    # apply the selection rule -> results/v2/chosen.json
  .venv/Scripts/python.exe -I src/05c_train_v2.py exams     # forward + unseen-season exams for the chosen one
  .venv/Scripts/python.exe -I src/05c_train_v2.py fit       # final models on all training land -> models/v2_*.joblib
  .venv/Scripts/python.exe -I src/05c_train_v2.py rehearse  # the full exam analysis on training land -> results/v2/cv_report/
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (PROCESSED, RESULTS, SEED, N_CV_FOLDS, LABEL_MONTHS, FEATURES_V2, V2_NEIGHBOUR)  # noqa: E402
from common import load_table  # noqa: E402
from methods import MODELS_DIR, LABELLED, best_threshold, scores  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = RESULTS / "v2"
CVDIR = PROCESSED / "cv_v2"
ALL_COLS = FEATURES_V2 + V2_NEIGHBOUR
HEADLINE = LABELLED                              # ["2024-12", "2025-07"]: the v1 exam months
FIT_FRACTION = 0.25                              # 18-month candidates learn from a random quarter of the pixel-months
FORWARD_LAST = "2024-12"                         # forward exam: learn from maps up to here ...
FORWARD_CHECK = ["2025-02", "2025-03", "2025-04", "2025-07"]      # ... and check these later months
SPRING = ["2024-03", "2024-04", "2024-05", "2025-03", "2025-04"]  # left out in the unseen-season exam
SEASON_CHECK = "2025-04"
TIE = 0.005
V1_FOLD_SPREAD = 0.03                            # v1's fold-to-fold spread (results/cv_scores.csv: 0.024-0.029)

LGBM = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100, subsample=0.5,
            subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", random_state=SEED, n_jobs=-1, verbose=-1)
# name: (features, training months, clean-pixel weights?, model kind, settings changed from LGBM)   -- simplest first
CANDIDATES = {
    "hist_2m":       (FEATURES_V2, HEADLINE,     False, "lgbm", {}),
    "hist_18m":      (FEATURES_V2, LABEL_MONTHS, False, "lgbm", {}),
    "hist_18m_w":    (FEATURES_V2, LABEL_MONTHS, True,  "lgbm", {}),
    "hist_18m_w_nb": (ALL_COLS,    LABEL_MONTHS, True,  "lgbm", {}),
    "rf_18m_w":      (FEATURES_V2, LABEL_MONTHS, True,  "rf",   {}),
    "lgbm_small":    (FEATURES_V2, LABEL_MONTHS, True,  "lgbm", {"num_leaves": 31, "min_child_samples": 300}),
    "lgbm_big":      (FEATURES_V2, LABEL_MONTHS, True,  "lgbm", {"num_leaves": 127, "min_child_samples": 50}),
    "lgbm_slow":     (FEATURES_V2, LABEL_MONTHS, True,  "lgbm", {"n_estimators": 800, "learning_rate": 0.025}),
}


MODEL_PREFIX, RF_TREES = "v2_", 200
CAND = None
if "--candidate" in sys.argv:           # name a candidate explicitly (see cmd_choose for how a final choice is recorded)
    i = sys.argv.index("--candidate")
    CAND = sys.argv[i + 1]
    del sys.argv[i:i + 2]
if "--quick" in sys.argv:
    # A fast dry run that only checks the code works end to end (4 months, tiny models, separate folders).
    # Its numbers mean nothing and are never reported.
    sys.argv.remove("--quick")
    LABEL_MONTHS[:] = ["2024-10", "2024-12", "2025-04", "2025-07"]
    OUT, CVDIR, MODEL_PREFIX, RF_TREES, FIT_FRACTION = RESULTS / "v2_quick", PROCESSED / "cv_v2_quick", "v2quick_", 20, 0.1
    LGBM.update(n_estimators=40)
    CANDIDATES = {k: CANDIDATES[k] for k in ("hist_2m", "hist_18m_w", "hist_18m_w_nb", "rf_18m_w")}
    FORWARD_CHECK = [m for m in FORWARD_CHECK if m in LABEL_MONTHS]


# ------------------------------------------------------------------ data
class Train:
    """All training-land rows of the 18 label months, as plain arrays (4.6 million rows)."""
    def __init__(self):
        parts = []
        for m in LABEL_MONTHS:
            t = load_table(m, "train", version="v2")        # training blocks only; the test lock is untouched
            t["month"] = m
            parts.append(t)
        t = pd.concat(parts, ignore_index=True)
        self.X = t[ALL_COLS].to_numpy(np.float32)
        self.y = t["label"].to_numpy(np.int8)
        self.month = t["month"].to_numpy()
        self.row, self.col = t["row"].to_numpy(), t["col"].to_numpy()
        self.block, self.plot = t["block_id"].to_numpy(), t["plot_id"].to_numpy()
        self.fold, self.near = t["cv_fold"].to_numpy(), t["cv_near"].to_numpy()
        crop, dmg = t["crop_cells"].to_numpy(np.float32), t["dmg_cells"].to_numpy(np.float32)
        share = dmg / np.maximum(crop, 1)
        self.clean = (crop == 16) & ((dmg == 0) | (dmg == 16))          # fully cropland and fully one UN class
        self.weight = np.abs(2 * share - 1) * crop / 16 + 0.05           # clean pixels ~1, half-and-half pixels ~0.05
        self.fit_ok = np.random.default_rng(SEED).random(len(t)) < FIT_FRACTION
        self.n = len(t)
        self.col_index = {c: i for i, c in enumerate(ALL_COLS)}

    def cols(self, names):
        return [self.col_index[c] for c in names]

    def fit_rows(self, k, months, subsample=True):
        """Rows a model may learn from when fold k is held out: other folds, outside the 100 m buffer of fold k."""
        m = (self.fold != k) & ((self.near >> k) & 1 == 0) & np.isin(self.month, months)
        if subsample and len(months) > len(HEADLINE):
            m &= self.fit_ok
        return np.flatnonzero(m)


def new_model(kind, changes):
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=RF_TREES, min_samples_leaf=20, max_features="sqrt", max_samples=0.25,
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    import lightgbm as lgb
    return lgb.LGBMClassifier(**{**LGBM, **changes})


def out_of_fold(D: Train, name, months=None) -> np.ndarray:
    """Each row's "damaged" score from a model that never saw its block. NaN where a row was not scored."""
    feats, cand_months, weighted, kind, changes = CANDIDATES[name]
    months = cand_months if months is None else months
    ci = D.cols(feats)
    oof = np.full(D.n, np.nan, np.float32)
    for k in range(N_CV_FOLDS):
        fit = D.fit_rows(k, months)
        chk = np.flatnonzero(D.fold == k)
        assert not set(D.block[fit]) & set(D.block[chk])             # folds are whole blocks
        model = new_model(kind, changes)
        model.fit(D.X[fit][:, ci], D.y[fit], sample_weight=D.weight[fit] if weighted else None)
        oof[chk] = model.predict_proba(D.X[chk][:, ci])[:, 1]
    return oof


def rule_out_of_fold(D: Train, feature, months_fit=None) -> np.ndarray:
    """A one-number rule: damaged if `feature` < threshold. One threshold per month, chosen on the fit folds.
    With months_fit (forward exam), the threshold of the LAST allowed month is used for later months."""
    x = np.nan_to_num(D.X[:, D.col_index[feature]], nan=0.0)        # no clear view = "normal"
    pred = np.full(D.n, np.nan, np.float32)
    for k in range(N_CV_FOLDS):
        chk = D.fold == k
        for m in LABEL_MONTHS:
            src = m if months_fit is None or m in months_fit else months_fit[-1]
            fit = D.fit_rows(k, [src], subsample=False)
            thr = best_threshold(x[fit], D.y[fit])
            sel = chk & (D.month == m)
            pred[sel] = x[sel] < thr
    return pred


def majority_out_of_fold(D: Train) -> np.ndarray:
    pred = np.full(D.n, np.nan, np.float32)
    for k in range(N_CV_FOLDS):
        for m in LABEL_MONTHS:
            fit = D.fit_rows(k, [m], subsample=False)
            pred[(D.fold == k) & (D.month == m)] = float(D.y[fit].mean() >= 0.5)
    return pred


# ------------------------------------------------------------------ scoring
def month_scores(D: Train, score, name, threshold=0.5, months=LABEL_MONTHS) -> list:
    rows = []
    pred = (score >= threshold).astype(np.int8)
    for m in months:
        sel = (D.month == m) & np.isfinite(score)
        if not sel.any():
            continue
        s = scores(D.y[sel], pred[sel])
        folds = [scores(D.y[sel & (D.fold == k)], pred[sel & (D.fold == k)])["macro_f1"] for k in range(N_CV_FOLDS)]
        c = sel & D.clean
        rows.append({"candidate": name, "month": m, "macro_f1": s["macro_f1"], "fold_mean": float(np.mean(folds)),
                     "fold_std": float(np.std(folds, ddof=1)), "macro_f1_clean_pixels": scores(D.y[c], pred[c])["macro_f1"],
                     "accuracy": s["accuracy"], "precision_intact": s["precision_intact"], "recall_intact": s["recall_intact"],
                     "precision_damaged": s["precision_damaged"], "recall_damaged": s["recall_damaged"], "n": s["n"]})
    return rows


def v1_reference(D: Train) -> list:
    """v1's own cross-validation answers (saved in Phase 3), scored on the same rows, for a like-for-like comparison."""
    z = np.load(PROCESSED / "cv" / "oof.npz", allow_pickle=True)
    v1 = pd.DataFrame({"row": z["row"], "col": z["col"], "month": z["date"], "p": z["rf_all#p"], "rule": z["ndvi_drop"]})
    me = pd.DataFrame({"row": D.row, "col": D.col, "month": D.month}).merge(v1, how="left", on=["row", "col", "month"])
    rows = month_scores(D, me["p"].to_numpy(np.float32), "v1_rf_all", months=HEADLINE)
    rows += month_scores(D, me["rule"].to_numpy(np.float32), "v1_rule_ndvi_drop", months=HEADLINE)
    return rows


# ------------------------------------------------------------------ steps
def cmd_cv():
    t0 = time.time()
    D = Train()
    print(f"training rows: {D.n:,} ({len(LABEL_MONTHS)} months), features: {len(FEATURES_V2)} core + {len(V2_NEIGHBOUR)} neighbourhood, "
          f"clean pixels {D.clean.mean():.1%}, loaded in {time.time() - t0:.0f}s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    CVDIR.mkdir(parents=True, exist_ok=True)
    np.savez(CVDIR / "meta.npz", row=D.row, col=D.col, month=D.month, label=D.y, fold=D.fold, block_id=D.block,
             plot_id=D.plot, clean=D.clean)
    rows = v1_reference(D)
    rows += month_scores(D, majority_out_of_fold(D), "majority")
    rows += month_scores(D, rule_out_of_fold(D, "an_ndvi"), "rule_now")
    rule_war = rule_out_of_fold(D, "war_ndvi_min")
    np.save(CVDIR / "oof_rule_war.npy", rule_war.astype(np.float16))
    rows += month_scores(D, rule_war, "rule_war")
    print(f"rules done at {time.time() - t0:.0f}s", flush=True)
    for name in CANDIDATES:
        t1 = time.time()
        oof = out_of_fold(D, name)
        assert np.isfinite(oof).all(), f"{name}: some rows never scored"
        np.save(CVDIR / f"oof_{name}.npy", oof.astype(np.float16))
        rows += month_scores(D, oof, name)
        h = [r for r in rows if r["candidate"] == name and r["month"] in HEADLINE]
        print(f"  {name:14s} " + " | ".join(f"{r['month']} F1 {r['macro_f1']:.3f} (folds {r["fold_mean"]:.3f} +- {r["fold_std"]:.3f}, "
              f"clean px {r['macro_f1_clean_pixels']:.3f})" for r in h) + f"  [{time.time() - t1:.0f}s]", flush=True)
        pd.DataFrame(rows).round(4).to_csv(OUT / "cv_by_month.csv", index=False)      # saved after every candidate
    summarise(pd.DataFrame(rows))


def summarise(by_month: pd.DataFrame) -> pd.DataFrame:
    """One line per candidate: the headline months, their mean, and the average over all 18 months."""
    h = by_month[by_month["month"].isin(HEADLINE)].pivot(index="candidate", columns="month", values="macro_f1")
    c = by_month[by_month["month"].isin(HEADLINE)].pivot(index="candidate", columns="month", values="macro_f1_clean_pixels")
    S = pd.DataFrame({f"f1_{m}": h[m] for m in HEADLINE})
    S["headline_mean"] = h.mean(axis=1)
    S["clean_pixels_mean"] = c.mean(axis=1)
    S["all_18_months_mean"] = by_month.groupby("candidate")["macro_f1"].mean().where(
        by_month.groupby("candidate")["month"].nunique() == len(LABEL_MONTHS))
    order = ["majority", "v1_rule_ndvi_drop", "rule_now", "rule_war", "v1_rf_all"] + list(CANDIDATES)
    S = S.reindex([o for o in order if o in S.index]).round(4)
    S.to_csv(OUT / "cv_candidates.csv")
    print(S.to_string())
    return S


SIDE = ["hist_2m"]      # also carried to the test as a declared side result: the two-month specialist


def decide_threshold(name):
    """The score above which a pixel is called damaged: 0.5, or one cut tuned on the OTHER folds'
    out-of-fold scores. The tuned cut is kept only if it helps by more than the tie band (0.005)."""
    meta = np.load(CVDIR / "meta.npz", allow_pickle=True)
    oof = np.load(CVDIR / f"oof_{name}.npy").astype(np.float32)
    y, fold, month = meta["label"], meta["fold"], meta["month"]
    head = np.isin(month, HEADLINE)
    tuned = np.zeros(len(y), np.int8)
    for k in range(N_CV_FOLDS):
        thr = -best_threshold(-oof[(fold != k) & head], y[(fold != k) & head])           # damaged if score > thr
        tuned[fold == k] = oof[fold == k] > thr
    f_half = float(np.mean([scores(y[month == m], (oof[month == m] >= 0.5).astype(np.int8))["macro_f1"] for m in HEADLINE]))
    f_tuned = float(np.mean([scores(y[month == m], tuned[month == m])["macro_f1"] for m in HEADLINE]))
    threshold = float(-best_threshold(-oof[head], y[head])) if f_tuned - f_half > TIE else 0.5
    return threshold, f_half, f_tuned


def cmd_choose():
    by_month = pd.read_csv(OUT / "cv_by_month.csv")
    S = summarise(by_month)
    cand = S.loc[[c for c in CANDIDATES if c in S.index], "headline_mean"]
    best = cand.max()
    rule_pick = next(c for c in CANDIDATES if c in cand.index and cand[c] >= best - TIE)   # simplest within the tie band
    chosen = CAND or rule_pick      # --candidate records a reasoned departure from the rule (explained in docs/test_plan_v2.md)
    v1 = float(S.loc["v1_rf_all", "headline_mean"])
    beats_v1 = bool(cand[chosen] - v1 > V1_FOLD_SPREAD)
    threshold, f_half, f_tuned = decide_threshold(chosen)
    out = {"rule":"highest mean macro F1 over Dec 2024 + Jul 2025 in cross-validation; within 0.005 the simpler candidate; "
                   "must beat v1 by more than v1's fold-to-fold spread (0.03)",
           "candidates_headline_mean": {c: float(v) for c, v in cand.items()}, "best_score": float(best),
           "rule_pick": rule_pick, "rule_pick_score": float(cand[rule_pick]), "departed_from_rule": bool(chosen != rule_pick),
           "all_18_months_mean": {c: float(S.loc[c, "all_18_months_mean"]) for c in cand.index},
           "chosen": chosen, "chosen_score": float(cand[chosen]), "v1_rf_all_score": v1,
           "v2_replaces_v1": beats_v1, "threshold_0.5_score": float(f_half), "threshold_tuned_score": float(f_tuned),
           "decision_threshold": threshold, "features": CANDIDATES[chosen][0], "training_months": CANDIDATES[chosen][1],
           "clean_pixel_weights": CANDIDATES[chosen][2], "model": CANDIDATES[chosen][3], "settings_changed": CANDIDATES[chosen][4]}
    (OUT / "chosen.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("features", "training_months")}, indent=1))


def cmd_exams():
    """Two harder exams for the chosen candidate (and the war rule): forward in time, and an unseen season."""
    chosen = CAND or json.loads((OUT / "chosen.json").read_text())["chosen"]
    D = Train()
    rows = []
    own = CANDIDATES[chosen][1]                                    # the months this candidate learns from
    upto = [m for m in LABEL_MONTHS if m <= FORWARD_LAST]
    fwd = out_of_fold(D, chosen, months=[m for m in own if m <= FORWARD_LAST])
    np.save(CVDIR / f"oof_{chosen}@forward.npy", fwd.astype(np.float16))
    rows += [dict(r, exam="forward: learn up to Dec 2024") for r in month_scores(D, fwd, chosen, months=FORWARD_CHECK)]
    rule_fwd = rule_out_of_fold(D, "war_ndvi_min", months_fit=upto)
    np.save(CVDIR / "oof_rule_war@forward.npy", rule_fwd.astype(np.float16))
    rows += [dict(r, exam="forward: learn up to Dec 2024") for r in month_scores(D, rule_fwd, "rule_war", months=FORWARD_CHECK)]
    nospring = out_of_fold(D, chosen, months=[m for m in own if m not in SPRING])
    rows += [dict(r, exam="unseen season: learn without Mar-May") for r in month_scores(D, nospring, chosen, months=[SEASON_CHECK])]
    full = np.load(CVDIR / f"oof_{chosen}.npy").astype(np.float32)
    rows += [dict(r, exam="same months seen (for comparison)") for r in month_scores(D, full, chosen, months=FORWARD_CHECK)]
    T = pd.DataFrame(rows).round(4)
    T.to_csv(OUT / f"cv_hard_exams_{chosen}.csv", index=False)
    print(T[["exam", "candidate", "month", "macro_f1", "fold_mean", "fold_std", "macro_f1_clean_pixels"]].to_string(index=False))


def cmd_fit():
    """Final v2 models, fitted on ALL training land: the chosen model, its forward-exam twin, the calibration
    curve (from out-of-fold scores) and the rule thresholds. File fingerprints are saved for the test step."""
    import hashlib
    import joblib
    from sklearn.isotonic import IsotonicRegression
    ch = json.loads((OUT / "chosen.json").read_text())
    name = ch["chosen"]
    feats, months, weighted, kind, changes = CANDIDATES[name]
    D = Train()
    ci = D.cols(feats)
    MODELS_DIR.mkdir(exist_ok=True)
    for old in MODELS_DIR.glob(f"{MODEL_PREFIX}*.joblib"):         # start clean: only this run's files get fingerprinted
        old.unlink()
    info = {"chosen": name, "decision_threshold": ch["decision_threshold"], "features": feats, "side_models": {}}

    def fit_pair(cand, threshold):
        """Fit one candidate on all training land, plus its forward-exam twin (maps up to Dec 2024 only)."""
        f, months_c, weighted_c, kind_c, changes_c = CANDIDATES[cand]
        cols, last = D.cols(f), None
        for tag, mm in (("", months_c), ("@forward", [m for m in months_c if m <= FORWARD_LAST])):
            fit = np.flatnonzero(np.isin(D.month, mm) & (D.fit_ok if len(mm) > len(HEADLINE) else True))
            last = new_model(kind_c, changes_c).fit(D.X[fit][:, cols], D.y[fit],
                                                    sample_weight=D.weight[fit] if weighted_c else None)
            joblib.dump({"model": last, "features": f, "threshold": threshold, "months": mm},
                        MODELS_DIR / f"{MODEL_PREFIX}{cand}{tag}.joblib", compress=3)
            print(f"fitted {MODEL_PREFIX}{cand}{tag} on {len(fit):,} rows ({len(mm)} months)", flush=True)
        return last

    for side in [c for c in SIDE if c != name and c in CANDIDATES]:
        thr_side = decide_threshold(side)[0]
        fit_pair(side, thr_side)
        info["side_models"][side] = {"decision_threshold": thr_side, "training_months": CANDIDATES[side][1]}
    model = fit_pair(name, ch["decision_threshold"])              # the chosen model last, so `model` is the main one
    info["training_months"] = months
    oof = np.load(CVDIR / f"oof_{name}.npy").astype(np.float32)
    curve = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(oof, D.y)
    joblib.dump(curve, MODELS_DIR / f"{MODEL_PREFIX}{name}_calibrator.joblib")
    rules = {}
    for feature, rule in (("an_ndvi", "rule_now"), ("war_ndvi_min", "rule_war")):
        x = np.nan_to_num(D.X[:, D.col_index[feature]], nan=0.0)
        rules[rule] = {"feature": feature, "thresholds": {m: best_threshold(x[D.month == m], D.y[D.month == m]) for m in LABEL_MONTHS}}
    (OUT / "rule_thresholds.json").write_text(json.dumps(rules, indent=1))
    info["majority"] = {m: int(D.y[D.month == m].mean() >= 0.5) for m in LABEL_MONTHS}
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
        (imp / imp.sum()).round(4).rename("importance").to_csv(OUT / "feature_importance.csv", index_label="feature")
        print((imp / imp.sum()).round(3).head(12).to_string())
    info["file_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(MODELS_DIR.glob(f"{MODEL_PREFIX}*.joblib"))}
    (OUT / "final_models.json").write_text(json.dumps(info, indent=1))


def cmd_rehearse():
    """Run the complete exam analysis (evaluation.py) for v2 on TRAINING land, with out-of-fold answers,
    next to v1's out-of-fold answers on the same pixels. This is the dress rehearsal of the second test look."""
    import rasterio
    from sklearn.isotonic import IsotonicRegression
    from methods import stacked_table, ORDER, V2_ORDER, V2_MAIN, V2_BASELINE, V2_GAPS, V2_SIDE, TIME_CHECK
    from evaluation import full_report
    ch = json.loads((OUT / "chosen.json").read_text())
    name, thr = ch["chosen"], ch["decision_threshold"]
    keep = ["post_ndvi", "pre_ndvi", "d_ndvi", "d_bsi", "prewar_peak_mean"]
    t = stacked_table("train")                                           # the v1 training table (2 months)
    z = np.load(PROCESSED / "cv" / "oof.npz", allow_pickle=True)         # v1 out-of-fold answers, same row order
    assert (z["row"] == t["row"]).all() and (z["col"] == t["col"]).all() and (z["date"] == t["date"]).all()
    P = t[["row", "col", "date", "label", "block_id", "plot_id", "cv_fold"] + keep].copy()
    for m in ORDER:
        P["pred__" + m], P["time__" + m] = z[m], z[m + "@time"]
    curve = lambda: IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")  # noqa: E731
    v1_raw = z[MAIN_V1 + "#p"]
    P["cal_v1"] = np.nan
    for k in range(N_CV_FOLDS):
        a, b = (P["cv_fold"] != k).to_numpy(), (P["cv_fold"] == k).to_numpy()
        P.loc[b, "cal_v1"] = curve().fit(v1_raw[a], P["label"].to_numpy()[a]).predict(v1_raw[b])
    # v2 out-of-fold answers (all 18 months) -> keep the two headline months, matched pixel by pixel
    meta = np.load(CVDIR / "meta.npz", allow_pickle=True)
    V = pd.DataFrame({"row": meta["row"], "col": meta["col"], "date": meta["month"], "fold": meta["fold"],
                      "y": meta["label"], "clean": meta["clean"],
                      "score": np.load(CVDIR / f"oof_{name}.npy").astype(np.float32),
                      "fwd": np.load(CVDIR / f"oof_{name}@forward.npy").astype(np.float32),
                      "rule": np.load(CVDIR / "oof_rule_war.npy").astype(np.float32),
                      "rule_fwd": np.load(CVDIR / "oof_rule_war@forward.npy").astype(np.float32)})
    V["cal"] = np.nan
    for k in range(N_CV_FOLDS):                                          # calibration learned on the other folds
        a, b = (V["fold"] != k).to_numpy(), (V["fold"] == k).to_numpy()
        V.loc[b, "cal"] = curve().fit(V["score"].to_numpy()[a], V["y"].to_numpy()[a]).predict(V["score"].to_numpy()[b])
    P = P.merge(V, how="left", on=["row", "col", "date"])
    assert P["score"].notna().all() and (P["label"] == P["y"]).all(), "v1 and v2 rows must match pixel by pixel"
    is_check = (P["date"] == TIME_CHECK).to_numpy()
    P["pred__v2"] = (P["score"] >= thr).astype(np.int8)
    P["pred__rule_war"] = P["rule"].astype(np.int8)
    P["time__v2"] = np.where(is_check, (P["fwd"] >= thr), -1).astype(np.int8)
    P["time__rule_war"] = np.where(is_check, P["rule_fwd"], -1).astype(np.int8)
    for side, method in V2_SIDE.items():                                 # the declared side result(s)
        if side == name:
            continue
        thr_s = decide_threshold(side)[0]
        S = pd.DataFrame({"row": meta["row"], "col": meta["col"], "date": meta["month"],
                          "s": np.load(CVDIR / f"oof_{side}.npy").astype(np.float32),
                          "sf": np.load(CVDIR / f"oof_{side}@forward.npy").astype(np.float32)})
        S = P[["row", "col", "date"]].merge(S, how="left", on=["row", "col", "date"])
        P["pred__" + method] = (S["s"].to_numpy() >= thr_s).astype(np.int8)
        P["time__" + method] = np.where(is_check, S["sf"].to_numpy() >= thr_s, -1).astype(np.int8)
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        shape = src.shape
    table, gaps = full_report(P, OUT / "cv_report", "Training land (cross-validation), v2", shape,
                              order=V2_ORDER, main=V2_MAIN, baseline=V2_BASELINE, gap_pairs=V2_GAPS)
    print(table[["exam", "method", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "macro_f1_clean_pixels"]].round(3).to_string(index=False))
    print(json.dumps(gaps, indent=1))


MAIN_V1 = "rf_all"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["cv", "choose", "exams", "fit", "rehearse"])
    {"cv": cmd_cv, "choose": cmd_choose, "exams": cmd_exams, "fit": cmd_fit,
     "rehearse": cmd_rehearse}[ap.parse_args().step]()
