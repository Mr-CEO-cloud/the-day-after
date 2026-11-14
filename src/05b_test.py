"""
05b_test.py - Phase 4: the real exam on the 40 locked test blocks.

Plain words:
- Until now, no model, threshold or decision has seen the test blocks.
- "unlock" opens them ONCE: it loads the finished models from models/, asks each
  one for its answers on the test pixels, and saves those answers. It refuses to
  run if any code is uncommitted, and it writes down the exact code version
  (git commit) and time, so anyone can check that nothing was changed afterwards.
  It refuses to run a second time.
- "report" turns the saved answers into scores, tables and pictures, using the
  same analysis code that was rehearsed on training land (src/evaluation.py).
  It can be re-run freely: it only reads the saved answers and never refits anything.

The rules of this exam were written down in docs/test_plan.md before unlocking.

Model v2 (season history, see 05c_train_v2.py) gets one more look, "unlock-v2", under the
same guards and with its own written rules (docs/test_plan_v2.md). That is the second and
last time the test blocks are scored.

Run:
  .venv/Scripts/python.exe -I src/05b_test.py unlock
  .venv/Scripts/python.exe -I src/05b_test.py report
  .venv/Scripts/python.exe -I src/05b_test.py unlock-v2
  .venv/Scripts/python.exe -I src/05b_test.py report-v2
"""
import sys
import json
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, PROCESSED, RESULTS  # noqa: E402
from methods import (TIME_CHECK, MODELS_DIR, ORDER, MAIN_MODEL, MAIN_BASELINE, stacked_table)  # noqa: E402
from evaluation import full_report  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

OUT = RESULTS / "test"
PRED_FILE = PROCESSED / "test" / "predictions.npz"
UNLOCK_FILE = OUT / "unlock.json"
KEEP = ["post_ndvi", "pre_ndvi", "d_ndvi", "d_bsi", "prewar_peak_mean"]   # features the error analysis describes


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def cmd_unlock():
    import joblib
    if UNLOCK_FILE.exists():
        sys.exit(f"The test set was already opened ({UNLOCK_FILE}). It is opened once. Use 'report' to re-read the saved answers.")
    dirty = git("status", "--porcelain", "--", "src", "tests", "docs/test_plan.md")
    if dirty:
        sys.exit("Commit all code and the test plan before opening the test set:\n" + dirty)
    commit = git("rev-parse", "HEAD")

    t = stacked_table("test", unlock_test=True)        # <- the only place in the project that passes the key
    P = t[["row", "col", "date", "label", "block_id", "plot_id"] + KEEP].copy()
    is_time_check = (t["date"] == TIME_CHECK).to_numpy()
    for m in ORDER:
        model = joblib.load(MODELS_DIR / f"{m}.joblib")                 # fitted on training land, both dates
        if m == MAIN_MODEL:
            P["score"] = model.proba(t).astype("float32")
            P["pred__" + m] = (P["score"] >= 0.5).astype("int8")
        else:
            P["pred__" + m] = model.predict(t)
        time_model = joblib.load(MODELS_DIR / f"{m}@time.joblib")       # fitted on training land, Dec 2024 only
        P["time__" + m] = np.int8(-1)
        P.loc[is_time_check, "time__" + m] = time_model.predict(t[is_time_check])
    curve = joblib.load(MODELS_DIR / f"{MAIN_MODEL}_calibrator.joblib")  # learned on training land
    P["cal"] = curve.predict(P["score"]).astype("float32")

    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED_FILE, **{c: P[c].to_numpy() for c in P.columns})
    OUT.mkdir(parents=True, exist_ok=True)
    # a small, readable copy of the answers, so anyone can recompute the scores without our code
    P[["date", "row", "col", "block_id", "plot_id", "label", "pred__" + MAIN_BASELINE, "pred__" + MAIN_MODEL,
       "time__" + MAIN_BASELINE, "time__" + MAIN_MODEL, "cal"]].round(4).to_csv(OUT / "predictions_main.csv.gz", index=False)
    UNLOCK_FILE.write_text(json.dumps({
        "opened_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_version_git_commit": commit, "n_rows": int(len(P)),
        "n_blocks": int(P["block_id"].nunique()), "rows_per_date": P["date"].value_counts().to_dict(),
        "main_model": MAIN_MODEL, "main_baseline": MAIN_BASELINE}, indent=2))
    print(f"test set opened: {len(P):,} rows, {P['block_id'].nunique()} blocks, code version {commit[:7]}")


def cmd_report():
    if not UNLOCK_FILE.exists():
        sys.exit("Run 'unlock' first.")
    with np.load(PRED_FILE, allow_pickle=True) as z:
        P = pd.DataFrame({c: z[c] for c in z.files})
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        shape = src.shape
        assert (src.read(1)[P["row"], P["col"]] == 2).all()             # every row is test land
    table, gaps = full_report(P, OUT, "Test land (40 unseen blocks)", shape)
    cols = ["exam", "method", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "accuracy",
            "precision_intact", "recall_intact", "precision_damaged", "recall_damaged"]
    print(table[cols].round(3).to_string(index=False))
    print(json.dumps(gaps, indent=1))


# ====================================================================== model v2: the second look
OUT_V2 = RESULTS / "test_v2"
PRED_V2 = PROCESSED / "test_v2" / "predictions.npz"
UNLOCK_V2 = OUT_V2 / "unlock.json"


def cmd_unlock_v2():
    """The SECOND and last look at the test blocks, for model v2.
    v2 was designed and chosen on training land only (05c_train_v2.py); its rules are in
    docs/test_plan_v2.md, which must be committed before this runs. Same guards as v1:
    clean git, the code version is recorded, and it refuses to run twice. In addition the
    saved model files must match the fingerprints recorded when they were fitted."""
    import hashlib
    import joblib
    from common import load_table
    from methods import LABELLED, TIME_FIT
    if not UNLOCK_FILE.exists():
        sys.exit("v1 must be tested first.")
    if UNLOCK_V2.exists():
        sys.exit(f"The v2 look at the test set was already taken ({UNLOCK_V2}). Use 'report-v2'.")
    if not (ROOT / "docs" / "test_plan_v2.md").exists():
        sys.exit("Write and commit docs/test_plan_v2.md first.")
    dirty = git("status", "--porcelain", "--", "src", "tests", "docs/test_plan_v2.md", "results/v2")
    if dirty:
        sys.exit("Commit all code, the v2 test plan and the v2 cross-validation results first:\n" + dirty)
    commit = git("rev-parse", "HEAD")
    info = json.loads((RESULTS / "v2" / "final_models.json").read_text())
    name = info["chosen"]
    for fname, digest in info["file_sha256"].items():          # the files are the ones that were fitted
        assert hashlib.sha256((MODELS_DIR / fname).read_bytes()).hexdigest() == digest, f"{fname} changed since it was fitted"
    rules = json.loads((RESULTS / "v2" / "rule_thresholds.json").read_text())["rule_war"]

    with np.load(PRED_FILE, allow_pickle=True) as z:                    # the v1 answers saved at the first look
        P = pd.DataFrame({c: z[c] for c in z.files})
    P = P.rename(columns={"cal": "cal_v1", "score": "score_v1"})
    parts = []
    for d in LABELLED:
        t = load_table(d, "test", unlock_test=True, version="v2")       # <- the only v2 use of the key
        t["date"] = d
        parts.append(t)
    T = pd.concat(parts, ignore_index=True)
    T = P[["row", "col", "date", "label"]].merge(T, how="left", on=["row", "col", "date"], suffixes=("", "_v2"))
    assert T["label_v2"].notna().all() and (T["label"] == T["label_v2"]).all(), "v1 and v2 must be scored on the same pixels and labels"
    is_check = (T["date"] == TIME_CHECK).to_numpy()

    bundle = joblib.load(MODELS_DIR / f"v2_{name}.joblib")              # fitted on training land, all its months
    forward = joblib.load(MODELS_DIR / f"v2_{name}@forward.joblib")     # fitted on training land, maps up to Dec 2024
    thr = bundle["threshold"]
    P["score"] = bundle["model"].predict_proba(T[bundle["features"]].to_numpy(np.float32))[:, 1].astype("float32")
    P["pred__v2"] = (P["score"] >= thr).astype("int8")
    P["time__v2"] = np.int8(-1)
    P.loc[is_check, "time__v2"] = (forward["model"].predict_proba(
        T.loc[is_check, forward["features"]].to_numpy(np.float32))[:, 1] >= thr).astype("int8")
    from methods import V2_SIDE
    for side, method in V2_SIDE.items():                                # declared side result(s), same treatment
        if side not in info["side_models"]:
            continue
        b, fw = joblib.load(MODELS_DIR / f"v2_{side}.joblib"), joblib.load(MODELS_DIR / f"v2_{side}@forward.joblib")
        P["pred__" + method] = (b["model"].predict_proba(T[b["features"]].to_numpy(np.float32))[:, 1] >= b["threshold"]).astype("int8")
        P["time__" + method] = np.int8(-1)
        P.loc[is_check, "time__" + method] = (fw["model"].predict_proba(
            T.loc[is_check, fw["features"]].to_numpy(np.float32))[:, 1] >= b["threshold"]).astype("int8")
    x = np.nan_to_num(T[rules["feature"]].to_numpy(), nan=0.0)          # the stronger rule: worst month since Oct 2023
    P["pred__rule_war"] = (x < T["date"].map(rules["thresholds"]).to_numpy()).astype("int8")
    P["time__rule_war"] = np.int8(-1)
    P.loc[is_check, "time__rule_war"] = (x[is_check] < rules["thresholds"][TIME_FIT]).astype("int8")
    P["cal"] = joblib.load(MODELS_DIR / f"v2_{name}_calibrator.joblib").predict(P["score"]).astype("float32")
    crop, dmg = T["crop_cells"].to_numpy(), T["dmg_cells"].to_numpy()
    P["clean"] = (crop == 16) & ((dmg == 0) | (dmg == 16))              # fully cropland and fully one UN class

    PRED_V2.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED_V2, **{c: P[c].to_numpy() for c in P.columns})
    OUT_V2.mkdir(parents=True, exist_ok=True)
    side_cols = [c for c in P.columns if c.endswith("__v2_2m")]
    P[["date", "row", "col", "block_id", "plot_id", "label", "clean", "pred__rule_war", "pred__rf_all", "pred__v2",
       "time__rule_war", "time__rf_all", "time__v2", "cal_v1", "cal"] + side_cols].round(4).to_csv(
        OUT_V2 / "predictions_main.csv.gz", index=False)
    UNLOCK_V2.write_text(json.dumps({
        "what": "second look at the test blocks (model v2); the first look was v1, see results/test/unlock.json",
        "opened_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_version_git_commit": commit, "n_rows": int(len(P)), "n_blocks": int(P["block_id"].nunique()),
        "chosen_model": name, "decision_threshold": thr}, indent=2))
    print(f"v2 scored on the test set: {len(P):,} rows, {P['block_id'].nunique()} blocks, code version {commit[:7]}")


def cmd_report_v2():
    from methods import V2_ORDER, V2_MAIN, V2_BASELINE, V2_GAPS
    if not UNLOCK_V2.exists():
        sys.exit("Run 'unlock-v2' first.")
    with np.load(PRED_V2, allow_pickle=True) as z:
        P = pd.DataFrame({c: z[c] for c in z.files})
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        shape = src.shape
        assert (src.read(1)[P["row"], P["col"]] == 2).all()
    table, gaps = full_report(P, OUT_V2, "Test land, second look (v2)", shape,
                              order=V2_ORDER, main=V2_MAIN, baseline=V2_BASELINE, gap_pairs=V2_GAPS)
    # the four v1 failure windows, looked at again: how many of those mistakes does v2 still make?
    again = []
    for f in json.loads((OUT / "failures.json").read_text()):
        w = P[(P["date"] == f["date"]) & P["row"].between(f["row0"], f["row0"] + f["window_px"] - 1)
              & P["col"].between(f["col0"], f["col0"] + f["window_px"] - 1)]
        y = w["label"].to_numpy()
        if f["kind"] == "missed damage":
            bad = lambda p: int(((y == 1) & (p == 0)).sum())  # noqa: E731
        else:
            bad = lambda p: int(((y == 0) & (p == 1)).sum())  # noqa: E731
        again.append({"date": f["date"], "kind": f["kind"], "block_ids": f["block_ids"], "n_px": int(len(w)),
                      "mistakes_v1": bad(w["pred__rf_all"].to_numpy()), "mistakes_v2": bad(w["pred__v2"].to_numpy()),
                      "mistakes_rule_war": bad(w["pred__rule_war"].to_numpy())})
    (OUT_V2 / "v1_failures_revisited.json").write_text(json.dumps(again, indent=1))
    cols = ["exam", "method", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "macro_f1_clean_pixels", "accuracy",
            "precision_intact", "recall_intact", "precision_damaged", "recall_damaged"]
    print(table[cols].round(3).to_string(index=False))
    print(json.dumps(gaps, indent=1))
    print(json.dumps(again, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["unlock", "report", "unlock-v2", "report-v2"])
    {"unlock": cmd_unlock, "report": cmd_report, "unlock-v2": cmd_unlock_v2,
     "report-v2": cmd_report_v2}[ap.parse_args().step]()
