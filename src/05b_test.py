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

Run:
  .venv/Scripts/python.exe -I src/05b_test.py unlock
  .venv/Scripts/python.exe -I src/05b_test.py report
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["unlock", "report"])
    {"unlock": cmd_unlock, "report": cmd_report}[ap.parse_args().step]()
