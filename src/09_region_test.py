"""
09_region_test.py - the new-place test: score the frozen Khan Younis models on a governorate
they have never seen (Deir al-Balah).

Plain words:
- Every model in this project learned from Khan Younis only. This script takes the saved models
  exactly as they are (no refitting, no new thresholds, no new calibration) and asks them about
  ALL the farmland of another governorate, then compares their answers with the UN maps there.
- The rules were written in docs/test_plan_region.md and pushed before this ran. The script
  refuses to run with uncommitted code, refuses to run twice, checks that the model files are
  the ones that were fitted, and records the code version and time.
- "report" can be re-run: it only reads the saved answers.

Run (the area is chosen with an environment variable, see config.py):
  DAYAFTER_AREA=deir_al_balah .venv/Scripts/python.exe -I src/09_region_test.py score
  DAYAFTER_AREA=deir_al_balah .venv/Scripts/python.exe -I src/09_region_test.py report
"""
import sys
import json
import hashlib
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ROOT, PROCESSED, RESULTS, MAIN_AREA, AREA_NAME, AREA_KEY, FEATURES_ALL, LABEL_MONTHS,  # noqa: E402
                    RESOLUTION_M)
from methods import (LABELLED, TIME_FIT, TIME_CHECK, MODELS_DIR, V2_MAIN, V2_BASELINE, V2_GAPS, scores)  # noqa: E402
from evaluation import full_report  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

HOME = ROOT / "results"                         # Khan Younis results (thresholds, fingerprints, its own test scores)
PRED = PROCESSED / "region_test" / "predictions.npz"
UNLOCK = RESULTS / "unlock.json"
ORDER = ["majority", "ndvi_drop", "rule_war", "rf_all", "v2"]
KEEP = ["post_ndvi", "pre_ndvi", "d_ndvi", "d_bsi", "prewar_peak_mean"]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def table(folder: str, month: str) -> pd.DataFrame:
    """Every cropland pixel of this governorate for one month, WITH its UN label (this whole area is the test)."""
    with np.load(PROCESSED / folder / f"features_{month}.npz") as z:
        return pd.DataFrame({c: z[c] for c in z.files})


def cmd_score():
    import joblib
    assert not MAIN_AREA, "set DAYAFTER_AREA to the new governorate: this test is never run on Khan Younis"
    if UNLOCK.exists():
        sys.exit(f"The new-place test was already scored ({UNLOCK}). Use 'report'.")
    dirty = git("status", "--porcelain", "--", "src", "tests", "docs/test_plan_region.md")
    if dirty:
        sys.exit("Commit all code and docs/test_plan_region.md first:\n" + dirty)
    commit = git("rev-parse", "HEAD")
    info = json.loads((HOME / "v2" / "final_models.json").read_text())
    name = info["chosen"]
    for fname, digest in info["file_sha256"].items():            # the v2 files are the ones fitted on Khan Younis
        assert hashlib.sha256((MODELS_DIR / fname).read_bytes()).hexdigest() == digest, f"{fname} changed since it was fitted"
    v1_hashes = json.loads((HOME / "test" / "model_hashes.json").read_text())["files"]
    for fname in ("rf_all.joblib", "rf_all@time.joblib", "ndvi_drop.joblib", "ndvi_drop@time.joblib", "rf_all_calibrator.joblib"):
        assert hashlib.sha256((MODELS_DIR / fname).read_bytes()).hexdigest() == v1_hashes[fname]["sha256"], f"{fname} changed"
    rules = json.loads((HOME / "v2" / "rule_thresholds.json").read_text())["rule_war"]
    majority = info["majority"]

    parts = []
    for d in LABELLED:
        a, b = table("features", d), table("features_v2", d)       # first-model features, season-history features
        assert (a["row"] == b["row"]).all() and (a["col"] == b["col"]).all()
        b = b.drop(columns=[c for c in b.columns if c in a.columns and c not in ("row", "col")] + ["row", "col"])
        t = pd.concat([a, b], axis=1)
        t["date"] = d
        parts.append(t)
    T = pd.concat(parts, ignore_index=True)
    T = T[(T["label"] >= 0) & np.isfinite(T[FEATURES_ALL]).all(axis=1)].reset_index(drop=True)   # same rule as the Khan Younis tests
    is_check = (T["date"] == TIME_CHECK).to_numpy()
    P = T[["row", "col", "date", "label", "block_id", "plot_id"] + KEEP].copy()

    P["pred__majority"] = T["date"].map(majority).to_numpy().astype("int8")
    for m in ("ndvi_drop", "rf_all"):                              # the first rule and the first model, as saved
        P["pred__" + m] = joblib.load(MODELS_DIR / f"{m}.joblib").predict(T).astype("int8")
        P["time__" + m] = np.int8(-1)
        P.loc[is_check, "time__" + m] = joblib.load(MODELS_DIR / f"{m}@time.joblib").predict(T[is_check]).astype("int8")
    P["time__majority"] = np.where(is_check, majority[TIME_FIT], -1).astype("int8")
    v1 = joblib.load(MODELS_DIR / "rf_all.joblib")
    P["cal_v1"] = joblib.load(MODELS_DIR / "rf_all_calibrator.joblib").predict(v1.proba(T)).astype("float32")

    bundle = joblib.load(MODELS_DIR / f"v2_{name}.joblib")
    forward = joblib.load(MODELS_DIR / f"v2_{name}@forward.joblib")
    thr = bundle["threshold"]
    P["score"] = bundle["model"].predict_proba(T[bundle["features"]].to_numpy(np.float32))[:, 1].astype("float32")
    P["pred__v2"] = (P["score"] >= thr).astype("int8")
    P["time__v2"] = np.int8(-1)
    P.loc[is_check, "time__v2"] = (forward["model"].predict_proba(
        T.loc[is_check, forward["features"]].to_numpy(np.float32))[:, 1] >= thr).astype("int8")
    x = np.nan_to_num(T[rules["feature"]].to_numpy(), nan=0.0)
    P["pred__rule_war"] = (x < T["date"].map(rules["thresholds"]).to_numpy()).astype("int8")
    P["time__rule_war"] = np.int8(-1)
    P.loc[is_check, "time__rule_war"] = (x[is_check] < rules["thresholds"][TIME_FIT]).astype("int8")
    P["cal"] = joblib.load(MODELS_DIR / f"v2_{name}_calibrator.joblib").predict(P["score"]).astype("float32")
    crop, dmg = T["crop_cells"].to_numpy(), T["dmg_cells"].to_numpy()
    P["clean"] = (crop == 16) & ((dmg == 0) | (dmg == 16))

    # second numbers: the main model and the season rule at each of the 18 UN months
    by_month = []
    for m in LABEL_MONTHS:
        t = table("features_v2", m)
        t = t[t["label"] >= 0]
        y = t["label"].to_numpy()
        p = bundle["model"].predict_proba(t[bundle["features"]].to_numpy(np.float32))[:, 1] >= thr
        r = np.nan_to_num(t[rules["feature"]].to_numpy(), nan=0.0) < rules["thresholds"][m]
        by_month.append({"month": m, "n": int(len(t)), "un_damaged_share": round(float(y.mean()), 4),
                         "v2_macro_f1": round(scores(y, p.astype(int))["macro_f1"], 4),
                         "rule_war_macro_f1": round(scores(y, r.astype(int))["macro_f1"], 4)})

    PRED.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED, **{c: P[c].to_numpy() for c in P.columns})
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(by_month).to_csv(RESULTS / "by_month.csv", index=False)
    P[["date", "row", "col", "block_id", "plot_id", "label", "clean", "pred__rule_war", "pred__rf_all", "pred__v2",
       "time__rule_war", "time__rf_all", "time__v2", "cal_v1", "cal"]].round(4).to_csv(RESULTS / "predictions_main.csv.gz", index=False)
    UNLOCK.write_text(json.dumps({
        "what": f"new-place test: frozen Khan Younis models scored once on all cropland of {AREA_NAME}",
        "scored_at": datetime.now().astimezone().isoformat(timespec="seconds"), "code_version_git_commit": commit,
        "n_rows": int(len(P)), "n_blocks": int(P["block_id"].nunique()), "rows_per_date": P["date"].value_counts().to_dict(),
        "main_model": f"v2 ({name})", "decision_threshold": thr}, indent=2))
    print(f"{AREA_NAME}: {len(P):,} rows, {P['block_id'].nunique()} blocks scored, code version {commit[:7]}")


def cmd_report():
    if not UNLOCK.exists():
        sys.exit("Run 'score' first.")
    with np.load(PRED, allow_pickle=True) as z:
        P = pd.DataFrame({c: z[c] for c in z.files})
    with rasterio.open(PROCESSED / "labels" / "labels_2025-07.tif") as src:
        shape = src.shape
    tab, gaps = full_report(P, RESULTS / "region_test", f"{AREA_NAME}, never seen by any model", shape,
                            order=ORDER, main=V2_MAIN, baseline=V2_BASELINE, gap_pairs=V2_GAPS)
    home = pd.read_csv(HOME / "test_v2" / "scores.csv").set_index(["exam", "method"])
    here = tab.set_index(["exam", "method"])
    unlock = json.loads(UNLOCK.read_text())
    names = {"same-date 2024-12": "Dec 2024", "same-date 2025-07": "Jul 2025", "time 2024-12->2025-07": "Forward in time"}
    rows = []
    for exam, label in names.items():
        r, g1, g2 = here.loc[(exam, "v2")], gaps[exam], gaps[f"{exam} | v2 vs rf_all"]
        rows.append([label, f"{r['macro_f1']:.2f} ({r['macro_f1_ci_low']:.2f} to {r['macro_f1_ci_high']:.2f})",
                     f"{home.loc[(exam, 'v2'), 'macro_f1']:.2f}", f"{here.loc[(exam, 'rule_war'), 'macro_f1']:.2f}",
                     f"{here.loc[(exam, 'rf_all'), 'macro_f1']:.2f}",
                     f"{g1['macro_f1_gap']:+.2f} ({g1['ci_low']:+.3f} to {g1['ci_high']:+.3f})"])
    ha = P.groupby("date").size() * RESOLUTION_M ** 2 / 10_000
    d, j = here.loc[("same-date 2024-12", "v2"), "macro_f1"], here.loc[("same-date 2025-07", "v2"), "macro_f1"]
    hd, hj = home.loc[("same-date 2024-12", "v2"), "macro_f1"], home.loc[("same-date 2025-07", "v2"), "macro_f1"]
    card = {
        "area": AREA_NAME, "area_key": AREA_KEY, "scored_at": unlock["scored_at"], "commit": unlock["code_version_git_commit"][:7],
        "summary": (f"The models learned only from Khan Younis. Without any change, they were scored once on all "
                    f"{ha.max():,.0f} hectares of farmland in {AREA_NAME}, the governorate to the north. The main model scored "
                    f"{d:.2f} (Dec 2024) and {j:.2f} (Jul 2025) there, against {hd:.2f} and {hj:.2f} on Khan Younis test land."),
        "columns": ["Test", f"Main model in {AREA_NAME} (95% range)", "Main model in Khan Younis", "Season rule here", "First model here",
                    "Main model minus rule (95% range)"],
        "rows": rows,
        "note": ("Scores are macro F1, from 0 to 1. The rules of this test were published before it was scored "
                 f"(docs/test_plan_region.md); it was run once, at {unlock['scored_at'].replace('T', ' ')[:16]}, code version "
                 f"{unlock['code_version_git_commit'][:7]}. A lead over the rule counts only if its whole range is above zero."),
    }
    (RESULTS / "region_test.json").write_text(json.dumps(card, indent=1))
    cols = ["exam", "method", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "macro_f1_clean_pixels", "accuracy",
            "precision_intact", "recall_intact", "precision_damaged", "recall_damaged"]
    print(tab[cols].round(3).to_string(index=False))
    print(json.dumps(gaps, indent=1))
    print(pd.read_csv(RESULTS / "by_month.csv").to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["score", "report"])
    {"score": cmd_score, "report": cmd_report}[ap.parse_args().step]()
