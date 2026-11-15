"""
07c_export_model.py - put the trained model itself into the web page ("check it yourself").

Plain words:
- A judge should not have to believe my score table. So the page carries the REAL trained
  model (v2) and a sample of test pixels the model never saw, and runs the model in the
  visitor's browser while they watch.
- This script writes three files for the page:
    web/data/model.json    the 400 decision trees of model v2, exactly as trained (every split:
                           which feature, which threshold, where to go next; every leaf value),
                           plus the calibration curve and the simple rule's thresholds
    web/data/sample.bin    the 61 feature values of 6,000 test pixels (3,000 from Dec 2024 and
                           3,000 from Jul 2025), picked at random with the project's fixed seed
    web/data/sample.json   for the same pixels: the UN label, the answer Python saved when the
                           test set was scored, and (for 600 of them) the monthly greenness
                           history since 2017 with the pixel's own normal
- Nothing is fitted, tuned or re-scored here. The labels and Python answers are copied from the
  file saved at the second test look (data/processed/test_v2/predictions.npz). The feature
  values are read from the label-free map tables.
- tests/test_web.py then checks that the page's JavaScript gives the same probability as Python
  for every one of the 6,000 pixels.

Run:  .venv/Scripts/python.exe -I src/07c_export_model.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ROOT, PROCESSED, RESULTS, SEED, METRIC_CRS, MONTHLY_FIRST, MONTHLY_LAST, NORMAL_YEARS)  # noqa: E402
from common import load_table  # noqa: E402
from methods import MODELS_DIR, LABELLED  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402

WEB = ROOT / "web" / "data"
N_PER_MONTH = 3000          # test pixels per exam month in the page's sample
N_HISTORY = 300             # of those, how many per month also carry their greenness history


def export_trees(booster) -> list:
    """Each tree as flat lists. A child >= 0 is another split; a child < 0 is leaf number (-child - 1)."""
    trees = []
    for info in booster.dump_model()["tree_info"]:
        n_split = info["num_leaves"] - 1
        f, t, left, right, flag = [0] * n_split, [0.0] * n_split, [0] * n_split, [0] * n_split, [0] * n_split
        leaf = [0.0] * info["num_leaves"]

        def walk(node):
            if "leaf_value" in node:
                leaf[node.get("leaf_index", 0)] = node["leaf_value"]
                return -node.get("leaf_index", 0) - 1
            i = node["split_index"]
            assert node["decision_type"] == "<=" and node["missing_type"] in ("NaN", "None")
            f[i], t[i] = node["split_feature"], node["threshold"]
            flag[i] = int(node["default_left"]) + 2 * int(node["missing_type"] == "NaN")   # bit 0: missing goes left; bit 1: NaN counts as missing
            left[i], right[i] = walk(node["left_child"]), walk(node["right_child"])
            return i
        root = walk(info["tree_structure"])
        assert root == 0 or n_split == 0
        trees.append({"f": f, "t": t, "l": left, "r": right, "d": flag, "v": leaf})
    return trees


def month_names():
    (y0, m0), (y1, m1) = MONTHLY_FIRST, MONTHLY_LAST
    return [f"{y}-{m:02d}" for y in range(y0, y1 + 1) for m in range(1, 13) if (y0, m0) <= (y, m) <= (y1, m1)]


def main():
    import joblib
    info = json.loads((RESULTS / "v2" / "final_models.json").read_text())
    name = info["chosen"]
    bundle = joblib.load(MODELS_DIR / f"v2_{name}.joblib")
    curve = joblib.load(MODELS_DIR / f"v2_{name}_calibrator.joblib")
    rules = json.loads((RESULTS / "v2" / "rule_thresholds.json").read_text())["rule_war"]
    feats = bundle["features"]
    WEB.mkdir(parents=True, exist_ok=True)
    model = {
        "name": f"v2 ({name})", "kind": "LightGBM gradient-boosted decision trees, binary: probability = 1 / (1 + exp(-sum of leaf values))",
        "features": feats, "threshold": bundle["threshold"], "trees": export_trees(bundle["model"].booster_),
        "calibration": {"x": [float(v) for v in curve.X_thresholds_], "y": [float(v) for v in curve.y_thresholds_]},
        "rule": {"feature": rules["feature"], "thresholds": {m: rules["thresholds"][m] for m in LABELLED}},
        "file_sha256": info["file_sha256"][f"v2_{name}.joblib"],
    }
    (WEB / "model.json").write_text(json.dumps(model, separators=(",", ":")))

    # ---- the sample of test pixels: labels and Python answers come from the saved second-look file
    with np.load(PROCESSED / "test_v2" / "predictions.npz", allow_pickle=True) as z:
        P = pd.DataFrame({c: z[c] for c in ("row", "col", "date", "label", "score", "cal", "pred__v2", "pred__rule_war", "clean")})
    rng = np.random.default_rng(SEED)
    picks = []
    for d in LABELLED:
        idx = np.flatnonzero((P["date"] == d).to_numpy())
        picks.append(np.sort(rng.choice(idx, N_PER_MONTH, replace=False)))
    S = P.iloc[np.concatenate(picks)].reset_index(drop=True)
    X = np.empty((len(S), len(feats)), np.float32)
    crop_index = np.empty(len(S), np.int64)                       # position of the pixel in the list of all cropland pixels
    for d in LABELLED:
        a = load_table(d, "all", version="v2")                    # label-free table of every cropland pixel
        key = pd.Series(np.arange(len(a)), index=pd.MultiIndex.from_arrays([a["row"], a["col"]]))
        m = (S["date"] == d).to_numpy()
        pos = key.reindex(pd.MultiIndex.from_arrays([S.loc[m, "row"], S.loc[m, "col"]])).to_numpy()
        assert np.isfinite(pos).all()
        X[m] = a[feats].to_numpy(np.float32)[pos.astype(int)]
        crop_index[m] = pos.astype(int)
    # the saved Python answer must be reproduced from these features by the saved model (a self-check of the export)
    again = bundle["model"].predict_proba(X)[:, 1]
    assert np.allclose(again, S["score"].to_numpy(), atol=1e-6), "sample features do not reproduce the saved test answers"
    X.astype("<f4").tofile(WEB / "sample.bin")

    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        tr = src.transform
        assert (src.read(1)[S["row"], S["col"]] == 2).all()        # every sampled pixel is test land
    x = tr.c + (S["col"].to_numpy() + 0.5) * tr.a
    y = tr.f + (S["row"].to_numpy() + 0.5) * tr.e
    lon, lat = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True).transform(x, y)

    # ---- greenness history for the first N_HISTORY sampled pixels of each month
    names = month_names()
    stack = np.load(PROCESSED / "features_v2" / "index_stack.npy", mmap_mode="r")      # [month, index (0 = NDVI), cropland pixel]
    hist_rows = np.concatenate([np.flatnonzero((S["date"] == d).to_numpy())[:N_HISTORY] for d in LABELLED])
    series = np.asarray(stack[:, 0, :][:, crop_index[hist_rows]]).T                   # [pixel, month]
    cal_month = np.array([int(n[5:]) - 1 for n in names])
    pre = np.array([int(n[:4]) in NORMAL_YEARS for n in names])
    normal = np.stack([np.nanmean(series[:, pre & (cal_month == k)], axis=1) for k in range(12)], axis=1)   # [pixel, 12]
    q = lambda a: [[None if not np.isfinite(v) else int(round(float(v) * 1000)) for v in r] for r in a]  # noqa: E731

    sample = {
        "n": int(len(S)), "n_features": len(feats), "months": LABELLED,
        "month": [LABELLED.index(d) for d in S["date"]],
        "label": S["label"].astype(int).tolist(),                              # the UN answer: 1 damaged, 0 non-affected
        "py_prob": [round(float(v), 7) for v in S["score"]],                   # the probability Python saved at the test look
        "py_cal": [round(float(v), 6) for v in S["cal"]],
        "py_rule": S["pred__rule_war"].astype(int).tolist(),
        "clean": S["clean"].astype(int).tolist(),
        "lat": np.round(lat, 5).tolist(), "lon": np.round(lon, 5).tolist(),
        "row": S["row"].astype(int).tolist(), "col": S["col"].astype(int).tolist(),
        "history": {"months": names, "pixels": hist_rows.astype(int).tolist(),
                    "ndvi_x1000": q(series), "normal_x1000": q(normal)},
        "note": "A random sample (fixed seed 42) of the 250,536 test pixels scored at the second test look.",
    }
    (WEB / "sample.json").write_text(json.dumps(sample, separators=(",", ":")))
    for f in ("model.json", "sample.bin", "sample.json"):
        print(f"{f}: {(WEB / f).stat().st_size / 1e6:.2f} MB")
    print(f"sample: {len(S):,} test pixels; Python answers reproduced from the exported features (max difference "
          f"{np.abs(again - S['score'].to_numpy()).max():.1e})")


if __name__ == "__main__":
    main()
