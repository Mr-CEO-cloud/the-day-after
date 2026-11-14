"""
07_precompute.py - Phase 6: pack everything the app needs into app/data and app/assets.

Plain words:
- The web app must never call a satellite service or train a model while a judge is
  watching. So everything it shows is computed ahead of time and saved in small files.
- This script reads the planner's tables (data/processed/planner_v2) and the test results
  (results/test_v2, results/planner_v2) and writes:
    app/data/plots.npz    per plot: place, size, pre-war facts; per month: damaged share,
                          status / confidence / "green now" codes, and the UN share where a UN map exists
    app/data/meta.json    month list, boundary outline, and every number the app prints
                          about test results (copied from results/, never typed by hand)
    app/assets/*.png      the score, confusion-matrix and failure-example figures
- The app turns the saved codes back into words with app/planner_rules.py.

Run:  .venv/Scripts/python.exe -I src/07_precompute.py
"""
import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, PROCESSED, RESULTS, METRIC_CRS, LABEL_MONTHS, V2_APP_MONTHS, DISCLAIMER, PLOT_SIZE_M, RESOLUTION_M  # noqa: E402
from common import area  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

APP = ROOT / "app"
MONTHS = LABEL_MONTHS + V2_APP_MONTHS
NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
STATUS = ["intact", "partly damaged", "mostly damaged"]
USE = ["seasonal crops", "mixed trees and crops", "orchard"]
CONF = ["higher", "lower"]
GREEN = ["below normal", "at or above normal"]


def month_label(m: str) -> str:
    return f"{NAMES[int(m[5:]) - 1]} {m[:4]}"


def main():
    (APP / "data").mkdir(parents=True, exist_ok=True)
    (APP / "assets").mkdir(parents=True, exist_ok=True)
    tables = {m: pd.read_csv(PROCESSED / "planner_v2" / f"plots_{m}.csv").set_index("plot_id").sort_index() for m in MONTHS}
    ref = tables[MONTHS[0]]
    static_cols = ["plot_row", "plot_col", "lon", "lat", "productivity_pct", "prewar_peak_ndvi", "tree_share", "built_share"]
    for m, t in tables.items():                                  # the same plots, with the same facts, in every month
        assert (t.index == ref.index).all(), m
        for c in static_cols:
            assert np.allclose(t[c], ref[c], equal_nan=True), (m, c)
    arrays = {"plot_id": ref.index.to_numpy(np.int32)}
    for c in static_cols:
        arrays[c] = ref[c].to_numpy(np.float32)
    for m, t in tables.items():
        assert np.allclose(t["cropland_ha"], ref["cropland_ha"])
    arrays["cropland_px"] = np.round(ref["cropland_ha"].to_numpy() / 0.01).astype(np.int16)   # exact; hectares = pixels x 0.01
    arrays["use_code"] = ref["prewar_use"].map(USE.index).to_numpy(np.int8)
    arrays["land"] = ref["land"].map({"train": 0, "test": 1, "buffer": 2}).fillna(3).to_numpy(np.int8)
    for key, col, names in (("share", "damaged_share", None), ("status", "status", STATUS),
                            ("conf", "confidence", CONF), ("green", "green_now", GREEN)):
        arrays[key] = np.stack([(tables[m][col].map(names.index) if names else tables[m][col]).to_numpy(np.float32 if not names else np.int8)
                                for m in MONTHS])
    arrays["ndvi_vs_normal"] = np.stack([tables[m]["ndvi_vs_normal"].to_numpy(np.float32) for m in MONTHS])
    arrays["un_share"] = np.stack([tables[m]["un_damaged_share"].to_numpy(np.float32) if "un_damaged_share" in tables[m]
                                   else np.full(len(ref), np.nan, np.float32) for m in MONTHS])
    np.savez_compressed(APP / "data" / "plots.npz", **arrays)

    # Khan Younis outline in plot-grid units (one unit = one 100 m plot), so the app can draw it on every map
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        tr, h, w = src.transform, src.height, src.width
    cell = PLOT_SIZE_M / RESOLUTION_M
    rings = []
    for geom in area().geometry:
        for poly in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
            x, y = np.array(poly.exterior.coords).T
            rings.append({"x": np.round((x - tr.c) / (tr.a * cell), 2).tolist(), "y": np.round((y - tr.f) / (tr.e * cell), 2).tolist()})

    # test-result numbers, copied from results/ so the app never prints a number that was typed by hand
    sc = pd.read_csv(RESULTS / "test_v2" / "scores.csv")
    meta = {
        "disclaimer": DISCLAIMER,
        "months": [{"value": m, "label": month_label(m), "has_un_map": m in LABEL_MONTHS} for m in MONTHS],
        "grid": {"rows": int(np.ceil(h / cell)), "cols": int(np.ceil(w / cell))},
        "boundary": rings,
        "scores": sc[["exam", "method", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "accuracy", "recall_intact",
                      "precision_intact", "recall_damaged", "precision_damaged", "macro_f1_clean_pixels"]].round(4).to_dict("records"),
        "gaps": json.loads((RESULTS / "test_v2" / "gap.json").read_text()),
        "plot_level": {d: {k: v for k, v in e["estimators"].items() if k in ("model_calibrated", "v1_calibrated", "baseline_share")}
                       | {"n_plots": e["n_plots"], "un_status_counts": e["un_status_counts"]}
                       for d, e in json.loads((RESULTS / "test_v2" / "plot_level.json").read_text()).items()},
        "area": json.loads((RESULTS / "test_v2" / "area.json").read_text()),
        "ranking": json.loads((RESULTS / "planner_v2" / "validation.json").read_text())["test"],
        "failures_revisited": json.loads((RESULTS / "test_v2" / "v1_failures_revisited.json").read_text()),
        "unlock": json.loads((RESULTS / "test_v2" / "unlock.json").read_text()),
        "n_plots_rated": int(len(ref)),
    }
    for e in meta["plot_level"].values():                       # the confusion matrices are not needed in the app
        for k in ("model_calibrated", "v1_calibrated", "baseline_share"):
            e[k].pop("status_confusion_rows_un_cols_pred", None)
    (APP / "data" / "meta.json").write_text(json.dumps(meta, indent=1))
    for name in ("scores", "confusion", "failures"):
        shutil.copy(RESULTS / "test_v2" / f"{name}.png", APP / "assets" / f"test_{name}.png")
    print(f"packed {len(ref):,} plots x {len(MONTHS)} months; app/data = "
          f"{sum(p.stat().st_size for p in (APP / 'data').iterdir()) / 1e6:.1f} MB, app/assets = "
          f"{sum(p.stat().st_size for p in (APP / 'assets').iterdir()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
