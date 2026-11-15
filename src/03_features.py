"""
03_features.py - Phase 2: turn the images into a table the model can learn from.

Plain words:
- A model cannot read a picture directly. It reads a table: one ROW per 10 m
  cropland pixel, one COLUMN per measurement ("feature").
- For each date I measure three kinds of things:
    1. NOW  (post_*): the 6 Sentinel-2 bands of the recent image.
    2. THEN (pre_*):  the same 6 bands in the same month of 2022, before the war.
    3. CHANGE (d_*):  now minus then.
  From the bands I also compute three "indices" (simple formulas that isolate
  one physical thing):
    NDVI = (NIR - red) / (NIR + red)          how green / alive the plants are
    NDMI = (NIR - SWIR1) / (NIR + SWIR1)      how much water is in the plants
    BSI  = bare-soil index                    high when the ground is bare earth
- Plus pre-war context that does not change with the date: how green the pixel
  was at its yearly peak in 2017-2022, how much it varies over the year
  (orchards stay green, seasonal crops go up and down), and whether ESA
  WorldCover 2021 called it trees or built-up.
- The pixel's position (row, col, block) is saved for bookkeeping but is NEVER
  given to the model. A model that knows positions could memorise the map.
- This script does not look at train/test. 04_split.py decides that, and
  common.load_table() hands out only the part a script is allowed to see.

Outputs:
  data/processed/features/features_<date>.npz       one per date (labelled or not)
  results/feature_summary.json                      row counts and missing values

Run:  .venv/Scripts/python.exe -I src/03_features.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (PROCESSED, RESULTS, DATES, PREWAR_YEARS, SEASONALITY_YEAR, FEATURES_ALL)  # noqa: E402
from common import NODATA, block_ids, plot_ids, read_labels  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402

COMP = PROCESSED / "composites"
OUT = PROCESSED / "features"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]   # order inside pre_/post_ files (see 02_imagery.py)
WC_TREE, WC_BUILT = 10, 50                           # ESA WorldCover class codes


def read(name: str) -> np.ndarray:
    with rasterio.open(COMP / f"{name}.tif") as src:
        return src.read()          # shape (bands, rows, cols)


def ratio(a, b):
    """Normalised difference (a - b) / (a + b): the shape of NDVI, NDMI and BSI."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def indices(img: np.ndarray) -> dict:
    """NDVI, NDMI and BSI from one 6-band image."""
    b = dict(zip(BANDS, img))
    return {
        "ndvi": ratio(b["B08"], b["B04"]),
        "ndmi": ratio(b["B08"], b["B11"]),
        "bsi": ratio(b["B11"] + b["B04"], b["B08"] + b["B02"]),
    }


def context() -> dict:
    """Pre-war features that are the same for every date."""
    peaks = np.concatenate([read(f"ndvi_peak_{y}") for y in PREWAR_YEARS])                         # (6 years, rows, cols)
    months = np.concatenate([read(f"ndvi_month_{SEASONALITY_YEAR}_{m:02d}") for m in range(1, 13)])  # (12 months, ...)
    wc = read("worldcover_2021")[0]
    return {
        "prewar_peak_mean": np.nanmean(peaks, axis=0),      # pre-war productivity: average yearly green peak
        "prewar_peak_std": np.nanstd(peaks, axis=0),        # how much that peak changed from year to year
        "season_min": np.nanmin(months, axis=0),            # greenness in the brownest month (orchards stay high)
        "season_amp": np.nanmax(months, axis=0) - np.nanmin(months, axis=0),   # yearly swing (seasonal crops swing a lot)
        "wc_tree": (wc == WC_TREE).astype("float32"),
        "wc_built": (wc == WC_BUILT).astype("float32"),
    }


def build(date: str, crop: np.ndarray, ctx: dict) -> pd.DataFrame:
    """The feature table for one date: one row per cropland pixel."""
    post, pre = read(f"post_{date}"), read(f"pre_{date}")
    f = {}
    for i, b in enumerate(BANDS):
        f[f"post_{b}"], f[f"pre_{b}"] = post[i], pre[i]
    ipost, ipre = indices(post), indices(pre)
    for k in ipost:
        f[f"post_{k}"], f[f"pre_{k}"], f[f"d_{k}"] = ipost[k], ipre[k], ipost[k] - ipre[k]
    f.update(ctx)

    rows, cols = np.nonzero(crop)
    df = pd.DataFrame({"row": rows.astype("int16"), "col": cols.astype("int16"),
                       "block_id": block_ids(crop.shape)[crop], "plot_id": plot_ids(crop.shape)[crop]})
    # label: 0 non-affected, 1 damaged, -1 = this date has no UN map
    df["label"] = read_labels(date)[crop].astype("int8") if DATES[date]["labelled"] else np.int8(-1)
    for name in FEATURES_ALL:                      # fixed column order, from config
        df[name] = f[name][crop].astype("float32")
    return df


def main():
    crop = read_labels("2025-07") != NODATA        # the UN cropland mask (identical at every label date)
    ctx = context()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"n_features": len(FEATURES_ALL), "features": FEATURES_ALL, "dates": {}}
    for date in DATES:
        df = build(date, crop, ctx)
        assert len(df) == int(crop.sum()) and list(df.columns[5:]) == FEATURES_ALL
        assert not df.duplicated(["row", "col"]).any()
        x = df[FEATURES_ALL].to_numpy()
        bad_rows = int((~np.isfinite(x)).any(axis=1).sum())
        # .npz = NumPy's own file format: one named array per column
        np.savez_compressed(OUT / f"features_{date}.npz", **{c: df[c].to_numpy() for c in df.columns})
        summary["dates"][date] = {
            "labelled": DATES[date]["labelled"], "n_rows": len(df), "rows_with_missing_value": bad_rows,
            "median_post_ndvi": round(float(np.nanmedian(df["post_ndvi"])), 4),
            "median_pre_ndvi": round(float(np.nanmedian(df["pre_ndvi"])), 4),
        }
        print(f"  {date}: {len(df):,} rows x {len(FEATURES_ALL)} features, {bad_rows} rows with a missing value")
    (RESULTS / "feature_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
