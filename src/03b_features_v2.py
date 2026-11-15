"""
03b_features_v2.py - model v2, step 3: the feature table with season history.

Plain words:
- v1 asked: "is this pixel less green than in the same month of 2022?" One month, one year.
- v2 asks what the UN analysts asked: "month after month, how does this pixel compare
  with ITS OWN NORMAL for that time of year?"
- Three ideas:
    1. NORMAL. For each pixel and each calendar month (Jan..Dec) I average six pre-war
       years (2017-2022). That is the pixel's normal greenness for, say, March.
    2. ANOMALY. Every month's value minus its normal. -0.20 means "much less green than
       usual for this time of year". I do this for greenness (NDVI), plant water (NDMI)
       and bare soil (BSI). A "z-score" divides the anomaly by how much that month
       normally varies, so a noisy month counts less.
    3. HISTORY. For the month I want to judge, I summarise the anomalies of the recent
       past: 1-6 months back, the last 3 / 6 / 12 months, and everything since Oct 2023
       (lowest point, average, how often far below normal, how long ago the lowest point).
- This fixes v1's blind spot: a field that is bare every December shows no change in
  December, but its missing green season in spring is now in the table.
- Optional extra group: neighbourhood averages (each pixel also sees its surroundings).
  The widest window reaches 50 m from the pixel; the split buffer is 100 m.
- Not features: position, block, plot, the date, the label, the UN damaged-cell count.

Outputs:
  data/processed/features_v2/index_stack.npy       NDVI/NDMI/BSI for every month and cropland pixel (cache)
  data/processed/features_v2/features_<month>.npz  one table per target month
  results/feature_summary_v2.json

Run:  .venv/Scripts/python.exe -I src/03b_features_v2.py
"""
import sys
import json
import warnings
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (PROCESSED, RESULTS, MONTHLY_FIRST, MONTHLY_LAST, NORMAL_YEARS, CONFLICT_START,  # noqa: E402
                    LABEL_MONTHS, V2_APP_MONTHS, FEATURES_CONTEXT, FEATURES_V2, V2_NEIGHBOUR)
from common import NODATA, block_ids, plot_ids, read_labels  # noqa: E402

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from scipy import ndimage  # noqa: E402

_spec = importlib.util.spec_from_file_location("features03", Path(__file__).resolve().parent / "03_features.py")
features03 = importlib.util.module_from_spec(_spec)          # reuse v1's index formulas (NDVI, NDMI, BSI)
_spec.loader.exec_module(features03)

MONTHLY = PROCESSED / "monthly"
OUT = PROCESSED / "features_v2"
BANDS = features03.BANDS
INDEX = ["ndvi", "ndmi", "bsi"]
Z_FLOOR = 0.03          # added to the normal spread so a pixel that never varies cannot give a huge z-score
LOW_Z = -1.0            # "far below normal" = more than one normal spread below


def months():
    (y0, m0), (y1, m1) = MONTHLY_FIRST, MONTHLY_LAST
    return [f"{y}-{m:02d}" for y in range(y0, y1 + 1) for m in range(1, 13) if (y0, m0) <= (y, m) <= (y1, m1)]


def read_month(name: str, r, c):
    """The 6 bands of one monthly image at the cropland pixels. All NaN if that month has no image."""
    path = MONTHLY / f"s2_{name}.tif"
    if not path.exists():
        return np.full((len(BANDS), len(r)), np.nan, np.float32)
    with rasterio.open(path) as src:
        return src.read()[:, r, c]


def index_stack(names, r, c) -> np.ndarray:
    """NDVI, NDMI, BSI for every month and cropland pixel: array [month, index, pixel]. Cached on disk."""
    cache, meta = OUT / "index_stack.npy", OUT / "index_stack.json"
    key = {"months": names, "n": int(len(r)), "images": [n for n in names if (MONTHLY / f"s2_{n}.tif").exists()]}
    if cache.exists() and meta.exists() and json.loads(meta.read_text()) == key:
        return np.load(cache, mmap_mode="r")          # same months and same images as last time: reuse
    stack = np.full((len(names), len(INDEX), len(r)), np.nan, np.float32)
    for i, name in enumerate(names):
        idx = features03.indices(read_month(name, r, c))
        for k, key in enumerate(INDEX):
            stack[i, k] = idx[key]
    np.save(cache, stack)
    meta.write_text(json.dumps(key))
    return stack


def normals(stack, names):
    """Per calendar month: each pixel's average and spread over the pre-war years. Arrays [12, index, pixel]."""
    mean = np.full((12,) + stack.shape[1:], np.nan, np.float32)
    std = mean.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                       # all-missing pixels simply stay missing
        for cm in range(1, 13):
            sel = [i for i, n in enumerate(names) if int(n[5:]) == cm and int(n[:4]) in NORMAL_YEARS]
            mean[cm - 1] = np.nanmean(stack[sel], axis=0)
            std[cm - 1] = np.nanstd(stack[sel], axis=0)
    return mean, std


def focal(values, r, c, shape, w, what="mean"):
    """Average (or spread) of a feature over the w x w pixel square around each cropland pixel,
    using cropland pixels only."""
    g = np.zeros(shape, np.float32)
    ok = np.isfinite(values)
    g[r[ok], c[ok]] = values[ok]
    n = np.zeros(shape, np.float32)
    n[r[ok], c[ok]] = 1
    den = ndimage.uniform_filter(n, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ndimage.uniform_filter(g, w) / den
        if what == "mean":
            return mean[r, c]
        sq = ndimage.uniform_filter(g * g, w) / den
        return np.sqrt(np.maximum(sq - mean * mean, 0))[r, c]


def build(t, names, stack, an, z, norm_mean, norm_std, static, r, c, shape) -> dict:
    """All v2 features for target month index t. Returns {column: array over cropland pixels}."""
    name = names[t]
    cm = int(name[5:]) - 1                                    # calendar month 0..11
    f = {}
    bands = read_month(name, r, c)
    for i, b in enumerate(BANDS):
        f[f"now_{b}"] = bands[i]
    for k, key in enumerate(INDEX):
        f[f"now_{key}"], f[f"an_{key}"] = stack[t, k], an[t, k]
    f["z_ndvi"], f["norm_ndvi"], f["norm_ndvi_std"] = z[t], norm_mean[cm, 0], norm_std[cm, 0]
    for i in range(1, 7):
        f[f"an_ndvi_lag{i}"] = an[t - i, 0]
    for i in (1, 2, 3):
        f[f"an_ndmi_lag{i}"], f[f"an_bsi_lag{i}"] = an[t - i, 1], an[t - i, 2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for w in (3, 6, 12):
            s = slice(t - w + 1, t + 1)                       # the last w months, this month included
            f[f"ndvi_min_{w}"], f[f"ndvi_mean_{w}"] = np.nanmin(an[s, 0], axis=0), np.nanmean(an[s, 0], axis=0)
            f[f"z_min_{w}"] = np.nanmin(z[s], axis=0)
            f[f"bsi_max_{w}"], f[f"ndmi_min_{w}"] = np.nanmax(an[s, 2], axis=0), np.nanmin(an[s, 1], axis=0)
        f["ndvi_max_12"] = np.nanmax(stack[t - 11:t + 1, 0], axis=0)
        f["peak_gap_12"] = f["ndvi_max_12"] - static["norm_peak"]
        t0 = names.index(f"{CONFLICT_START[0]}-{CONFLICT_START[1]:02d}")
        war, zw = an[t0:t + 1], z[t0:t + 1]                   # every month since October 2023
        seen = np.isfinite(zw)
        f["war_ndvi_min"], f["war_ndvi_mean"] = np.nanmin(war[:, 0], axis=0), np.nanmean(war[:, 0], axis=0)
        f["war_z_min"] = np.nanmin(zw, axis=0)
        f["war_frac_low"] = (zw < LOW_Z).sum(axis=0) / np.maximum(seen.sum(axis=0), 1)
        worst = np.argmin(np.where(seen, zw, np.inf), axis=0)
        f["war_since_min"] = np.where(seen.any(axis=0), len(zw) - 1 - worst, np.nan).astype(np.float32)   # months since the worst month
        f["war_bsi_max"], f["war_ndmi_min"] = np.nanmax(war[:, 2], axis=0), np.nanmin(war[:, 1], axis=0)
        f["war_frac_seen"] = seen.mean(axis=0)
    f.update(static)
    for w in (5, 11):
        for s in ("an_ndvi", "war_ndvi_min", "war_ndvi_mean", "ndvi_min_12", "now_ndvi"):
            f[f"nb{w}_{s}"] = focal(f[s], r, c, shape, w)
    f["nb5_sd_an_ndvi"] = focal(f["an_ndvi"], r, c, shape, 5, "sd")
    return f


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", default=None, help="only these target months, e.g. 2024-12,2025-07 (for a quick test)")
    only = ap.parse_args().months
    targets = only.split(",") if only else LABEL_MONTHS + V2_APP_MONTHS
    OUT.mkdir(parents=True, exist_ok=True)
    crop = read_labels("2025-07") != NODATA                   # the project's cropland mask
    shape = crop.shape
    r, c = np.nonzero(crop)
    names = months()
    missing = [n for n in names if not (MONTHLY / f"s2_{n}.tif").exists()]
    print(f"{len(names)} months, {len(missing)} without an image {missing[:10]}", flush=True)
    stack = index_stack(names, r, c)
    norm_mean, norm_std = normals(stack, names)
    cm = np.array([int(n[5:]) - 1 for n in names])
    an = stack - norm_mean[cm]                                # anomaly: value minus the normal of that calendar month
    z = an[:, 0] / (norm_std[cm, 0] + Z_FLOOR)                # greenness anomaly in "normal spreads"
    # pre-war facts that never change: v1's context columns + cropland share + the normal yearly high and low
    with np.load(PROCESSED / "features" / "features_2024-12.npz") as v1:
        assert (v1["row"] == r).all() and (v1["col"] == c).all(), "v1 and v2 tables must list the same pixels in the same order"
        static = {k: v1[k] for k in FEATURES_CONTEXT}
    with rasterio.open(PROCESSED / "labels" / "crop_share.tif") as src:
        static["crop_share"] = src.read(1)[r, c]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        static["norm_peak"], static["norm_trough"] = np.nanmax(norm_mean[:, 0], axis=0), np.nanmin(norm_mean[:, 0], axis=0)

    bid, pid = block_ids(shape)[r, c], plot_ids(shape)[r, c]
    summary = {"n_features_core": len(FEATURES_V2), "n_features_neighbourhood": len(V2_NEIGHBOUR), "months": {}}
    for name in targets:
        f = build(names.index(name), names, stack, an, z, norm_mean, norm_std, static, r, c, shape)
        cols = {"row": r.astype(np.int16), "col": c.astype(np.int16), "block_id": bid, "plot_id": pid}
        if name in LABEL_MONTHS:
            with rasterio.open(PROCESSED / "labels" / "monthly" / f"labels_{name}.tif") as src:
                lab = src.read(1)[r, c].astype(np.int16)
            with rasterio.open(PROCESSED / "labels" / "monthly" / f"cells_{name}.tif") as src:
                cells = src.read()[:, r, c]
            cols["label"] = np.where(lab == NODATA, -1, lab).astype(np.int8)     # -1 = not cropland in this month's UN map
            cols["crop_cells"], cols["dmg_cells"] = cells[0], cells[1]
        else:
            cols["label"] = np.full(len(r), -1, np.int8)                          # no UN map for this month
            cols["crop_cells"] = np.round(static["crop_share"] * 16).astype(np.uint8)
            cols["dmg_cells"] = np.zeros(len(r), np.uint8)
        for k in FEATURES_V2 + V2_NEIGHBOUR:
            cols[k] = np.asarray(f[k], dtype=np.float32)
        np.savez(OUT / f"features_{name}.npz", **cols)
        x = np.stack([cols[k] for k in FEATURES_V2])
        summary["months"][name] = {"labelled": name in LABEL_MONTHS, "n_rows": int(len(r)),
                                   "n_labelled_rows": int((cols["label"] >= 0).sum()),
                                   "share_of_missing_values": round(float(np.isnan(x).mean()), 5),
                                   "median_an_ndvi": round(float(np.nanmedian(cols["an_ndvi"])), 4),
                                   "median_war_ndvi_min": round(float(np.nanmedian(cols["war_ndvi_min"])), 4)}
        print(name, summary["months"][name], flush=True)
    if not only:
        (RESULTS / "feature_summary_v2.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
