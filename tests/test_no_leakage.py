"""
test_no_leakage.py - automatic checks that the train/test split cannot leak.

Plain words: "leakage" means the model secretly gets information about the test
land while training, which makes scores look better than they really are. Each
check below is one way that could happen. If any check fails, the script stops
with an error. It never reads test labels or test feature values: for test land
it uses positions only.

Run:  .venv/Scripts/python.exe -I tests/test_no_leakage.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import (PROCESSED, DATES, PREWAR_YEARS, BUFFER_M, RESOLUTION_M, N_CV_FOLDS,  # noqa: E402
                    FEATURES_ALL, METRIC_CRS)
from common import block_ids, plot_ids, read_labels, load_table, cv_splits, grid  # noqa: E402

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from scipy import ndimage  # noqa: E402

LABELLED = [d for d, w in DATES.items() if w["labelled"]]
BOOKKEEPING = {"row", "col", "block_id", "plot_id", "label", "cv_fold", "cv_near"}
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def raster(path):
    with rasterio.open(path) as src:
        return src.read()


def main():
    split = raster(PROCESSED / "split" / "split.tif")[0]
    bid, pid = block_ids(split.shape), plot_ids(split.shape)
    train_px, test_px = split == 1, split == 2

    # 1. no 1 km block holds both train and test pixels
    assert not set(np.unique(bid[train_px])) & set(np.unique(bid[test_px]))
    print("ok 1: no block is in both train and test")

    # 2. every train pixel is further than the buffer from every test pixel
    gap = (ndimage.distance_transform_edt(~test_px) * RESOLUTION_M)[train_px].min()
    assert gap > BUFFER_M
    print(f"ok 2: closest train pixel is {gap:.0f} m from test land (buffer {BUFFER_M} m)")

    # 3. the model's feature list contains no position or answer columns
    assert not set(FEATURES_ALL) & BOOKKEEPING
    print("ok 3: no position / label column in the feature list")

    # 4. the test set is locked by default, and the map-drawing table carries no labels
    try:
        load_table(LABELLED[0], "test")
        raise AssertionError("test table loaded without unlock")
    except PermissionError:
        pass
    everything = load_table(LABELLED[0], "all")
    assert "label" not in everything.columns
    print('ok 4: test data is refused without the Phase 4 key; the "all" table has no label column')

    # 5. the train table only holds train pixels, at every date, with the right labels
    tables = {d: load_table(d, "train") for d in LABELLED}
    for d, t in tables.items():
        assert (split[t["row"], t["col"]] == 1).all()
        assert (t["block_id"].to_numpy() == bid[t["row"], t["col"]]).all()
        assert (t["label"].to_numpy() == read_labels(d)[t["row"], t["col"]]).all() and t["label"].isin([0, 1]).all()
        assert np.isfinite(t[FEATURES_ALL].to_numpy()).all()
        print(f"ok 5 ({d}): {len(t):,} train rows, all inside train blocks, labels match the UN raster")

    # 6. cross-validation folds are whole blocks AND buffered from each other
    t = tables[LABELLED[0]]
    assert (t.groupby("block_id")["cv_fold"].nunique() == 1).all() and set(t["cv_fold"]) == set(range(N_CV_FOLDS))
    worst, kept = np.inf, []
    for k, fit, check in cv_splits(t):
        assert not set(fit) & set(check)
        assert not set(t["block_id"].iloc[fit]) & set(t["block_id"].iloc[check])
        check_px = np.zeros(split.shape, bool)
        check_px[t["row"].iloc[check], t["col"].iloc[check]] = True
        dist = ndimage.distance_transform_edt(~check_px) * RESOLUTION_M
        worst = min(worst, dist[t["row"].iloc[fit], t["col"].iloc[fit]].min())
        kept.append(len(fit) / (len(t) - len(check)))
    assert worst > BUFFER_M
    print(f"ok 6: CV folds are whole blocks; closest fit pixel is {worst:.0f} m from its check fold "
          f"({min(kept):.0%}-{max(kept):.0%} of the other folds' pixels kept)")

    # 7. 1 ha plots never straddle two blocks (so plot-level scores are leak-free too), using the stored ids
    assert (everything["plot_id"].to_numpy() == pid[everything["row"], everything["col"]]).all()
    assert (everything.groupby("plot_id")["block_id"].nunique() == 1).all()
    print(f"ok 7: each of the {everything['plot_id'].nunique():,} plots lies inside exactly one block")

    # 8. every raster sits on the one project grid (same shape, corner, pixel size and map projection)
    g = grid()
    files = sorted((PROCESSED / "composites").glob("*.tif")) + sorted((PROCESSED / "labels").glob("*.tif")) \
        + sorted((PROCESSED / "split").glob("*.tif"))
    for f in files:
        with rasterio.open(f) as src:
            assert (src.height, src.width) == tuple(g.shape), f.name
            assert src.crs.to_string() == METRIC_CRS, f.name
            assert np.allclose(tuple(src.transform)[:6], tuple(g.affine)[:6]), f.name
    print(f"ok 8: all {len(files)} rasters share one grid")

    # 9. table values equal the images at the same (row, col), for every date and every kind of feature
    peaks = np.concatenate([raster(PROCESSED / "composites" / f"ndvi_peak_{y}.tif") for y in PREWAR_YEARS])
    for d in DATES:
        a = load_table(d, "all")
        a = a[split[a["row"], a["col"]] == 1].sample(2000, random_state=0)     # train land only
        r, c = a["row"].to_numpy(), a["col"].to_numpy()
        post, pre = raster(PROCESSED / "composites" / f"post_{d}.tif"), raster(PROCESSED / "composites" / f"pre_{d}.tif")
        for i, b in enumerate(BANDS):
            assert np.allclose(post[i][r, c], a[f"post_{b}"], equal_nan=True), (d, b)
            assert np.allclose(pre[i][r, c], a[f"pre_{b}"], equal_nan=True), (d, b)
        ndvi = (post[3] - post[2]) / (post[3] + post[2])
        assert np.allclose(ndvi[r, c], a["post_ndvi"], atol=1e-5, equal_nan=True)
        assert np.allclose(a["post_ndvi"] - a["pre_ndvi"], a["d_ndvi"], atol=1e-5, equal_nan=True)
        assert np.allclose(np.nanmean(peaks, axis=0)[r, c], a["prewar_peak_mean"], atol=1e-5)
    print(f"ok 9: table values match the images pixel for pixel at all {len(DATES)} dates")
    print("ALL LEAKAGE CHECKS PASSED")


if __name__ == "__main__":
    main()
