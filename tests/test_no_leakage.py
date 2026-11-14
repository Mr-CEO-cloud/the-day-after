"""
test_no_leakage.py - automatic checks that the train/test split cannot leak.

Plain words: "leakage" means the model secretly gets information about the test
land while training, which makes scores look better than they really are. Each
check below is one way that could happen. If any check fails, the script stops
with an error. It reads positions only, never test labels or test features.

Run:  .venv/Scripts/python.exe -I tests/test_no_leakage.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PROCESSED, DATES, BUFFER_M, RESOLUTION_M, N_CV_FOLDS, FEATURES_ALL  # noqa: E402
from common import NODATA, block_ids, plot_ids, read_labels, load_table  # noqa: E402

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from scipy import ndimage  # noqa: E402

LABELLED = [d for d, w in DATES.items() if w["labelled"]]
BOOKKEEPING = {"row", "col", "block_id", "plot_id", "label", "cv_fold"}


def main():
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        split = src.read(1)
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

    # 4. the test set is locked by default
    try:
        load_table(LABELLED[0], "test")
        raise AssertionError("test table loaded without unlock")
    except PermissionError:
        print("ok 4: loading test data without the Phase 4 key is refused")

    # 5. the train table only holds train pixels, at every date, and CV folds are whole blocks
    for d in LABELLED:
        t = load_table(d, "train")
        assert (split[t["row"], t["col"]] == 1).all()
        assert (t["block_id"].to_numpy() == bid[t["row"], t["col"]]).all()
        assert (t["label"].to_numpy() == read_labels(d)[t["row"], t["col"]]).all() and t["label"].isin([0, 1]).all()
        assert (t.groupby("block_id")["cv_fold"].nunique() == 1).all() and set(t["cv_fold"]) == set(range(N_CV_FOLDS))
        assert np.isfinite(t[FEATURES_ALL].to_numpy()).all()
        print(f"ok 5 ({d}): {len(t):,} train rows, all inside train blocks, labels match the UN raster, folds are whole blocks")

    # 6. the split is the same land at every date (a test pixel is never a train pixel at another date)
    keys = [set(zip(load_table(d, "train")["row"], load_table(d, "train")["col"])) for d in LABELLED]
    test_keys = set(zip(*np.nonzero(test_px)))
    assert all(not (k & test_keys) for k in keys)
    print("ok 6: no train row at any date sits on a test pixel")

    # 7. 1 ha plots never straddle two blocks (so plot-level scores are leak-free too)
    crop = read_labels(LABELLED[0]) != NODATA
    pairs = np.unique(np.stack([pid[crop], bid[crop]]), axis=1)
    assert len(np.unique(pairs[0])) == pairs.shape[1]
    print(f"ok 7: each of the {pairs.shape[1]:,} plots lies inside exactly one block")

    # 8. table values equal the rasters at the same (row, col): the table and the map are aligned
    d = LABELLED[0]
    t = load_table(d, "train").sample(2000, random_state=0)
    with rasterio.open(PROCESSED / "composites" / f"post_{d}.tif") as src:
        post = src.read()
    red, nir = post[2][t["row"], t["col"]], post[3][t["row"], t["col"]]
    assert np.allclose((nir - red) / (nir + red), t["post_ndvi"], atol=1e-5)
    assert np.allclose(t["post_ndvi"] - t["pre_ndvi"], t["d_ndvi"], atol=1e-5)
    print("ok 8: feature values match the images pixel for pixel")
    print("ALL LEAKAGE CHECKS PASSED")


if __name__ == "__main__":
    main()
