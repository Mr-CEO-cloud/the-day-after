"""
common.py - small helpers shared by several scripts.

The most important one is grid(): ONE fixed 10 m pixel grid over Khan Younis.
Every raster in the project (satellite composites, labels, WorldCover) is put on
this exact grid, so pixel (row, col) is the same patch of ground everywhere.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED, METRIC_CRS, RESOLUTION_M, BLOCK_SIZE_M, PLOT_SIZE_M  # noqa: E402

import numpy as np  # noqa: E402
import geopandas as gpd  # noqa: E402
from odc.geo.geobox import GeoBox  # noqa: E402

NODATA = 255   # label value for "not cropland / outside the area"


def area() -> gpd.GeoDataFrame:
    """Khan Younis governorate outline (OCHA COD-AB), in metres (UTM 36N)."""
    return gpd.read_file(PROCESSED / "area.geojson").to_crs(METRIC_CRS)


def grid(margin_m: int = 200) -> GeoBox:
    """The project's fixed 10 m grid: Khan Younis bounding box + a small margin."""
    x0, y0, x1, y1 = area().total_bounds
    return GeoBox.from_bbox((x0 - margin_m, y0 - margin_m, x1 + margin_m, y1 + margin_m),
                            crs=METRIC_CRS, resolution=RESOLUTION_M)


def read_labels(date: str) -> np.ndarray:
    """UN label raster for one date: 0 non-affected, 1 damaged, 255 not cropland."""
    import rasterio
    with rasterio.open(PROCESSED / "labels" / f"labels_{date}.tif") as src:
        return src.read(1)


def _cell_ids(shape, size_m: int) -> np.ndarray:
    """Number the squares of a size_m grid laid over the pixel grid (id = square_row * 1000 + square_col)."""
    n = size_m // RESOLUTION_M
    rows, cols = np.indices(shape)
    return ((rows // n) * 1000 + (cols // n)).astype(np.int32)


def block_ids(shape) -> np.ndarray:
    """Which 1 km block each pixel belongs to. Blocks are the unit of the train/test split."""
    return _cell_ids(shape, BLOCK_SIZE_M)


def plot_ids(shape) -> np.ndarray:
    """Which 1 ha plot (100 m cell) each pixel belongs to. 10 x 10 plots sit exactly inside one block."""
    return _cell_ids(shape, PLOT_SIZE_M)


SPLIT_CODES = {"train": 1, "test": 2}   # as written by 04_split.py (3 = buffer, never used)


LABEL_COLUMNS = ["label", "dmg_cells"]   # the answer key (dmg_cells = the UN's fine-grid damaged count, v2 tables only)


def load_table(date: str, part: str, unlock_test: bool = False, version: str = "v1"):
    """The feature table for one date, limited to the pixels a script is allowed to see.

    part = "train": training blocks only, with `cv_fold` (0-4) and `cv_near` columns for
                    cross-validation. Use cv_splits() below to get the folds.
    part = "test":  the locked test blocks. Refused unless unlock_test=True, which only the
                    test-set evaluation code (05b_test.py) may pass.
    part = "all":   every cropland pixel, for drawing prediction maps. The answer-key columns
                    are removed, so nothing can be fitted or scored on it.
    version = "v1": tables from 03_features.py. Rows with a missing feature value (a handful
                    of always-cloudy pixels) are dropped for train/test.
    version = "v2": tables from 03b_features_v2.py (season history). Rows that are not cropland
                    in that month's UN map are dropped for train/test. Missing feature values
                    are kept: the v2 models can handle them.
    Always pass the model X = table[FEATURES_...] from config, never the whole table.
    """
    import pandas as pd
    import rasterio
    from config import FEATURES_ALL
    folder = {"v1": "features", "v2": "features_v2"}[version]
    with np.load(PROCESSED / folder / f"features_{date}.npz") as z:
        df = pd.DataFrame({c: z[c] for c in z.files})
    if part == "all":
        return df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns])
    if part == "test" and not unlock_test:
        raise PermissionError("The test blocks are locked (pass unlock_test=True in the test-set evaluation code only).")
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        split = src.read(1)
    df = df[split[df["row"], df["col"]] == SPLIT_CODES[part]].copy()
    if part == "train":
        for name in ("cv_fold", "cv_near"):
            with rasterio.open(PROCESSED / "split" / f"{name}.tif") as src:
                df[name] = src.read(1)[df["row"], df["col"]]
    if version == "v1":
        return df[np.isfinite(df[FEATURES_ALL]).all(axis=1)].reset_index(drop=True)
    return df[df["label"] >= 0].reset_index(drop=True)


def cv_splits(train):
    """Cross-validation folds for a train table: yields (fold, fit_index, check_index).

    For each fold k: we "check" on the pixels of fold k and "fit" on the other four folds,
    minus any pixel within 100 m of a fold-k block (the same buffer rule as the real test).
    Every tuning step in Phase 3 must use this function, so the practice scores are honest.
    """
    from config import N_CV_FOLDS
    fold, near = train["cv_fold"].to_numpy(), train["cv_near"].to_numpy()
    for k in range(N_CV_FOLDS):
        check = fold == k
        fit = ~check & ((near >> k) & 1 == 0)
        yield k, np.flatnonzero(fit), np.flatnonzero(check)
