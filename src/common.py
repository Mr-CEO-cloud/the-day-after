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


def load_table(date: str, part: str, unlock_test: bool = False):
    """The feature table for one date, limited to the pixels a script is allowed to see.

    part = "train": training blocks only, with a `cv_fold` column (0-4) for cross-validation.
    part = "test":  the locked test blocks. Refused unless unlock_test=True, which only the
                    Phase 4 evaluation code may pass.
    part = "all":   every cropland pixel, for drawing prediction maps. No model may be
                    fitted or tuned on this.
    Rows with a missing feature value (a handful of always-cloudy pixels) are dropped for
    train/test and kept for "all".
    """
    import pandas as pd
    import rasterio
    from config import FEATURES_ALL
    with np.load(PROCESSED / "features" / f"features_{date}.npz") as z:
        df = pd.DataFrame({c: z[c] for c in z.files})
    if part == "all":
        return df
    if part == "test" and not unlock_test:
        raise PermissionError("The test blocks are locked until Phase 4 (pass unlock_test=True there only).")
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        split = src.read(1)
    df = df[split[df["row"], df["col"]] == SPLIT_CODES[part]].copy()
    if part == "train":
        with rasterio.open(PROCESSED / "split" / "cv_fold.tif") as src:
            df["cv_fold"] = src.read(1)[df["row"], df["col"]]
    return df[np.isfinite(df[FEATURES_ALL]).all(axis=1)].reset_index(drop=True)
