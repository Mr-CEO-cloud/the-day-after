"""
common.py - small helpers shared by several scripts.

The most important one is grid(): ONE fixed 10 m pixel grid over Khan Younis.
Every raster in the project (satellite composites, labels, WorldCover) is put on
this exact grid, so pixel (row, col) is the same patch of ground everywhere.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED, METRIC_CRS, RESOLUTION_M  # noqa: E402

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
