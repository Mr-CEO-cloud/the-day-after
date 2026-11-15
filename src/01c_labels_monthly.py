"""
01c_labels_monthly.py - model v2, step 2: the answer key for all 18 UN monthly maps.

Plain words:
- v1 learned from two UN maps (Dec 2024, Jul 2025). The UN geodatabase holds 18 monthly
  maps (Oct 2023 to Jul 2025). v2 learns from all of them, so it sees every season.
- For each month I save the same kind of label as v1: the UN class at the CENTRE of
  each 10 m pixel (0 non-affected, 1 damaged, 255 not cropland). v1 and v2 are therefore
  scored against the same answers.
- New: the UN map is drawn on a finer 5 m grid than my 10 m pixels, so many of my
  pixels are only partly cropland, or part damaged and part intact. I measure this by
  cutting every pixel into 4 x 4 small cells (2.5 m) and counting:
      crop cells     how many of the 16 cells are UN cropland
      damaged cells  how many of the 16 cells are UN "damaged"
  Uses:  * crop share (from the latest map's cropland outline) is a model FEATURE. It
           describes the land, not the damage.
         * damaged cells are used ONLY to give clean pixels more weight in training and
           to report a second score on clean pixels. Never a feature.

Outputs (data/processed/labels/):
  monthly/labels_YYYY-MM.tif    centre label per pixel
  monthly/cells_YYYY-MM.tif     band 1 = crop cells (0-16), band 2 = damaged cells (0-16)
  crop_share.tif                share of each pixel that is cropland (0-1), latest UN outline
  results/labels_monthly.json   counts per month

Run:  .venv/Scripts/python.exe -I src/01c_labels_monthly.py
"""
import sys
import json
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW, PROCESSED, RESULTS, LABEL_GDB_ZIP, LABEL_MONTHS, LABEL_LAYERS  # noqa: E402
from common import grid, area, NODATA, read_labels  # noqa: E402

import numpy as np  # noqa: E402
import pyogrio  # noqa: E402
import rasterio  # noqa: E402
from rasterio import features  # noqa: E402
from affine import Affine  # noqa: E402

_spec = importlib.util.spec_from_file_location("labels01", Path(__file__).resolve().parent / "01_labels.py")
labels01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(labels01)

OUT = PROCESSED / "labels" / "monthly"
K = 4          # each 10 m pixel is cut into K x K = 16 cells of 2.5 m


def fine_cells(gdf, gbox) -> tuple[np.ndarray, np.ndarray]:
    """Count, per 10 m pixel, how many of its 16 small cells are cropland and how many are damaged."""
    h, w = gbox.shape.y, gbox.shape.x
    fine = features.rasterize(((g, int(v) + 1) for g, v in zip(gdf.geometry, gdf["label"])),
                              out_shape=(h * K, w * K), transform=gbox.transform * Affine.scale(1 / K),
                              fill=0, dtype="uint8")                 # 0 not cropland, 1 non-affected, 2 damaged
    blocks = fine.reshape(h, K, w, K)
    return (blocks > 0).sum(axis=(1, 3)).astype(np.uint8), (blocks == 2).sum(axis=(1, 3)).astype(np.uint8)


def main():
    gdb = labels01.find_gdb(RAW / "labels" / LABEL_GDB_ZIP)
    layers = sorted((n for n, _ in pyogrio.list_layers(gdb) if n.startswith("Cropland_Damage")), key=lambda n: n[-8:])
    months = [f"{n[-8:-4]}-{n[-4:-2]}" for n in layers]
    assert months == LABEL_MONTHS, f"config.LABEL_MONTHS does not match the geodatabase: {months}"
    area_gdf, gbox = area(), grid()
    project_crop = read_labels("2025-07") != NODATA          # the project's cropland mask (latest UN map)
    with rasterio.open(PROCESSED / "labels" / "labels_2025-07.tif") as src:
        prof = src.profile
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for month, layer in zip(months, layers):
        gdf = labels01.read_label_layer(gdb, layer)
        lab = labels01.rasterize_labels(gdf, gbox, area_gdf)            # the v1 rule: class at the pixel centre
        lab[~project_crop] = NODATA
        crop_cells, dmg_cells = fine_cells(gdf, gbox)
        used = lab != NODATA
        share = dmg_cells[used] / np.maximum(crop_cells[used], 1)
        clean = (crop_cells[used] == K * K) & ((dmg_cells[used] == 0) | (dmg_cells[used] == K * K))
        summary[month] = {
            "layer": layer, "n_px": int(used.sum()), "damaged_px": int((lab == 1).sum()),
            "damaged_share": round(float((lab == 1).sum() / used.sum()), 4),
            "clean_px_share": round(float(clean.mean()), 4),                                   # fully cropland and fully one class
            "partly_outside_cropland_share": round(float((crop_cells[used] < K * K).mean()), 4),
            "centre_label_equals_majority_share": round(float(((share >= 0.5) == (lab[used] == 1)).mean()), 4),
        }
        p = dict(prof, dtype="uint8", nodata=NODATA, count=1, compress="deflate")
        with rasterio.open(OUT / f"labels_{month}.tif", "w", **p) as dst:
            dst.write(lab, 1)
        p = dict(prof, dtype="uint8", nodata=None, count=2, compress="deflate")
        with rasterio.open(OUT / f"cells_{month}.tif", "w", **p) as dst:
            dst.write(np.stack([crop_cells, dmg_cells]))
        print(month, summary[month], flush=True)
        if month in LABEL_LAYERS:                                        # must equal the v1 label rasters exactly
            assert (lab == read_labels(month)).all(), f"{month}: differs from the v1 label raster"
    # static feature: cropland share of each pixel, from the latest outline
    p = dict(prof, dtype="float32", nodata=None, count=1, compress="deflate")
    with rasterio.open(PROCESSED / "labels" / "crop_share.tif", "w", **p) as dst:
        dst.write((crop_cells / (K * K)).astype("float32"), 1)
    (RESULTS / "labels_monthly.json").write_text(json.dumps(summary, indent=1))
    print("v1 label rasters reproduced exactly for", list(LABEL_LAYERS))


if __name__ == "__main__":
    main()
