"""
01b_label_history.py - the UN damage map through time (18 monthly snapshots).

Plain words:
- The UN geodatabase holds one damage map per month from Oct 2023 to Jul 2025.
- We put every snapshot on our grid and record, for each cropland pixel, the
  FIRST month it was marked "damaged".
- This is used for two things only:
    1. A cross-check of our numbers against the published FAO table
       (Khan Younis, March 2025: 3 230 of 4 278 ha damaged = 75.5%).
    2. Explaining mistakes in Phase 4 ("was this field damaged long ago and
       has since regrown?").
  It is NEVER given to the model as a feature: it is part of the answer key.

Outputs:
  data/processed/labels/first_damaged.tif   per pixel: index of the first snapshot marked damaged
                                            (0 = Oct 2023 ... 17 = Jul 2025), 254 = never, 255 = not cropland
  results/label_history.json                damaged hectares per snapshot + the FAO cross-check

Run:  .venv/Scripts/python.exe -I src/01b_label_history.py
"""
import sys
import json
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW, PROCESSED, RESULTS, LABEL_GDB_ZIP, METRIC_CRS  # noqa: E402
from common import grid, area, NODATA  # noqa: E402

import numpy as np  # noqa: E402
import geopandas as gpd  # noqa: E402
import pyogrio  # noqa: E402
import rasterio  # noqa: E402

# reuse the exact reading / rasterising functions of 01_labels.py (a file name starting with a digit needs importlib)
_spec = importlib.util.spec_from_file_location("labels01", Path(__file__).resolve().parent / "01_labels.py")
labels01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(labels01)

NEVER = 254
# Published figures: FAO, "Land availability for cultivation in the Gaza Strip as of April 2025"
# (CD5442EN/1/05.25), Table 1 "Damage to cropland in March 2025 (in ha)", Khan Younis row.
FAO_MARCH_2025 = {"damaged_ha": 3230, "total_ha": 4278, "damaged_pct": 75.5}


def main():
    gdb = labels01.find_gdb(RAW / "labels" / LABEL_GDB_ZIP)
    layers = sorted((n for n, _ in pyogrio.list_layers(gdb) if n.startswith("Cropland_Damage")),
                    key=lambda n: n[-8:])                       # the name ends with the image date YYYYMMDD
    area_gdf, gbox = area(), grid()
    first = None
    history, prev = [], None
    for i, layer in enumerate(layers):
        gdf = labels01.read_label_layer(gdb, layer)
        lab = labels01.rasterize_labels(gdf, gbox, area_gdf)
        crop = lab != NODATA
        if first is None:
            first = np.full(lab.shape, NEVER, np.uint8)
            crop0 = crop
        dmg = lab == 1
        first[dmg & (first == NEVER)] = i
        entry = {"snapshot": layer[-8:], "layer": layer, "cropland_ha": round(float(crop.sum()) / 100, 1),
                 "damaged_ha": round(float(dmg.sum()) / 100, 1),
                 "damaged_pct": round(100 * float(dmg.sum()) / float(crop.sum()), 1),
                 "same_cropland_mask_as_first": bool((crop == crop0).all())}
        if prev is not None:   # how many pixels went back from damaged to non-affected since the previous month
            entry["reverted_px"] = int((prev & ~dmg & crop).sum())
        if layer.endswith("20250330"):
            # exact hectares from the polygons (not pixels), for the comparison with the FAO table
            clipped = gpd.clip(gdf, area_gdf)
            ha = (clipped.dissolve("Class").area / 10_000)
            entry["vector_ha"] = {k: round(float(v), 1) for k, v in ha.items()}
        history.append(entry)
        prev = dmg
        print(entry)

    first[~crop] = NODATA      # keep only the cropland mask of the latest snapshot (the one the project uses)
    out = {"snapshots": history}
    m = next(e for e in history if e["snapshot"] == "20250330")
    ours_d, ours_t = m["vector_ha"]["Damaged cropland"], sum(m["vector_ha"].values())
    out["fao_cross_check_march_2025"] = {
        "source": "FAO 2025, Land availability for cultivation in the Gaza Strip as of April 2025 (CD5442EN/1/05.25), Table 1",
        "fao_khan_younis": FAO_MARCH_2025,
        "ours_khan_younis": {"damaged_ha": round(ours_d, 1), "total_ha": round(ours_t, 1),
                             "damaged_pct": round(100 * ours_d / ours_t, 1)},
        "difference_pct_points": round(100 * ours_d / ours_t - FAO_MARCH_2025["damaged_pct"], 1),
        "note": "FAO used UNMAP 2023 boundaries; we use the OCHA COD-AB Khan Younis outline, so totals differ slightly.",
    }
    with rasterio.open(PROCESSED / "labels" / "labels_2025-07.tif") as src:
        prof = src.profile
    prof.update(dtype="uint8", nodata=NODATA, count=1, compress="deflate")
    with rasterio.open(PROCESSED / "labels" / "first_damaged.tif", "w", **prof) as dst:
        dst.write(first, 1)
    (RESULTS / "label_history.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["fao_cross_check_march_2025"], indent=2))


if __name__ == "__main__":
    main()
