"""
02b_monthly.py - model v2, step 1: one cloud-free satellite image per calendar month, 2017 to 2026.

Plain words:
- Model v1 compared ONE month with the same month of 2022. A field that is normally
  bare in that month shows no change, so damage there was invisible.
- For v2 we build the field's whole history: one clean image for every month from
  January 2017 to October 2026 (118 months).
    * 2017-2022 tells us what is NORMAL for each pixel in each calendar month.
    * 2023 onward is the period we want to judge, month after month.
- Each monthly image is made exactly like the Phase 1 composites (same functions from
  02_imagery.py): take every photo of the month, hide clouds and shadows with the
  satellite's own cloud map, keep the median per pixel, fix the +1000 offset, and put
  it on the project's fixed 10 m grid.
- The script can be stopped and started again: months already saved are skipped.

Outputs:
  data/processed/monthly/s2_YYYY-MM.tif    6 bands: B02 B03 B04 B08 B11 B12 (reflectance 0-1)
  data/processed/monthly/s2_YYYY-MM.json   how many photos went in, share of valid pixels
  results/monthly_manifest.json            all of the above in one file (made by --manifest)

Run:
  .venv/Scripts/python.exe -I src/02b_monthly.py --years 2023-2026
  .venv/Scripts/python.exe -I src/02b_monthly.py --manifest
"""
import sys
import json
import time
import calendar
import argparse
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED, RESULTS, MONTHLY_FIRST, MONTHLY_LAST  # noqa: E402
from common import grid  # noqa: E402

import numpy as np  # noqa: E402

# reuse the Phase 1 compositing functions (a file name starting with a digit needs importlib)
_spec = importlib.util.spec_from_file_location("imagery02", Path(__file__).resolve().parent / "02_imagery.py")
imagery = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(imagery)

OUT = PROCESSED / "monthly"
BANDS = imagery.REFL_BANDS          # B02 B03 B04 B08 B11 B12


def all_months():
    """Every (year, month) from MONTHLY_FIRST to MONTHLY_LAST."""
    (y0, m0), (y1, m1) = MONTHLY_FIRST, MONTHLY_LAST
    return [(y, m) for y in range(y0, y1 + 1) for m in range(1, 13) if (y0, m0) <= (y, m) <= (y1, m1)]


def build(year: int, month: int, cat, gbox) -> dict:
    """Make and save the composite of one month. Returns its manifest entry."""
    name = f"s2_{year}-{month:02d}"
    start, end = f"{year}-{month:02d}-01", f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
    t0 = time.time()
    comp, n = imagery.composite(cat, start, end, gbox, BANDS)
    if comp is None:
        entry = {"start": start, "end": end, "n_scene_days": 0, "valid_pixel_pct": 0.0}
    else:
        imagery.save(comp, gbox, OUT / f"{name}.tif")
        valid = float(np.isfinite(comp.values).all(axis=0).mean() * 100)      # pixels with all 6 bands
        entry = {"start": start, "end": end, "n_scene_days": n, "valid_pixel_pct": round(valid, 1)}
    entry["seconds"] = round(time.time() - t0)
    (OUT / f"{name}.json").write_text(json.dumps(entry))
    return entry


def write_manifest():
    man = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("s2_*.json"))}
    (RESULTS / "monthly_manifest.json").write_text(json.dumps(man, indent=1))
    missing = [f"{y}-{m:02d}" for y, m in all_months() if f"s2_{y}-{m:02d}" not in man]
    empty = [k for k, v in man.items() if v["n_scene_days"] == 0]
    print(f"manifest: {len(man)} months saved, {len(missing)} missing {missing[:12]}, {len(empty)} with no usable photo {empty}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default=None, help="e.g. 2023-2026 (first-last, inclusive)")
    ap.add_argument("--manifest", action="store_true", help="only collect the per-month files into results/monthly_manifest.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        return write_manifest()
    todo = all_months()
    if args.years:
        a, b = (int(v) for v in args.years.split("-"))
        todo = [(y, m) for y, m in todo if a <= y <= b]
    gbox, cat = grid(), imagery.catalog()
    for y, m in todo:
        name = f"s2_{y}-{m:02d}"
        if (OUT / f"{name}.json").exists():
            continue                                            # already done in an earlier run
        for attempt in (1, 2, 3):                               # the image server sometimes hiccups: try up to 3 times
            try:
                e = build(y, m, cat, gbox)
                print(f"{name}: {e['n_scene_days']} days, {e['valid_pixel_pct']}% valid, {e['seconds']}s", flush=True)
                break
            except Exception as err:                            # noqa: BLE001
                print(f"{name}: attempt {attempt} failed: {repr(err)[:200]}", flush=True)
                time.sleep(20 * attempt)
                cat = imagery.catalog()                         # fresh connection and fresh access tokens
    print("done", flush=True)


if __name__ == "__main__":
    main()
