"""
02_imagery.py - Phase 1: cloud-free Sentinel-2 "composites" + ESA WorldCover.

Plain words:
- Sentinel-2 photographs Gaza every ~5 days, but some photos have clouds.
- For each time window we take ALL photos, hide cloudy/shadow pixels using the
  satellite's own Scene Classification layer (SCL), then take the MEDIAN of what
  is left, per pixel. The median ignores one-off oddities, giving one clean image.
- Every image is put on the SAME 10 m grid (in metres, UTM 36N), so pixel (i, j)
  is the same spot of land in every image. That makes before/after comparisons fair.
- Correction: images processed since January 2022 store reflectance with +1000
  added (ESA "processing baseline 04.00"). We subtract it, otherwise every
  post-2022 image would look brighter than the pre-war ones.

Outputs (data/processed/composites/):
  post_<date>.tif, pre_<date>.tif   6 bands: B02 B03 B04 B08 B11 B12 (reflectance 0-1)
  ndvi_peak_<year>.tif              pre-war productivity (Feb-Apr NDVI median), 2017-2022
  ndvi_month_2022_<mm>.tif          monthly NDVI in 2022 (seasonality)
  worldcover_2021.tif               ESA WorldCover classes
  manifest.json                     how many photos went into each composite (provenance)

Run:  .venv/Scripts/python.exe -I src/02_imagery.py [--only post,pre,peak,month,wc]
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (PROCESSED, METRIC_CRS, RESOLUTION_M, S2_COLLECTION, MAX_CLOUD_PCT,  # noqa: E402
                    DATES, PREWAR_YEARS, PEAK_SEASON, SEASONALITY_YEAR, SCL_VALID)

import numpy as np  # noqa: E402
import geopandas as gpd  # noqa: E402
import xarray as xr  # noqa: E402
import rioxarray  # noqa: E402,F401  (adds .rio to xarray)
import pystac_client  # noqa: E402
import planetary_computer as pc  # noqa: E402
import odc.stac  # noqa: E402
from odc.geo.geobox import GeoBox  # noqa: E402

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
REFL_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
OUT = PROCESSED / "composites"


from common import grid  # noqa: E402,F401  (the shared 10 m grid)


def catalog():
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def search(cat, start, end, gbox, max_cloud=MAX_CLOUD_PCT):
    bbox = gbox.extent.to_crs("EPSG:4326").boundingbox
    items = list(cat.search(collections=[S2_COLLECTION], bbox=tuple(bbox),
                            datetime=f"{start}/{end}",
                            query={"eo:cloud_cover": {"lt": max_cloud}}).items())
    return items


def baseline_offset(item) -> int:
    """+1000 offset applies to processing baseline >= 04.00 (from 25 Jan 2022)."""
    pb = item.properties.get("s2:processing_baseline", "00.00")
    return 1000 if float(pb) >= 4.0 else 0


def composite(cat, start, end, gbox, bands):
    """Median of cloud-masked reflectance over a time window. Returns (DataArray[band,y,x], n_scenes)."""
    items = search(cat, start, end, gbox)
    if not items:
        return None, 0
    # one offset per calendar day (all tiles from the same day share a baseline)
    offsets = {}
    for it in items:
        offsets[it.datetime.date()] = baseline_offset(it)
    ds = odc.stac.load(items, bands=bands + ["SCL"], geobox=gbox, groupby="solar_day",
                       resampling={"SCL": "nearest", "*": "bilinear"}, pool=16)
    valid = ds["SCL"].isin(SCL_VALID)
    off = xr.DataArray([offsets.get(t.date(), 0) for t in ds.time.to_index()], dims="time")
    layers = []
    for b in bands:
        raw = ds[b].astype("float32")
        refl = (raw - off) / 10_000.0
        refl = refl.where(valid & (ds[b] > 0) & (refl > 0))
        layers.append(refl.median("time", skipna=True))
    out = xr.concat(layers, dim="band").assign_coords(band=bands)
    return out, int(ds.sizes["time"])


def ndvi_of(c):
    nir, red = c.sel(band="B08"), c.sel(band="B04")
    return ((nir - red) / (nir + red)).astype("float32")


def save(da: xr.DataArray, gbox: GeoBox, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    da = da.rio.write_crs(METRIC_CRS)
    if "band" not in da.dims:
        da = da.expand_dims(band=["value"])
    da.rio.to_raster(path, compress="deflate", dtype="float32")


def worldcover(cat, gbox):
    items = list(cat.search(collections=["esa-worldcover"], bbox=tuple(gbox.extent.to_crs("EPSG:4326").boundingbox),
                            query={"esa_worldcover:product_version": {"eq": "2.0.0"}}).items())
    ds = odc.stac.load(items, bands=["map"], geobox=gbox, resampling="nearest", pool=16)
    return ds["map"].isel(time=0).astype("float32"), len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="post,pre,peak,month,wc")
    parts = set(ap.parse_args().only.split(","))
    gbox = grid()
    print(f"grid: {gbox.shape} pixels at {RESOLUTION_M} m, CRS {METRIC_CRS}")
    cat = catalog()
    man_path = OUT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}

    def run(name, start, end, bands, transform=None):
        t0 = time.time()
        c, n = composite(cat, start, end, gbox, bands)
        if c is None:
            print(f"  {name}: NO scenes in {start}..{end}")
            manifest[name] = {"start": start, "end": end, "n_scenes": 0}
            return
        da = transform(c) if transform else c
        valid_pct = float(np.isfinite(da.values).mean() * 100)
        save(da, gbox, OUT / f"{name}.tif")
        manifest[name] = {"start": start, "end": end, "n_scenes": n, "valid_pixel_pct": round(valid_pct, 1)}
        print(f"  {name}: {n} scenes, {valid_pct:.1f}% valid pixels, {time.time() - t0:.0f}s")
        man_path.write_text(json.dumps(manifest, indent=2))

    if "post" in parts or "pre" in parts:
        for d, w in DATES.items():
            for kind in ("post", "pre"):
                if kind in parts:
                    run(f"{kind}_{d}", *w[kind], REFL_BANDS)
    if "peak" in parts:
        for y in PREWAR_YEARS:
            run(f"ndvi_peak_{y}", f"{y}-{PEAK_SEASON[0]}", f"{y}-{PEAK_SEASON[1]}", ["B04", "B08"], ndvi_of)
    if "month" in parts:
        import calendar
        for m in range(1, 13):
            last = calendar.monthrange(SEASONALITY_YEAR, m)[1]
            run(f"ndvi_month_{SEASONALITY_YEAR}_{m:02d}", f"{SEASONALITY_YEAR}-{m:02d}-01",
                f"{SEASONALITY_YEAR}-{m:02d}-{last}", ["B04", "B08"], ndvi_of)
    if "wc" in parts:
        wc, n = worldcover(cat, gbox)
        save(wc, gbox, OUT / "worldcover_2021.tif")
        manifest["worldcover_2021"] = {"n_items": n, "source": "ESA WorldCover 2021 v200"}
        man_path.write_text(json.dumps(manifest, indent=2))
        print("  worldcover_2021 saved")


if __name__ == "__main__":
    main()
