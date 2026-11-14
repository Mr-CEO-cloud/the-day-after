"""
01_labels.py - Phase 1: get the expert labels and the area outline.

Plain words:
1. Download the UNOSAT-FAO cropland damage maps (the "answer key" made by UN experts).
2. Download the official Khan Younis outline (OCHA admin boundaries) and save it.
3. Print what is inside each damage map (layers, columns, categories), so we can
   see which classes really exist before building anything on top of them.

Run:  .venv/Scripts/python.exe -I src/01_labels.py [--inspect-only]
"""
import sys
import zipfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (RAW, PROCESSED, RESULTS, LABEL_SOURCES, LABEL_GDB_ZIP, LABEL_LAYERS,  # noqa: E402
                    AREA_NAME, METRIC_CRS)
from common import grid, area, NODATA  # noqa: E402

import json  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
import rioxarray  # noqa: E402,F401
import geopandas as gpd  # noqa: E402
import pyogrio  # noqa: E402
from rasterio import features  # noqa: E402

ADMIN_URL = ("https://data.humdata.org/dataset/2caf8373-816f-458c-9913-71bddb9cab7c/resource/"
             "ca372385-4c79-4378-abf1-cb506fb98023/download/pse_admin_boundaries.geojson.zip")
AREA_PCODE = "PS0270"   # Khan Younis governorate in OCHA COD-AB


def download(url: str, dest: Path) -> Path:
    """Download a file once; skip if it is already there."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    return dest


def unzip(zpath: Path) -> Path:
    out = zpath.with_suffix("")
    if not out.exists():
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
    return out


def find_gdb(folder: Path) -> Path:
    gdbs = [p for p in folder.rglob("*.gdb") if p.is_dir()]
    if not gdbs:
        raise FileNotFoundError(f"no .gdb inside {folder}")
    return gdbs[0]


def area_outline() -> gpd.GeoDataFrame:
    """Official Khan Younis outline from OCHA COD-AB (CC BY-IGO)."""
    z = download(ADMIN_URL, RAW / "admin" / "pse_admin_boundaries.geojson.zip")
    folder = unzip(z) if not (RAW / "admin" / "pse_admin2.geojson").exists() else RAW / "admin"
    a2 = gpd.read_file(next(folder.rglob("pse_admin2.geojson")))
    area = a2[a2["adm2_pcode"] == AREA_PCODE][["adm2_name", "adm2_pcode", "geometry"]]
    assert len(area) == 1, "Khan Younis outline not found"
    out = PROCESSED / "area.geojson"
    area.to_file(out, driver="GeoJSON")
    ha = area.to_crs(METRIC_CRS).area.iloc[0] / 10_000
    print(f"{AREA_NAME} outline saved -> {out}  ({ha:,.0f} ha, bbox {tuple(round(v, 4) for v in area.total_bounds)})")
    return area


def inspect(gdb: Path) -> None:
    """Print every layer, its columns, and the categories in text columns."""
    print(f"\n=== {gdb.name}")
    for name, geom in pyogrio.list_layers(gdb):
        gdf = gpd.read_file(gdb, layer=name)
        print(f"\n--- layer '{name}' ({geom}), {len(gdf)} rows, CRS {gdf.crs}")
        print("columns:", [c for c in gdf.columns if c != "geometry"])
        for c in gdf.columns:
            if c == "geometry":
                continue
            s = gdf[c]
            if s.dtype == object or s.nunique() <= 12:
                print(f"  {c}: {s.value_counts(dropna=False).head(12).to_dict()}")
            else:
                print(f"  {c}: numeric, min {s.min()} max {s.max()}")


def read_label_layer(gdb: Path, layer: str) -> gpd.GeoDataFrame:
    """One UNOSAT snapshot -> GeoDataFrame in metres with a 0/1 'label' column.

    The GDB stores a compound CRS (WGS84 + EGM96 heights); the horizontal part
    is plain WGS84, so we declare EPSG:4326 and then project to UTM 36N.
    """
    gdf = gpd.read_file(gdb, layer=layer)
    gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs(METRIC_CRS)
    gdf["label"] = gdf["Class"].map({"Non-affected cropland": 0, "Damaged cropland": 1})
    assert gdf["label"].notna().all(), f"unexpected classes: {gdf['Class'].unique()}"
    return gdf[["label", "Class", "geometry"]]


def rasterize_labels(gdf: gpd.GeoDataFrame, gbox, area_gdf) -> np.ndarray:
    """Burn the polygons onto the project grid: 0 intact, 1 damaged, 255 not cropland.

    A pixel gets a label only if its CENTRE falls inside the polygon (all_touched=False),
    and only if it is inside the Khan Younis outline.
    """
    shape = (gbox.shape.y, gbox.shape.x)
    lab = features.rasterize(((g, int(v)) for g, v in zip(gdf.geometry, gdf["label"])),
                             out_shape=shape, transform=gbox.transform, fill=NODATA,
                             dtype="uint8", all_touched=False)
    inside = features.rasterize(((g, 1) for g in area_gdf.geometry), out_shape=shape,
                                transform=gbox.transform, fill=0, dtype="uint8")
    lab[inside == 0] = NODATA
    return lab


def build_labels():
    gdb = find_gdb(unzip(download(LABEL_SOURCES["2025-07"], RAW / "labels" / f"{LABEL_GDB_ZIP}.zip")))
    area_gdf = area()
    gbox = grid()
    out_dir = PROCESSED / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, layer in LABEL_LAYERS.items():
        gdf = read_label_layer(gdb, layer)
        # hectares from the vector polygons, clipped to Khan Younis (exact geometry)
        clipped = gpd.clip(gdf, area_gdf)
        ha_vec = (clipped.dissolve("Class").area / 10_000).round(1).to_dict()
        lab = rasterize_labels(gdf, gbox, area_gdf)
        n_int, n_dmg = int((lab == 0).sum()), int((lab == 1).sum())
        # each 10 m pixel = 0.01 ha
        summary[key] = {"layer": layer, "ha_vector": ha_vec,
                        "pixels": {"intact": n_int, "damaged": n_dmg},
                        "ha_raster": {"intact": n_int / 100, "damaged": n_dmg / 100},
                        "damaged_share_pct": round(100 * n_dmg / (n_int + n_dmg), 1)}
        xr.DataArray(lab, dims=("y", "x"), coords={"y": gbox.coords["y"].values, "x": gbox.coords["x"].values}) \
          .rio.write_crs(METRIC_CRS).rio.write_nodata(NODATA) \
          .rio.to_raster(out_dir / f"labels_{key}.tif", compress="deflate")
        print(f"{key}: {summary[key]}")
    (RESULTS / "label_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"saved -> {RESULTS / 'label_summary.json'}")


def main():
    area_outline()
    if "--inspect-only" in sys.argv:
        for key, url in LABEL_SOURCES.items():
            inspect(find_gdb(unzip(download(url, RAW / "labels" / Path(url).name))))
        return
    build_labels()


if __name__ == "__main__":
    main()
