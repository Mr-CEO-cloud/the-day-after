"""
quicklook.py - Phase 1 sanity check picture.

Plain words: before trusting any numbers, LOOK at the data. For one date I draw
  1) the pre-war image (same season, 2022), true colour
  2) the recent image, true colour
  3) the change in greenness (NDVI after - NDVI before), with the UN expert
     labels on top (red = damaged, blue = intact).
If the red outlines sit on land that turned brown, labels and imagery line up.

Run:  .venv/Scripts/python.exe -I src/quicklook.py 2025-07
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED, RESULTS  # noqa: E402
from common import area, NODATA  # noqa: E402

import numpy as np  # noqa: E402
import rioxarray  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]


def load(path):
    return rioxarray.open_rasterio(path).values  # (band, y, x)


def rgb(arr):
    """True-colour picture: red=B04, green=B03, blue=B02, brightened for display."""
    img = np.stack([arr[2], arr[1], arr[0]], axis=-1)
    return np.clip(img / 0.3, 0, 1)


def ndvi(arr):
    return (arr[3] - arr[2]) / (arr[3] + arr[2])


def main(date: str):
    comp = PROCESSED / "composites"
    pre, post = load(comp / f"pre_{date}.tif"), load(comp / f"post_{date}.tif")
    lab = rioxarray.open_rasterio(PROCESSED / "labels" / f"labels_{date}.tif")
    extent = [float(lab.x.min()), float(lab.x.max()), float(lab.y.min()), float(lab.y.max())]
    lab = lab.values[0]
    d_ndvi = ndvi(post) - ndvi(pre)

    fig, ax = plt.subplots(1, 3, figsize=(18, 7.5), constrained_layout=True)
    ax[0].imshow(rgb(pre), extent=extent)
    ax[0].set_title(f"Pre-war, same season (2022)")
    ax[1].imshow(rgb(post), extent=extent)
    ax[1].set_title(f"Recent ({date})")
    im = ax[2].imshow(np.where(lab != NODATA, d_ndvi, np.nan), cmap="RdYlGn", vmin=-0.5, vmax=0.5, extent=extent)
    ax[2].contour(np.flipud(lab == 1).astype(float), levels=[0.5], colors="red", linewidths=0.4,
                  extent=extent, origin="lower")
    ax[2].contour(np.flipud(lab == 0).astype(float), levels=[0.5], colors="blue", linewidths=0.4,
                  extent=extent, origin="lower")
    ax[2].set_title("Greenness change on cropland\nred line = UN 'damaged', blue = 'intact'")
    fig.colorbar(im, ax=ax[2], shrink=0.6, label="NDVI change (after - before)")
    outline = area()
    for a in ax:
        outline.boundary.plot(ax=a, color="black", linewidth=0.8)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Khan Younis, {date}: Sentinel-2 composites + UNOSAT-FAO labels", fontsize=14)
    out = RESULTS / "figures" / f"quicklook_{date}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    # simple numeric check: damaged land should lose more greenness than intact land
    for v, name in [(1, "damaged"), (0, "intact")]:
        print(f"{date} {name:8s}: median NDVI change {np.nanmedian(d_ndvi[lab == v]):+.3f}  ({(lab == v).sum():,} px)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2025-07")
