"""
config.py - every setting for The Day After, in one place.

Plain words: instead of scattering numbers (dates, areas, weights) through many
scripts, we keep them here. If a judge asks "why this date?" or "why this
weight?", the answer lives in this one file, next to its reason.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]      # the repo folder
RAW = ROOT / "data" / "raw"                     # downloads (not committed)
PROCESSED = ROOT / "data" / "processed"         # cleaned tables / composites
RESULTS = ROOT / "results"                      # everything we report

# ---------------------------------------------------------------- reproducibility
SEED = 42   # fixed random seed, so the same code gives the same numbers

# ---------------------------------------------------------------- area
AREA_NAME = "Khan Younis"
# Rough box around Khan Younis governorate (lon_min, lat_min, lon_max, lat_max), WGS84 degrees.
# APPROXIMATE: Phase 1 replaces it with the official OCHA COD-AB governorate outline.
AREA_BBOX = (34.22, 31.29, 34.40, 31.40)

# Metric projection for measuring areas in hectares (UTM zone 36N covers Gaza).
METRIC_CRS = "EPSG:32636"

# ---------------------------------------------------------------- labels (UNOSAT-FAO, CC BY-SA, via HDX)
LABEL_SOURCES = {
    "2025-07": "https://unosat.org/static/unosat_filesystem/4160/FAO-PAL-015_UNOSAT_Cropland_DA_GDB_July2025.zip",
    "2024-12": "https://unosat.org/static/unosat_filesystem/4072/FAO-PAL-012_UNOSAT_Cropland_DA_GDB_December2024.zip",
}
# What Phase 1 found inside: the July 2025 GDB holds 18 monthly snapshots (Oct 2023 -> Jul 2025),
# each layer = 2 dissolved multipolygons: "Damaged cropland" / "Non-affected cropland".
# => labels are BINARY (no "destroyed" class) and NOT per field. Their union = the cropland mask.
LABEL_GDB_ZIP = "FAO-PAL-015_UNOSAT_Cropland_DA_GDB_July2025"   # superset of the Dec 2024 file
LABEL_LAYERS = {   # our date key -> UNOSAT layer (layer name ends with the imagery date)
    "2024-12": "Cropland_Damage_NDVI_Gaza_DA_2017_2024_20241230",
    "2025-07": "Cropland_Damage_NDVI_Gaza_DA_2017_2025_20250728",
}
CLASSES = ["intact", "damaged"]   # 0 = "Non-affected cropland", 1 = "Damaged cropland"

# ---------------------------------------------------------------- imagery
S2_COLLECTION = "sentinel-2-l2a"
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]   # blue, green, red, NIR, SWIR1, SWIR2, scene class (clouds)
RESOLUTION_M = 10
MAX_CLOUD_PCT = 30   # skip scenes that are mostly cloud

# Each "date" the app offers = a recent composite ("post") compared with the SAME
# season in 2022 ("pre"), so normal seasonal change (e.g. summer vs winter) is not
# mistaken for damage. 2022 is the last full year before October 2023.
# Label-date windows are checked against the GDB metadata in Phase 1.
DATES = {
    # dates with UNOSAT-FAO expert labels (used for training / testing)
    "2024-12": {"post": ("2024-12-01", "2024-12-30"), "pre": ("2022-12-01", "2022-12-30"), "labelled": True},
    "2025-07": {"post": ("2025-07-01", "2025-07-28"), "pre": ("2022-07-01", "2022-07-28"), "labelled": True},
    # later dates with no expert map yet (model-only "Day After" maps)
    "2025-11": {"post": ("2025-11-01", "2025-11-30"), "pre": ("2022-11-01", "2022-11-30"), "labelled": False},
    "2026-04": {"post": ("2026-04-01", "2026-04-30"), "pre": ("2022-04-01", "2022-04-30"), "labelled": False},
    "2026-10": {"post": ("2026-10-01", "2026-10-31"), "pre": ("2022-10-01", "2022-10-31"), "labelled": False},
}
PREWAR_YEARS = list(range(2017, 2023))   # 2017-2022: pre-war productivity
PEAK_SEASON = ("02-01", "04-30")          # Feb-Apr: typical green peak (checked with the 2022 monthly curve)
SEASONALITY_YEAR = 2022                   # monthly NDVI curve -> orchard (steady) vs seasonal crops (up/down)

# Scene Classification (SCL) values we trust: 4 vegetation, 5 bare soil, 6 water, 7 unclassified.
# Everything else (clouds, shadows, cirrus, saturated, no-data) is masked out.
SCL_VALID = [4, 5, 6, 7]

# ---------------------------------------------------------------- split
BLOCK_SIZE_M = 1000      # 1 km blocks for the spatial train/test split
TEST_FRACTION = 0.30

# ---------------------------------------------------------------- planner
STATUS_WEIGHT = {"intact": 1.0, "damaged": 0.6, "destroyed": 0.2}

DISCLAIMER = "This project is a prototype, not agricultural or humanitarian advice."
