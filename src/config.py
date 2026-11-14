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

# ---------------------------------------------------------------- imagery
S2_COLLECTION = "sentinel-2-l2a"
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]   # blue, green, red, NIR, SWIR1, SWIR2, scene class (clouds)
RESOLUTION_M = 10
MAX_CLOUD_PCT = 30   # skip scenes that are mostly cloud

# Date windows for composites (start, end). Label-date windows are checked against
# the GDB metadata in Phase 1 and corrected here if needed.
WINDOWS = {
    "prewar_2022_dec": ("2021-11-15", "2021-12-31"),   # same season as the Dec 2024 labels, before the war
    "prewar_2022_jul": ("2022-06-15", "2022-07-31"),   # same season as the Jul 2025 labels, before the war
    "label_2024_12":   ("2024-11-15", "2024-12-31"),
    "label_2025_07":   ("2025-07-01", "2025-07-28"),
}
PREWAR_YEARS = list(range(2017, 2023))   # 2017-2022: pre-war productivity
PEAK_SEASON = ("02-01", "04-30")          # Feb-Apr: typical green peak (to be checked with the data)

# ---------------------------------------------------------------- split
BLOCK_SIZE_M = 1000      # 1 km blocks for the spatial train/test split
TEST_FRACTION = 0.30

# ---------------------------------------------------------------- planner
STATUS_WEIGHT = {"intact": 1.0, "damaged": 0.6, "destroyed": 0.2}

DISCLAIMER = "This project is a prototype, not agricultural or humanitarian advice."
