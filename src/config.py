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

# ---------------------------------------------------------------- model v2: monthly history
MONTHLY_FIRST = (2017, 1)       # first monthly image (year, month)
MONTHLY_LAST = (2026, 10)       # last monthly image
NORMAL_YEARS = list(range(2017, 2023))   # 2017-2022 define what is "normal" for each pixel in each calendar month
CONFLICT_START = (2023, 10)     # "since October 2023" features count from this month
# The 18 months that have a UN damage map (layer names in the geodatabase end with the image date).
LABEL_MONTHS = ["2023-10", "2023-11", "2023-12", "2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
                "2024-07", "2024-08", "2024-09", "2024-10", "2024-12", "2025-02", "2025-03", "2025-04", "2025-07"]

# ---------------------------------------------------------------- split
BLOCK_SIZE_M = 1000      # 1 km blocks for the spatial train/test split
TEST_FRACTION = 0.30     # share of blocks locked away for the final test
BUFFER_M = 100           # training pixels closer than this to a test block are dropped (no neighbours across the line)
N_CV_FOLDS = 5           # training blocks are dealt into 5 groups for cross-validation
PLOT_SIZE_M = 100        # planner unit: 1 ha plots (100 m cells); 10 x 10 plots fit exactly in one block
PLOT_MIN_CROPLAND_PX = 20             # a plot needs >= 0.2 ha of cropland to be rated (fewer pixels = too noisy)
PLOT_STATUS_CUTS = (0.25, 0.75)       # damaged share of a plot: < 25% intact, 25-75% partly, >= 75% mostly damaged
PLOT_STATUS_NAMES = ["intact", "partly damaged", "mostly damaged"]
N_BOOTSTRAP = 2000                    # resamples of whole blocks for the error bars on test scores

# ---------------------------------------------------------------- features (built by 03_features.py)
# Groups let Phase 3/4 test which information actually helps (an "ablation").
_BANDS = ("B02", "B03", "B04", "B08", "B11", "B12")
FEATURES_NDVI = ["post_ndvi", "pre_ndvi", "d_ndvi"]                          # greenness only (what the baseline sees)
FEATURES_SPECTRAL = (
    [f"post_{b}" for b in _BANDS]                                            # the 6 bands, recent image
    + [f"pre_{b}" for b in _BANDS]                                           # the 6 bands, same season in 2022
    + ["post_ndmi", "pre_ndmi", "d_ndmi", "post_bsi", "pre_bsi", "d_bsi"]    # moisture and bare-soil indices
)
FEATURES_CONTEXT = ["prewar_peak_mean", "prewar_peak_std", "season_min", "season_amp", "wc_tree", "wc_built"]
FEATURES_ALL = FEATURES_NDVI + FEATURES_SPECTRAL + FEATURES_CONTEXT
# Never used as features: row, col, block_id, plot_id (a model given positions could memorise the map).

# ---------------------------------------------------------------- model v2 features (built by 03b_features_v2.py)
# "an_" = anomaly = value minus the pixel's own normal for that calendar month (normal = 2017-2022 average).
# "z_"  = the anomaly divided by how much that month normally varies from year to year.
V2_APP_MONTHS = ["2025-11", "2026-04", "2026-10"]          # months with no UN map that the app shows
V2_NOW = ([f"now_{b}" for b in _BANDS]                                         # the 6 bands this month
          + ["now_ndvi", "now_ndmi", "now_bsi", "an_ndvi", "an_ndmi", "an_bsi", "z_ndvi",
             "norm_ndvi", "norm_ndvi_std"])                                    # what is normal for this calendar month
V2_LAGS = ([f"an_ndvi_lag{i}" for i in range(1, 7)]                            # greenness anomaly 1..6 months ago
           + [f"an_ndmi_lag{i}" for i in (1, 2, 3)] + [f"an_bsi_lag{i}" for i in (1, 2, 3)])
V2_WINDOWS = ([f"{s}_{w}" for w in (3, 6, 12)                                  # the last 3 / 6 / 12 months, this month included
               for s in ("ndvi_min", "ndvi_mean", "z_min", "bsi_max", "ndmi_min")]
              + ["ndvi_max_12", "peak_gap_12"])                                # greenest moment of the last year, and how far below the normal yearly peak
V2_WAR = ["war_ndvi_min", "war_ndvi_mean", "war_z_min", "war_frac_low",        # everything since October 2023
          "war_since_min", "war_bsi_max", "war_ndmi_min", "war_frac_seen"]
V2_CONTEXT = FEATURES_CONTEXT + ["crop_share", "norm_peak", "norm_trough"]     # pre-war facts about the pixel
V2_NEIGHBOUR = ([f"nb{w}_{s}" for w in (5, 11)                                 # averages over the 50 m / 110 m square around the pixel
                 for s in ("an_ndvi", "war_ndvi_min", "war_ndvi_mean", "ndvi_min_12", "now_ndvi")]
                + ["nb5_sd_an_ndvi"])
FEATURES_V2 = V2_NOW + V2_LAGS + V2_WINDOWS + V2_WAR + V2_CONTEXT              # the core v2 set (neighbourhood is optional)
# Never features in v2 either: position, block, plot, the date, the label, and the UN damaged-cell count.

# ---------------------------------------------------------------- planner
# Priority = pre-war productivity (percentile, 0-1) x status weight.
# Status weight falls in a straight line with the plot's estimated damaged share s (0-1):
#     weight = 1 - DAMAGE_PENALTY * s      ->  intact plot (s = 0): 1.0,  half damaged: 0.7,  fully damaged: 0.4
# There is no "destroyed" class (the UN labels do not measure it), so even a fully damaged plot keeps
# 0.4: a very productive damaged plot can still rank above a poor intact one. This is a judgement call,
# not a measured fact; the app lets the user change it.
DAMAGE_PENALTY = 0.6
STATUS_WEIGHT = {"intact": 1.0, "partly damaged": 0.7, "mostly damaged": 0.4}   # the same line at s = 0, 0.5, 1 (for explaining)
PRODUCTIVITY_LEVELS = (1 / 3, 2 / 3)   # productivity percentile: < 1/3 low, 1/3-2/3 medium, >= 2/3 high
ORCHARD_TREE_SHARE = 0.5               # >= 50% of the plot was trees in ESA WorldCover 2021 -> "orchard"
MIXED_TREE_SHARE = 0.2                 # 20-50% trees -> "mixed trees and crops"; below -> "seasonal crops"
BUILT_SHARE_FLAG = 0.5                 # >= 50% WorldCover "built-up" -> flag (possibly greenhouses or buildings)
LOW_CONFIDENCE_MARGIN = 0.10           # damaged share within 0.10 of a status cut (0.25 / 0.75) -> "uncertain"
GREENER_THAN_2022 = 0.10               # NDVI now minus same month 2022 above this -> warn: may be weeds / regrowth

DISCLAIMER = "This project is a prototype, not agricultural or humanitarian advice."
