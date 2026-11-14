"""
planner_rules.py - the Day After planner's rules, in one small file the app can read on its own.

Plain words: these are the same rules as src/06_planner.py (status cut-offs, the one-line
reason, the suggested first step). They are copied here so the app folder is self-contained
and can be uploaded to Hugging Face by itself. tests/test_app.py checks that the text
produced here is identical to the planner's own output, for every plot and every month.
"""
import numpy as np

STATUS_NAMES = ["intact", "partly damaged", "mostly damaged"]
STATUS_CUTS = (0.25, 0.75)          # damaged share: < 25% intact, 25-75% partly, >= 75% mostly
LOW_CONFIDENCE_MARGIN = 0.10        # within 0.10 of a cut -> "lower" confidence
PRODUCTIVITY_LEVELS = (1 / 3, 2 / 3)
BUILT_SHARE_FLAG = 0.5
DEFAULT_PENALTY = 0.6               # priority = productivity x (1 - penalty x damaged share)
USE_NAMES = ["seasonal crops", "mixed trees and crops", "orchard"]   # codes 0, 1, 2
CONFIDENCE_NAMES = ["higher", "lower"]
GREEN_NAMES = ["below normal", "at or above normal"]

USE_RULES = {   # (status, pre-war use) -> suggested first step. Planning suggestions, not agronomic advice.
    ("intact", "orchard"): "Keep the trees producing: restore water and tree care first.",
    ("intact", "mixed trees and crops"): "Restore water; keep the trees and plant the open parts next season.",
    ("intact", "seasonal crops"): "Ready for the next planting season: seeds, fertiliser and water.",
    ("partly damaged", "orchard"): "Survey which trees survived; repair irrigation and replant gaps.",
    ("partly damaged", "mixed trees and crops"): "Plant the undamaged parts first; survey trees and clear the rest.",
    ("partly damaged", "seasonal crops"): "Plant the undamaged parts first; clear and check the rest.",
    ("mostly damaged", "orchard"): "Longer term: clear and check the land, then replant trees.",
    ("mostly damaged", "mixed trees and crops"): "Longer term: clear and check soil and safety before planting.",
    ("mostly damaged", "seasonal crops"): "Clear and check soil and safety first, then plant seasonal crops.",
}


def status_of(share):
    """0 intact, 1 partly damaged, 2 mostly damaged."""
    return np.digitize(share, STATUS_CUTS)


def productivity_label(pct) -> str:
    return ["low", "medium", "high"][int(np.digitize(pct, PRODUCTIVITY_LEVELS))]


def reason(use: str, productivity_pct: float, damaged_share: float, cropland_ha: float, status: str) -> str:
    level = productivity_label(productivity_pct)
    top = max(1, int(round(100 * (1 - productivity_pct))))
    prod = f"top {top}% for pre-war greenness" if level == "high" else f"{level} pre-war greenness"
    return (f"{use.capitalize()}, {prod}; the model estimates {damaged_share:.0%} of its "
            f"{cropland_ha:.1f} ha damaged since Oct 2023 ({status}).")


def suggestion(status: str, use: str, built_share: float, green_now: str, confidence: str) -> str:
    text = USE_RULES[(status, use)]
    if built_share >= BUILT_SHARE_FLAG:
        text += " Pre-war map shows built-up cover here (possibly greenhouses): check structures."
    if status != "intact" and green_now == "at or above normal":
        text += " Damaged, but as green as normal this month: could be replanted crops, weeds or regrowth."
    text += " Uncertain estimate: verify on the ground first." if confidence == "lower" else " Verify on the ground."
    return text


def priority(productivity_pct, damaged_share, penalty: float):
    return productivity_pct * (1 - penalty * damaged_share)
