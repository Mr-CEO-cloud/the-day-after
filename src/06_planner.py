"""
06_planner.py - Phase 5: the Day After planner.

Plain words:
- Layer 1 (the AI) gives every 10 m cropland pixel a calibrated probability of
  being damaged. Layer 2 (this file) turns that into decisions about 1-hectare
  plots, using simple rules anyone can check.
- For each plot (a 100 m x 100 m square with at least 0.2 ha of cropland):
    damaged share   the average damage probability of its pixels (on test land this
                    was within 0.11-0.14 of the UN share on average, with almost no bias)
    status          intact (< 25% damaged), partly damaged (25-75%), mostly damaged (>= 75%)
    productivity    how green the plot was at its yearly peak in 2017-2022, as a
                    percentile among all Khan Younis plots (1.0 = the greenest)
    pre-war use     orchard / mixed trees and crops / seasonal crops, from ESA WorldCover 2021
    priority        productivity x (1 - 0.6 x damaged share)
    reason          one line saying why it ranks where it does
    suggested use   one line from a small rule table (status x pre-war use)
- Where do the pixel probabilities come from?
    * Dates with no UN map (Nov 2025, Apr 2026, Oct 2026): the final model, trained on
      training land, answers for every pixel.
    * Dates with a UN map (Dec 2024, Jul 2025): test and buffer pixels get the final
      model's answer (it never trained on them). Training pixels get their
      cross-validation answer, from a model that never saw their block. So every
      pixel on these maps is an "unseen land" prediction, never a memorised one.
- The UN share is kept next to our estimate on labelled dates, for comparison only.
  It is never used to compute the priority.

Outputs:
  data/processed/maps/prob_<date>.tif   calibrated damage probability per pixel (for the app maps)
  results/planner/plots_<date>.csv      every rated plot, best priority first
  results/planner/summary.json          hectares by status, counts, top plots, per date
  results/planner/validation.json       does the ranking point to land the UN map calls intact?

Run:  .venv/Scripts/python.exe -I src/06_planner.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (PROCESSED, RESULTS, DATES, METRIC_CRS, RESOLUTION_M, PLOT_SIZE_M,  # noqa: E402
                    PLOT_MIN_CROPLAND_PX, PLOT_STATUS_CUTS, PLOT_STATUS_NAMES, DAMAGE_PENALTY,
                    PRODUCTIVITY_LEVELS, ORCHARD_TREE_SHARE, MIXED_TREE_SHARE, BUILT_SHARE_FLAG,
                    LOW_CONFIDENCE_MARGIN, GREENER_THAN_2022, DISCLAIMER, FEATURES_ALL)
from common import load_table, NODATA  # noqa: E402
from methods import MODELS_DIR, MAIN_MODEL, MAIN_BASELINE  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402

OUT = RESULTS / "planner"
MAPS = PROCESSED / "maps"
HA_PER_PX = RESOLUTION_M ** 2 / 10_000
CELL = PLOT_SIZE_M // RESOLUTION_M          # 10 pixels per plot side


# ------------------------------------------------------------------ 1. pixel probabilities
def pixel_probabilities(date: str, model, curve, oof: pd.DataFrame, split: np.ndarray) -> pd.DataFrame:
    """Calibrated damage probability for every cropland pixel of one date (see the docstring above)."""
    a = load_table(date, "all")                                   # every cropland pixel, no label column
    a = a[np.isfinite(a[FEATURES_ALL]).all(axis=1)].reset_index(drop=True)   # drop the few always-cloudy pixels
    raw = model.proba(a)
    source = np.full(len(a), "final model", dtype=object)
    if DATES[date]["labelled"]:
        o = oof[oof["date"] == date].set_index(["row", "col"])["raw"]
        key = pd.MultiIndex.from_arrays([a["row"], a["col"]])
        is_train = split[a["row"], a["col"]] == 1
        cv_raw = o.reindex(key).to_numpy()
        assert np.isfinite(cv_raw[is_train]).all(), "every training pixel needs a cross-validation answer"
        raw = np.where(is_train, cv_raw, raw)
        source[is_train] = "cross-validation"
    a["prob"] = curve.predict(raw).astype("float32")
    a["prob_source"] = source
    return a


def save_prob_map(a: pd.DataFrame, date: str, profile: dict):
    arr = np.full((profile["height"], profile["width"]), np.nan, np.float32)
    arr[a["row"], a["col"]] = a["prob"]
    MAPS.mkdir(parents=True, exist_ok=True)
    p = dict(profile, dtype="float32", nodata=np.nan, count=1, compress="deflate")
    with rasterio.open(MAPS / f"prob_{date}.tif", "w", **p) as dst:
        dst.write(arr, 1)


# ------------------------------------------------------------------ 2. plots
def prewar_plots(split: np.ndarray) -> pd.DataFrame:
    """Facts about each plot that do not depend on the date: size, pre-war productivity and use.
    Computed once from ALL cropland pixels, so every date ranks the same plots with the same productivity."""
    a = load_table(next(iter(DATES)), "all")
    g = a.groupby("plot_id")
    F = pd.DataFrame({"cropland_px": g.size(),
                      "prewar_peak_ndvi": g["prewar_peak_mean"].mean(),    # pandas skips the few missing values
                      "dry_season_ndvi": g["season_min"].mean(),           # greenness in the brownest month of 2022
                      "tree_share": g["wc_tree"].mean(),
                      "built_share": g["wc_built"].mean(),
                      "block_id": g["block_id"].first()})
    F["land"] = a.assign(s=split[a["row"], a["col"]]).groupby("plot_id")["s"].agg(   # train / test / buffer (majority)
        lambda v: {1: "train", 2: "test", 3: "buffer"}.get(int(np.bincount(v).argmax()), "other"))
    F = F[F["cropland_px"] >= PLOT_MIN_CROPLAND_PX].copy()
    F["cropland_ha"] = F["cropland_px"] * HA_PER_PX
    # productivity: percentile of the pre-war peak greenness among all rated plots (1.0 = the greenest)
    F["productivity_pct"] = F["prewar_peak_ndvi"].rank(pct=True)
    F["productivity"] = np.array(["low", "medium", "high"])[np.digitize(F["productivity_pct"], PRODUCTIVITY_LEVELS)]
    F["prewar_use"] = np.select([F["tree_share"] >= ORCHARD_TREE_SHARE, F["tree_share"] >= MIXED_TREE_SHARE],
                                ["orchard", "mixed trees and crops"], "seasonal crops")
    return F


def plot_table(a: pd.DataFrame, date: str, un: np.ndarray | None, transform, F: pd.DataFrame) -> pd.DataFrame:
    """One row per rated plot for one date: the fixed pre-war facts plus this date's damage estimate."""
    g = a.groupby("plot_id")
    T = F.join(pd.DataFrame({"damaged_share": g["prob"].mean(),
                             "ndvi_now": g["post_ndvi"].mean(),
                             "ndvi_change_vs_2022": g["d_ndvi"].mean()}), how="left")
    assert T["damaged_share"].notna().all(), "every rated plot needs a damage estimate"
    T["damaged_ha_est"] = T["damaged_share"] * T["cropland_ha"]
    # where the plot sits: centre of its 100 m square, in metres (UTM 36N) and in longitude / latitude
    prow, pcol = T.index // 1000, T.index % 1000
    x = transform.c + (pcol * CELL + CELL / 2) * transform.a
    y = transform.f + (prow * CELL + CELL / 2) * transform.e
    T["lon"], T["lat"] = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True).transform(x.to_numpy(), y.to_numpy())
    T["plot_row"], T["plot_col"] = prow, pcol
    if un is not None:   # reference only: the UN share of the same plot (never used for the priority)
        T["un_damaged_share"] = a.assign(u=un[a["row"], a["col"]]).groupby("plot_id")["u"].mean().reindex(T.index)
    return apply_rules(T, date)


def apply_rules(T: pd.DataFrame, date: str) -> pd.DataFrame:
    """The transparent rules: status, productivity, pre-war use, priority, reason, suggested use."""
    T = T.copy()
    s = T["damaged_share"].to_numpy()
    T["status"] = np.array(PLOT_STATUS_NAMES)[np.digitize(s, PLOT_STATUS_CUTS)]
    margin = np.min(np.abs(s[:, None] - np.array(PLOT_STATUS_CUTS)[None, :]), axis=1)
    T["confidence"] = np.where(margin < LOW_CONFIDENCE_MARGIN, "lower", "higher")
    T["status_weight"] = 1 - DAMAGE_PENALTY * T["damaged_share"]
    T["priority"] = T["productivity_pct"] * T["status_weight"]
    T = T.sort_values("priority", ascending=False)
    T["rank"] = np.arange(1, len(T) + 1)
    T["reason"] = [reason(r, date) for r in T.itertuples()]
    T["suggested_use"] = [suggestion(r, date) for r in T.itertuples()]
    return T


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


def reason(r, date) -> str:
    top = max(1, int(round(100 * (1 - r.productivity_pct))))
    prod = f"top {top}% for pre-war greenness" if r.productivity == "high" else f"{r.productivity} pre-war greenness"
    text = (f"{r.prewar_use.capitalize()}, {prod}; the model estimates {r.damaged_share:.0%} of its "
            f"{r.cropland_ha:.1f} ha damaged ({r.status}).")
    return text


def suggestion(r, date) -> str:
    text = USE_RULES[(r.status, r.prewar_use)]
    if r.built_share >= BUILT_SHARE_FLAG:
        text += " Pre-war map shows built-up cover here (possibly greenhouses): check structures."
    if not DATES[date]["labelled"] and r.ndvi_change_vs_2022 > GREENER_THAN_2022:
        text += " Greener than in 2022: this may be weeds or regrowth, not active farming."
    if r.confidence == "lower":
        text += " Uncertain estimate: verify on the ground first."
    else:
        text += " Verify on the ground."
    return text


# ------------------------------------------------------------------ 3. does the ranking point to intact land?
def validation(P: pd.DataFrame, productivity_pct: pd.Series) -> dict:
    """P: one row per pixel with label (UN), cal (our calibrated prob), base (NDVI-drop answer 0/1), plot_id.
    Four ways to rank the plots, compared by how much UN-INTACT cropland lands in the top 10%:
      productivity only   ignore damage (what you would do without any damage map)
      baseline            productivity x weight(share the NDVI-drop rule calls damaged)
      our planner         productivity x weight(calibrated damage share)    <- what the app does
      UN map (oracle)     productivity x weight(UN damaged share)           <- best possible, needs the UN map"""
    g = P.groupby("plot_id")
    T = pd.DataFrame({"n": g.size(), "un": g["label"].mean(), "ours": g["cal"].mean(), "base": g["base"].mean()})
    T = T[T["n"] >= PLOT_MIN_CROPLAND_PX].join(productivity_pct.rename("prod"), how="inner")
    w = lambda s: 1 - DAMAGE_PENALTY * s  # noqa: E731
    rankings = {"productivity only": T["prod"], "baseline (NDVI-drop)": T["prod"] * w(T["base"]),
                "our planner": T["prod"] * w(T["ours"]), "UN map (oracle)": T["prod"] * w(T["un"])}
    k = max(1, len(T) // 10)
    out = {"n_plots": int(len(T)), "top_k": int(k),
           "all_plots_un_intact_share": round(float(1 - (T["un"] * T["n"]).sum() / T["n"].sum()), 4)}
    for name, score in rankings.items():
        top = T.loc[score.sort_values(ascending=False).index[:k]]
        out[name] = {"un_intact_share_of_top": round(float(1 - (top["un"] * top["n"]).sum() / top["n"].sum()), 4),
                     "un_intact_ha_in_top": round(float(((1 - top["un"]) * top["n"]).sum() * HA_PER_PX), 1),
                     "mean_prewar_productivity_pct_of_top": round(float(top["prod"].mean()), 4),
                     "spearman_with_oracle": round(float(score.corr(rankings["UN map (oracle)"], method="spearman")), 4)}
    return out


# ------------------------------------------------------------------ main
def main():
    import joblib
    model = joblib.load(MODELS_DIR / f"{MAIN_MODEL}.joblib")
    curve = joblib.load(MODELS_DIR / f"{MAIN_MODEL}_calibrator.joblib")
    with np.load(PROCESSED / "cv" / "oof.npz", allow_pickle=True) as z:
        oof = pd.DataFrame({"row": z["row"], "col": z["col"], "date": z["date"], "raw": z[MAIN_MODEL + "#p"],
                            "label": z["label"], "base": z[MAIN_BASELINE]})
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        split, profile, transform = src.read(1), src.profile, src.transform
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"disclaimer": DISCLAIMER, "damage_penalty": DAMAGE_PENALTY, "dates": {}}
    F = prewar_plots(split)
    prod_ref = F["productivity_pct"]
    for date in DATES:
        a = pixel_probabilities(date, model, curve, oof, split)
        save_prob_map(a, date, profile)
        un = None
        if DATES[date]["labelled"]:
            with rasterio.open(PROCESSED / "labels" / f"labels_{date}.tif") as src:
                un = src.read(1).astype("float32")
            un[un == NODATA] = np.nan
        T = plot_table(a, date, un, transform, F)
        cols = ["rank", "priority", "lon", "lat", "plot_row", "plot_col", "cropland_ha", "status", "damaged_share",
                "damaged_ha_est", "confidence", "productivity", "productivity_pct", "prewar_peak_ndvi", "prewar_use",
                "tree_share", "built_share", "dry_season_ndvi", "ndvi_now", "ndvi_change_vs_2022", "status_weight",
                "reason", "suggested_use", "land", "block_id"] + (["un_damaged_share"] if un is not None else [])
        T[cols].round(4).to_csv(OUT / f"plots_{date}.csv", index_label="plot_id")
        by = T.groupby("status")
        summary["dates"][date] = {
            "labelled": DATES[date]["labelled"], "n_plots": int(len(T)),
            "cropland_ha_rated": round(float(T["cropland_ha"].sum()), 1),
            "damaged_ha_est_all_pixels": round(float(a["prob"].sum()) * HA_PER_PX, 1),
            "cropland_ha_all_pixels": round(len(a) * HA_PER_PX, 1),
            "pixel_sources": a["prob_source"].value_counts().to_dict(),
            "plots_by_status": {k: int(v) for k, v in by.size().items()},
            "cropland_ha_by_status": {k: round(float(v), 1) for k, v in by["cropland_ha"].sum().items()},
            "lower_confidence_plots": int((T["confidence"] == "lower").sum()),
            "plots_greener_than_2022": int((T["ndvi_change_vs_2022"] > GREENER_THAN_2022).sum()),
            "top10": T.head(10)[["rank", "lon", "lat", "cropland_ha", "status", "damaged_share", "productivity",
                                 "prewar_use", "reason", "suggested_use"]].round(4).to_dict("records"),
        }
        if un is not None:
            summary["dates"][date]["damaged_ha_un_all_pixels"] = round(float(np.nansum(un)) * HA_PER_PX, 1)
        print(f"{date}: {len(T):,} plots, " + ", ".join(f"{k} {v}" for k, v in summary["dates"][date]["plots_by_status"].items()))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    # validation: rehearsal on training land (cross-validation answers), then the test land
    val = {"note": "Planner rules were fixed before these checks; nothing was tuned on them.", "train_cv": {}, "test": {}}
    with np.load(PROCESSED / "test" / "predictions.npz", allow_pickle=True) as z:
        test = pd.DataFrame({"date": z["date"], "plot_id": z["plot_id"], "label": z["label"], "cal": z["cal"],
                             "base": z["pred__" + MAIN_BASELINE]})
    oof["cal"] = curve.predict(oof["raw"])
    a0 = load_table(next(iter(DATES)), "all")[["row", "col", "plot_id"]]
    oof = oof.merge(a0, on=["row", "col"], how="left")
    for date in [d for d, w in DATES.items() if w["labelled"]]:
        val["train_cv"][date] = validation(oof[oof["date"] == date], prod_ref)
        val["test"][date] = validation(test[test["date"] == date], prod_ref)
    # the trade-off behind DAMAGE_PENALTY, on training land only: more penalty = safer but less productive top 10%
    val["penalty_tradeoff_train_cv"] = {}
    for date in [d for d, w in DATES.items() if w["labelled"]]:
        P = oof[oof["date"] == date]
        g = P.groupby("plot_id")
        T = pd.DataFrame({"n": g.size(), "un": g["label"].mean(), "ours": g["cal"].mean()})
        T = T[T["n"] >= PLOT_MIN_CROPLAND_PX].join(prod_ref.rename("prod"), how="inner")
        k = max(1, len(T) // 10)
        rows = []
        for pen in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            top = T.loc[(T["prod"] * (1 - pen * T["ours"])).sort_values(ascending=False).index[:k]]
            rows.append({"damage_penalty": pen,
                         "un_intact_share_of_top": round(float(1 - (top["un"] * top["n"]).sum() / top["n"].sum()), 4),
                         "mean_productivity_pct_of_top": round(float(top["prod"].mean()), 4)})
        val["penalty_tradeoff_train_cv"][date] = rows
    (OUT / "validation.json").write_text(json.dumps(val, indent=1))
    for date, rows in val["penalty_tradeoff_train_cv"].items():
        print("trade-off (train)", date, [(r["damage_penalty"], r["un_intact_share_of_top"], r["mean_productivity_pct_of_top"]) for r in rows])
    for part in ("train_cv", "test"):
        for date, v in val[part].items():
            print(part, date, f"top {v['top_k']} of {v['n_plots']} plots; all plots UN-intact {v['all_plots_un_intact_share']:.0%} |",
                  " | ".join(f"{k}: {v[k]['un_intact_share_of_top']:.0%} intact, rho {v[k]['spearman_with_oracle']:.2f}"
                             for k in ("productivity only", "baseline (NDVI-drop)", "our planner", "UN map (oracle)")))


if __name__ == "__main__":
    main()
