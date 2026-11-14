"""
06_planner.py - the Day After planner (now driven by model v2).

Plain words:
- Layer 1 (the AI, model v2) gives every 10 m cropland pixel a calibrated probability of
  having been damaged since October 2023. Layer 2 (this file) turns that into decisions
  about 1-hectare plots, using simple rules anyone can check.
- For each plot (a 100 m x 100 m square with at least 0.2 ha of cropland):
    damaged share   the average damage probability of its pixels
    status          intact (< 25% damaged), partly damaged (25-75%), mostly damaged (>= 75%)
    productivity    how green the plot was at its yearly peak in 2017-2022, as a
                    percentile among all Khan Younis plots (1.0 = the greenest)
    pre-war use     orchard / mixed trees and crops / seasonal crops, from ESA WorldCover 2021
    priority        productivity x (1 - 0.6 x damaged share)
    green now?      this month's greenness compared with the plot's own normal for this month
    reason          one line saying why it ranks where it does
    suggested use   one line from a small rule table (status x pre-war use), plus warnings
- "Damaged" follows the UN's meaning: damaged at some point since October 2023. It does not
  go away when a field turns green again. So the planner adds the "green now?" column: a
  damaged plot that is green again may have been replanted, or may be weeds. Only a field
  visit can tell.
- Where do the pixel probabilities come from?
    * Months with no UN map (Nov 2025, Apr 2026, Oct 2026): the final model answers everywhere.
    * Months with a UN map: test and buffer pixels get the final model's answer (it never
      trained on them). Training pixels get their cross-validation answer, from a model that
      never saw their block. So every pixel on these maps is an "unseen land" prediction.
- The UN share is kept next to our estimate on labelled months, for comparison only. It is
  never used to compute the priority.
- The v1 planner results (results/planner/) were made by the earlier version of this file
  and are kept for the v1-v2 comparison.

Outputs:
  data/processed/maps_v2/prob_<month>.tif    calibrated damage probability per pixel
  data/processed/planner_v2/plots_<month>.csv   every rated plot for all 21 months (for the app)
  results/planner_v2/plots_<month>.csv       the same for the five months the write-up uses
  results/planner_v2/summary.json            hectares by status, counts, top plots, per month
  results/planner_v2/validation.json         does the ranking point to land the UN map calls intact?

Run:  .venv/Scripts/python.exe -I src/06_planner.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (PROCESSED, RESULTS, DATES, METRIC_CRS, RESOLUTION_M, PLOT_SIZE_M,  # noqa: E402
                    PLOT_MIN_CROPLAND_PX, PLOT_STATUS_CUTS, PLOT_STATUS_NAMES, DAMAGE_PENALTY,
                    PRODUCTIVITY_LEVELS, ORCHARD_TREE_SHARE, MIXED_TREE_SHARE, BUILT_SHARE_FLAG,
                    LOW_CONFIDENCE_MARGIN, GREEN_AGAIN_ANOMALY, DISCLAIMER, LABEL_MONTHS, V2_APP_MONTHS)
from common import load_table, NODATA  # noqa: E402
from methods import MODELS_DIR, LABELLED  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from pyproj import Transformer  # noqa: E402

OUT = RESULTS / "planner_v2"
ALL_OUT = PROCESSED / "planner_v2"
MAPS = PROCESSED / "maps_v2"
CVDIR = PROCESSED / "cv_v2"
HA_PER_PX = RESOLUTION_M ** 2 / 10_000
CELL = PLOT_SIZE_M // RESOLUTION_M          # 10 pixels per plot side
MONTHS = LABEL_MONTHS + V2_APP_MONTHS       # 18 months with a UN map + 3 later months


# ------------------------------------------------------------------ 1. pixel probabilities
def pixel_probabilities(month: str, bundle, curve, oof: pd.DataFrame, split: np.ndarray) -> pd.DataFrame:
    """Calibrated damage probability for every cropland pixel of one month (see the docstring above)."""
    a = load_table(month, "all", version="v2")                   # every cropland pixel, no answer-key columns
    raw = bundle["model"].predict_proba(a[bundle["features"]].to_numpy(np.float32))[:, 1]
    source = np.full(len(a), "final model", dtype=object)
    if month in LABEL_MONTHS:
        o = oof[oof["month"] == month].set_index(["row", "col"])["raw"]
        cv_raw = o.reindex(pd.MultiIndex.from_arrays([a["row"], a["col"]])).to_numpy()
        use_cv = np.isfinite(cv_raw)                             # training pixels that are cropland in this month's UN map
        assert (split[a["row"], a["col"]][use_cv] == 1).all()
        raw = np.where(use_cv, cv_raw, raw)
        source[use_cv] = "cross-validation"
    a["prob"] = curve.predict(raw).astype("float32")
    a["prob_source"] = source
    return a


def save_prob_map(a: pd.DataFrame, month: str, profile: dict):
    arr = np.full((profile["height"], profile["width"]), np.nan, np.float32)
    arr[a["row"], a["col"]] = a["prob"]
    MAPS.mkdir(parents=True, exist_ok=True)
    p = dict(profile, dtype="float32", nodata=np.nan, count=1, compress="deflate")
    with rasterio.open(MAPS / f"prob_{month}.tif", "w", **p) as dst:
        dst.write(arr, 1)


# ------------------------------------------------------------------ 2. plots
def prewar_plots(split: np.ndarray, transform) -> pd.DataFrame:
    """Facts about each plot that do not depend on the month: size, place, pre-war productivity and use."""
    a = load_table(MONTHS[0], "all", version="v2")
    g = a.groupby("plot_id")
    F = pd.DataFrame({"cropland_px": g.size(),
                      "prewar_peak_ndvi": g["prewar_peak_mean"].mean(),
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
    # where the plot sits: centre of its 100 m square, as longitude / latitude
    F["plot_row"], F["plot_col"] = F.index // 1000, F.index % 1000
    x = transform.c + (F["plot_col"] * CELL + CELL / 2) * transform.a
    y = transform.f + (F["plot_row"] * CELL + CELL / 2) * transform.e
    F["lon"], F["lat"] = Transformer.from_crs(METRIC_CRS, "EPSG:4326", always_xy=True).transform(x.to_numpy(), y.to_numpy())
    return F


def plot_table(a: pd.DataFrame, un: np.ndarray | None, F: pd.DataFrame) -> pd.DataFrame:
    """One row per rated plot for one month: the fixed pre-war facts plus this month's estimates and rules."""
    g = a.groupby("plot_id")
    T = F.join(pd.DataFrame({"damaged_share": g["prob"].mean(),
                             "ndvi_now": g["now_ndvi"].mean(),
                             "ndvi_vs_normal": g["an_ndvi"].mean()}), how="left")     # this month minus the plot's normal
    assert T["damaged_share"].notna().all(), "every rated plot needs a damage estimate"
    T["damaged_ha_est"] = T["damaged_share"] * T["cropland_ha"]
    if un is not None:   # reference only: the UN share of the same plot (never used for the priority)
        T["un_damaged_share"] = a.assign(u=un[a["row"], a["col"]]).groupby("plot_id")["u"].mean().reindex(T.index)
    s = T["damaged_share"].to_numpy()
    T["status"] = np.array(PLOT_STATUS_NAMES)[np.digitize(s, PLOT_STATUS_CUTS)]
    margin = np.min(np.abs(s[:, None] - np.array(PLOT_STATUS_CUTS)[None, :]), axis=1)
    T["confidence"] = np.where(margin < LOW_CONFIDENCE_MARGIN, "lower", "higher")
    T["green_now"] = np.where(T["ndvi_vs_normal"] >= GREEN_AGAIN_ANOMALY, "at or above normal", "below normal")
    T["status_weight"] = 1 - DAMAGE_PENALTY * T["damaged_share"]
    T["priority"] = T["productivity_pct"] * T["status_weight"]
    T = T.sort_values("priority", ascending=False)
    T["rank"] = np.arange(1, len(T) + 1)
    T["reason"] = [reason(r) for r in T.itertuples()]
    T["suggested_use"] = [suggestion(r) for r in T.itertuples()]
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


def reason(r) -> str:
    top = max(1, int(round(100 * (1 - r.productivity_pct))))
    prod = f"top {top}% for pre-war greenness" if r.productivity == "high" else f"{r.productivity} pre-war greenness"
    return (f"{r.prewar_use.capitalize()}, {prod}; the model estimates {r.damaged_share:.0%} of its "
            f"{r.cropland_ha:.1f} ha damaged since Oct 2023 ({r.status}).")


def suggestion(r) -> str:
    text = USE_RULES[(r.status, r.prewar_use)]
    if r.built_share >= BUILT_SHARE_FLAG:
        text += " Pre-war map shows built-up cover here (possibly greenhouses): check structures."
    if r.status != "intact" and r.green_now == "at or above normal":
        text += " Damaged, but as green as normal this month: could be replanted crops, weeds or regrowth."
    text += " Uncertain estimate: verify on the ground first." if r.confidence == "lower" else " Verify on the ground."
    return text


# ------------------------------------------------------------------ 3. does the ranking point to intact land?
def validation(P: pd.DataFrame, productivity_pct: pd.Series) -> dict:
    """P: one row per pixel with label (UN), cal (v2), cal_v1 (v1), base (the new rule's answer 0/1), plot_id.
    Five ways to rank the plots, compared by how much UN-INTACT cropland lands in the top 10%:
      productivity only   ignore damage (what you would do without any damage map)
      rule                productivity x weight(share the season-history rule calls damaged)
      v1 planner          productivity x weight(v1 calibrated damage share)
      v2 planner          productivity x weight(v2 calibrated damage share)   <- what the app does
      UN map (oracle)     productivity x weight(UN damaged share)             <- best possible, needs the UN map"""
    g = P.groupby("plot_id")
    T = pd.DataFrame({"n": g.size(), "un": g["label"].mean(), "v2": g["cal"].mean(), "v1": g["cal_v1"].mean(),
                      "base": g["base"].mean()})
    T = T[T["n"] >= PLOT_MIN_CROPLAND_PX].join(productivity_pct.rename("prod"), how="inner")
    w = lambda s: 1 - DAMAGE_PENALTY * s  # noqa: E731
    rankings = {"productivity only": T["prod"], "rule (worst month since Oct 2023)": T["prod"] * w(T["base"]),
                "v1 planner": T["prod"] * w(T["v1"]), "v2 planner": T["prod"] * w(T["v2"]),
                "UN map (oracle)": T["prod"] * w(T["un"])}
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
    name = json.loads((RESULTS / "v2" / "final_models.json").read_text())["chosen"]
    bundle = joblib.load(MODELS_DIR / f"v2_{name}.joblib")
    curve = joblib.load(MODELS_DIR / f"v2_{name}_calibrator.joblib")
    meta = np.load(CVDIR / "meta.npz", allow_pickle=True)
    oof = pd.DataFrame({"row": meta["row"], "col": meta["col"], "month": meta["month"], "label": meta["label"],
                        "plot_id": meta["plot_id"], "raw": np.load(CVDIR / f"oof_{name}.npy").astype(np.float32),
                        "base": np.load(CVDIR / "oof_rule_war.npy").astype(np.float32)})
    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        split, profile, transform = src.read(1), src.profile, src.transform
    OUT.mkdir(parents=True, exist_ok=True)
    ALL_OUT.mkdir(parents=True, exist_ok=True)
    F = prewar_plots(split, transform)
    summary = {"disclaimer": DISCLAIMER, "model": f"v2 ({name})", "damage_penalty": DAMAGE_PENALTY, "months": {}}
    cols = ["rank", "priority", "lon", "lat", "plot_row", "plot_col", "cropland_ha", "status", "damaged_share",
            "damaged_ha_est", "confidence", "green_now", "ndvi_now", "ndvi_vs_normal", "productivity",
            "productivity_pct", "prewar_peak_ndvi", "prewar_use", "tree_share", "built_share", "dry_season_ndvi",
            "status_weight", "reason", "suggested_use", "land", "block_id"]
    for month in MONTHS:
        a = pixel_probabilities(month, bundle, curve, oof, split)
        save_prob_map(a, month, profile)
        un = None
        if month in LABEL_MONTHS:
            with rasterio.open(PROCESSED / "labels" / "monthly" / f"labels_{month}.tif") as src:
                un = src.read(1).astype("float32")
            un[un == NODATA] = np.nan
        T = plot_table(a, un, F)
        table = T[cols + (["un_damaged_share"] if un is not None else [])].round(4)
        table.to_csv(ALL_OUT / f"plots_{month}.csv", index_label="plot_id")
        if month in DATES:
            table.to_csv(OUT / f"plots_{month}.csv", index_label="plot_id")
        by = T.groupby("status")
        dmg = T["status"] != "intact"
        summary["months"][month] = {
            "has_un_map": month in LABEL_MONTHS, "n_plots": int(len(T)),
            "cropland_ha_all_pixels": round(len(a) * HA_PER_PX, 1),
            "damaged_ha_est_all_pixels": round(float(a["prob"].sum()) * HA_PER_PX, 1),
            "pixel_sources": a["prob_source"].value_counts().to_dict(),
            "plots_by_status": {k: int(v) for k, v in by.size().items()},
            "cropland_ha_by_status": {k: round(float(v), 1) for k, v in by["cropland_ha"].sum().items()},
            "lower_confidence_plots": int((T["confidence"] == "lower").sum()),
            "damaged_plots_green_at_or_above_normal": int((dmg & (T["green_now"] == "at or above normal")).sum()),
            "median_ndvi_vs_normal": round(float(T["ndvi_vs_normal"].median()), 4),
            "top10": T.head(10)[["rank", "lon", "lat", "cropland_ha", "status", "damaged_share", "productivity",
                                 "prewar_use", "green_now", "reason", "suggested_use"]].round(4).to_dict("records"),
        }
        if un is not None:
            summary["months"][month]["damaged_ha_un_all_pixels"] = round(float(np.nansum(un)) * HA_PER_PX, 1)
        e = summary["months"][month]
        print(f"{month}: est. damaged {e['damaged_ha_est_all_pixels']:,.0f} ha"
              + (f" (UN {e['damaged_ha_un_all_pixels']:,.0f})" if un is not None else " (no UN map)")
              + ", plots " + ", ".join(f"{k} {v}" for k, v in e["plots_by_status"].items())
              + f", damaged-but-green {e['damaged_plots_green_at_or_above_normal']}", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))

    # validation on the two exam months: training land (cross-validation answers) and test land (saved answers)
    val = {"note": "Planner rules were fixed before these checks; nothing was tuned on them.", "train_cv": {}, "test": {}}
    with np.load(PROCESSED / "test_v2" / "predictions.npz", allow_pickle=True) as z:
        test = pd.DataFrame({"date": z["date"], "plot_id": z["plot_id"], "label": z["label"], "cal": z["cal"],
                             "cal_v1": z["cal_v1"], "base": z["pred__rule_war"]})
    oof["cal"] = curve.predict(oof["raw"])
    with np.load(PROCESSED / "cv" / "oof.npz", allow_pickle=True) as z:      # v1 cross-validation answers
        v1 = pd.DataFrame({"row": z["row"], "col": z["col"], "month": z["date"], "raw_v1": z["rf_all#p"]})
    v1["cal_v1"] = joblib.load(MODELS_DIR / "rf_all_calibrator.joblib").predict(v1["raw_v1"])
    train = oof.merge(v1[["row", "col", "month", "cal_v1"]], how="inner", on=["row", "col", "month"])
    for month in LABELLED:
        val["train_cv"][month] = validation(train[train["month"] == month], F["productivity_pct"])
        val["test"][month] = validation(test[test["date"] == month], F["productivity_pct"])
    (OUT / "validation.json").write_text(json.dumps(val, indent=1))
    for part in ("train_cv", "test"):
        for month, v in val[part].items():
            print(part, month, f"top {v['top_k']} of {v['n_plots']} plots; all plots UN-intact {v['all_plots_un_intact_share']:.0%} |",
                  " | ".join(f"{k}: {v[k]['un_intact_share_of_top']:.0%}" for k in
                             ("productivity only", "rule (worst month since Oct 2023)", "v1 planner", "v2 planner", "UN map (oracle)")))


if __name__ == "__main__":
    main()
