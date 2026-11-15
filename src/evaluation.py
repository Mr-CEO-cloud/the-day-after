"""
evaluation.py - every analysis I run on a table of predictions.

Plain words: the practice exam (training land, cross-validation) and the real
exam (the locked test blocks) both end with a table that has, for every pixel,
the UN label and each method's answer. This file turns such a table into the
numbers and pictures I report. Because both exams go through the SAME code,
the analysis was written and rehearsed on training land before the test set
was opened, and nothing here can be tweaked to fit the test.

The table ("P") has one row per pixel per date with:
  row, col, date, label, block_id, plot_id, a few features,
  pred__<method>   each method's answer for the same-date exam (0 intact, 1 damaged)
  time__<method>   each method's answer for the time test (-1 where not part of it)
  score            the main model's raw "damaged" score (0-1)
  cal              the same score after calibration (see calibrate() below)

What full_report() produces (into one folder):
  scores.csv        every method x exam: accuracy, precision/recall/F1 per class, macro F1,
                    confusion-matrix counts, and a 95% interval from a block bootstrap
  gap.json          main model minus main baseline, with its 95% interval
  plot_level.json   the same question at the planner's unit: 1 ha plots
  area.json         damaged hectares: UN map vs each method
  calibration.csv   does a score of 0.7 really mean "70% of such pixels are damaged"?
  subgroups.csv     error rates by kind of land (orchard / crop, edge / interior, ...)
  failures.json     the places with the most mistakes (chosen by a fixed rule, not by hand)
  figures           scores.png, confusion.png, failures.png
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ROOT, PROCESSED, RESULTS, SEED, CLASSES, RESOLUTION_M, PLOT_MIN_CROPLAND_PX,  # noqa: E402
                    PLOT_STATUS_CUTS, PLOT_STATUS_NAMES, N_BOOTSTRAP)
from common import NODATA, read_labels, load_table  # noqa: E402
from methods import (LABELLED, TIME_CHECK, ORDER, MAIN_MODEL, MAIN_BASELINE, EXAMS,  # noqa: E402
                     scores_from_cm, scores)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from scipy import ndimage  # noqa: E402

GAP_PAIRS = [(MAIN_MODEL, MAIN_BASELINE)]       # (model, rival) pairs whose score gap gets a paired interval
# ORDER, MAIN_MODEL, MAIN_BASELINE and GAP_PAIRS describe the v1 report by default; full_report() can
# swap them for the v2 report, so both versions go through exactly the same analysis code.

HA_PER_PX = RESOLUTION_M ** 2 / 10_000          # one 10 m pixel = 0.01 hectare
WINDOW_PX, STRIDE_PX, WINDOW_MIN_PX = 40, 20, 300   # failure search: 400 m windows, moved in 200 m steps
WC_NAMES = {10: "trees (orchard)", 40: "cropland", 50: "built-up / greenhouses"}


# ------------------------------------------------------------------ helpers
def exam_rows(P, exam):
    """Which rows belong to an exam, and which prediction columns to read."""
    if exam.startswith("same-date"):
        return (P["date"] == exam.split()[-1]).to_numpy(), "pred__"
    return (P["date"] == TIME_CHECK).to_numpy(), "time__"


def macro_f1_from_counts(S):
    """Macro F1 for many confusion matrices at once. S[..., 4] = tn, fp, fn, tp."""
    tn, fp, fn, tp = S[..., 0], S[..., 1], S[..., 2], S[..., 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        f_dam = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f_int = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
    return (f_dam + f_int) / 2


def raster(path):
    with rasterio.open(path) as src:
        return src.read()


# ------------------------------------------------------------------ 1. scores + block bootstrap
def score_table(P):
    """Scores for every method and exam. The 95% interval comes from a BLOCK bootstrap:
    I redraw the list of blocks 2,000 times (with repeats) and recompute the score each time.
    Whole blocks are redrawn, not single pixels, because pixels inside a block are alike;
    redrawing pixels would make the interval look far too narrow."""
    rng = np.random.default_rng(SEED)
    blocks, bidx = np.unique(P["block_id"].to_numpy(), return_inverse=True)
    nb = len(blocks)
    draws = rng.integers(0, nb, (N_BOOTSTRAP, nb))      # the same redraws for every method (a paired comparison)
    y_all = P["label"].to_numpy().astype(np.int64)
    rows, boot = [], {}
    for exam in EXAMS:
        sel, prefix = exam_rows(P, exam)
        y, b = y_all[sel], bidx[sel]
        for m in ORDER:
            pred = P[prefix + m].to_numpy()[sel].astype(np.int64)
            assert set(np.unique(pred)) <= {0, 1}, f"{exam} {m}: missing predictions"
            C = np.bincount(b * 4 + y * 2 + pred, minlength=nb * 4).reshape(nb, 4).astype(float)   # per block: tn fp fn tp
            s = scores_from_cm(*C.sum(0))
            assert abs(s["macro_f1"] - scores(y, pred)["macro_f1"]) < 1e-9       # my formula == scikit-learn's
            f = macro_f1_from_counts(C[draws].sum(1))
            boot[(exam, m)] = f
            (tn, fp), (fn, tp) = s.pop("cm")
            row = {"exam": exam, "method": m, **s, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                   "macro_f1_ci_low": float(np.percentile(f, 2.5)), "macro_f1_ci_high": float(np.percentile(f, 97.5))}
            if "clean" in P:        # second number: only pixels that are fully cropland and fully one UN class
                c = P["clean"].to_numpy()[sel]
                row["macro_f1_clean_pixels"] = scores(y[c], pred[c])["macro_f1"]
                row["clean_pixel_share"] = float(c.mean())
            rows.append(row)
    table = pd.DataFrame(rows)
    gaps = {}
    for model, rival in GAP_PAIRS:
        for exam in EXAMS:
            d = boot[(exam, model)] - boot[(exam, rival)]
            point = (table.query("exam == @exam and method == @model")["macro_f1"].iloc[0]
                     - table.query("exam == @exam and method == @rival")["macro_f1"].iloc[0])
            entry = {"model": model, "baseline": rival, "macro_f1_gap": round(float(point), 4),
                     "ci_low": round(float(np.percentile(d, 2.5)), 4), "ci_high": round(float(np.percentile(d, 97.5)), 4),
                     "share_of_resamples_model_better": round(float((d > 0).mean()), 4), "n_blocks": int(nb)}
            if (model, rival) == GAP_PAIRS[0]:
                gaps[exam] = entry                                   # same layout as the v1 report
            else:
                gaps[f"{exam} | {model} vs {rival}"] = entry
    return table, gaps


# ------------------------------------------------------------------ 2. plot level (the planner's unit)
def plot_status(share):
    """0 intact (< 25% damaged), 1 partly damaged (25-75%), 2 mostly damaged (>= 75%)."""
    return np.digitize(share, PLOT_STATUS_CUTS)


def plot_level(P):
    """The same comparison at 1 ha plots. "Truth" = the share of a plot's cropland pixels the UN marked damaged.
    Three ways to estimate that share:
      baseline_share    share of pixels the NDVI-drop rule calls damaged
      model_share       share of pixels the main model calls damaged
      model_calibrated  average calibrated damage probability of the main model"""
    out = {}
    for d in LABELLED:
        s = P[P["date"] == d]
        g = s.groupby("plot_id")
        T = pd.DataFrame({"n": g.size(), "un": g["label"].mean(),
                          "baseline_share": g["pred__" + MAIN_BASELINE].mean(),
                          "model_share": g["pred__" + MAIN_MODEL].mean(),
                          "model_calibrated": g["cal"].mean()})
        T = T[T["n"] >= PLOT_MIN_CROPLAND_PX]
        true = plot_status(T["un"])
        entry = {"n_plots": int(len(T)), "min_cropland_px": PLOT_MIN_CROPLAND_PX,
                 "un_status_counts": dict(zip(PLOT_STATUS_NAMES, np.bincount(true, minlength=3).tolist())),
                 "un_mean_damaged_share": round(float(T["un"].mean()), 4), "estimators": {}}
        if "cal_v1" in s:
            T["v1_calibrated"] = g["cal_v1"].mean().reindex(T.index)
        for est in [e for e in ("baseline_share", "model_share", "model_calibrated", "v1_calibrated") if e in T]:
            pred = plot_status(T[est])
            f1 = [2 * ((pred == c) & (true == c)).sum() / max((pred == c).sum() + (true == c).sum(), 1) for c in range(3)]
            k = max(len(T) // 10, 1)
            least = T.nsmallest(k, est)                     # the 10% of plots this estimator calls least damaged
            entry["estimators"][est] = {
                "status_accuracy": round(float((pred == true).mean()), 4),
                "status_macro_f1": round(float(np.mean(f1)), 4),
                "share_mae": round(float((T[est] - T["un"]).abs().mean()), 4),       # average size of the error
                "share_bias": round(float((T[est] - T["un"]).mean()), 4),            # + = says more damage than the UN
                "pearson_r": round(float(T[est].corr(T["un"])), 4),
                "spearman_r": round(float(T[est].corr(T["un"], method="spearman")), 4),   # does the model rank plots in the right order?
                "status_off_by_two": round(float((np.abs(pred - true) == 2).mean()), 4),  # intact <-> mostly damaged mix-ups
                "least_damaged_10pct_un_share": round(float(least["un"].mean()), 4),
                "status_confusion_rows_un_cols_pred": pd.crosstab(pd.Categorical(true, [0, 1, 2]),
                                                                  pd.Categorical(pred, [0, 1, 2]), dropna=False).to_numpy().tolist(),
            }
        out[d] = entry
    return out


# ------------------------------------------------------------------ 3. hectares and calibration
def area_totals(P):
    out = {}
    for d in LABELLED:
        s = P[P["date"] == d]
        out[d] = {"cropland_ha": round(len(s) * HA_PER_PX, 1),
                  "damaged_ha_un": round(float(s["label"].sum()) * HA_PER_PX, 1),
                  "damaged_ha_model_calibrated": round(float(s["cal"].sum()) * HA_PER_PX, 1),
                  "damaged_ha_model_pixel_count": round(float(s["pred__" + MAIN_MODEL].sum()) * HA_PER_PX, 1),
                  "damaged_ha_baseline_pixel_count": round(float(s["pred__" + MAIN_BASELINE].sum()) * HA_PER_PX, 1)}
        if "cal_v1" in s:       # v2 report: the v1 model's calibrated estimate on the same pixels
            out[d]["damaged_ha_v1_calibrated"] = round(float(s["cal_v1"].sum()) * HA_PER_PX, 1)
    return out


def calibration_table(P):
    """Group pixels by score (0-0.1, 0.1-0.2, ...) and compare the average score with the share the UN marked damaged."""
    rows = []
    for d in LABELLED:
        s = P[P["date"] == d]
        for col in ("score", "cal"):
            bins = pd.cut(s[col], np.linspace(0, 1, 11), include_lowest=True)
            g = s.groupby(bins, observed=True)
            for b, n, mean_p, obs in zip(g.size().index, g.size(), g[col].mean(), g["label"].mean()):
                rows.append({"date": d, "kind": "raw score" if col == "score" else "calibrated", "bin": str(b),
                             "n_px": int(n), "mean_predicted": round(float(mean_p), 4), "un_damaged_share": round(float(obs), 4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 4. subgroups: where does it work, where not?
def add_context(P):
    """Attach descriptions used only to EXPLAIN errors (never as model inputs)."""
    P = P.copy()
    r, c = P["row"].to_numpy(), P["col"].to_numpy()
    wc = raster(PROCESSED / "composites" / "worldcover_2021.tif")[0]
    P["landcover"] = pd.Series(wc[r, c]).map(WC_NAMES).fillna("other (shrub, grass, bare)").to_numpy()
    cuts_file = ROOT / "results" / "productivity_cuts.json"      # cut points from Khan Younis TRAIN land, saved once
    if cuts_file.exists():
        cuts = np.array(json.loads(cuts_file.read_text())["cuts"])
    else:
        cuts = np.quantile(load_table(LABELLED[0], "train")["prewar_peak_mean"], [1 / 3, 2 / 3])
        cuts_file.write_text(json.dumps({"cuts": [float(c) for c in cuts], "from": "pre-war peak NDVI terciles of Khan Younis training land"}))
    P["productivity"] = np.array(["low", "medium", "high"])[np.digitize(P["prewar_peak_mean"], cuts)]
    P["greenness_now"] = np.array(["NDVI < 0.10 (bare)", "NDVI 0.10-0.20", "NDVI >= 0.20 (green)"])[
        np.digitize(P["post_ndvi"], [0.10, 0.20])]
    first = raster(PROCESSED / "labels" / "first_damaged.tif")[0]
    hist = json.loads((RESULTS / "label_history.json").read_text())["snapshots"]
    snap = [h["snapshot"] for h in hist]                              # YYYYMMDD of each snapshot index
    P["edge"], P["timing"] = "", ""
    for d in LABELLED:
        m = (P["date"] == d).to_numpy()
        lab = read_labels(d)
        k = np.ones((3, 3), bool)
        edge = ((lab == 0) & ndimage.binary_dilation(lab == 1, structure=k)) | \
               ((lab == 1) & ndimage.binary_dilation(lab == 0, structure=k))
        P.loc[m, "edge"] = np.where(edge[r[m], c[m]], "edge (next to the other class)", "interior")
        f = first[r[m], c[m]]
        ym = np.array([int(snap[i][:6]) if i < len(snap) else 999999 for i in f])     # YYYYMM first marked damaged
        now = int(d.replace("-", ""))
        months = (now // 100 - ym // 100) * 12 + (now % 100 - ym % 100)               # months since first marked damaged
        timing = np.select([ym > now, months <= 3, months <= 9], ["not damaged (UN)", "damaged 0-3 months ago",
                                                                  "damaged 4-9 months ago"], "damaged 10+ months ago")
        P.loc[m, "timing"] = timing
    return P


def subgroup_table(P):
    rows = []
    for d in LABELLED:
        s = P[P["date"] == d]
        for dim in ("landcover", "productivity", "edge", "greenness_now", "timing"):
            for name, g in s.groupby(dim):
                y, pm, pb = g["label"].to_numpy(), g["pred__" + MAIN_MODEL].to_numpy(), g["pred__" + MAIN_BASELINE].to_numpy()
                dmg, ok = y == 1, y == 0
                rows.append({"date": d, "dimension": dim, "group": name, "n_px": int(len(g)),
                             "share_of_pixels": round(len(g) / len(s), 4), "un_damaged_share": round(float(y.mean()), 4),
                             "model_error_rate": round(float((pm != y).mean()), 4),
                             "baseline_error_rate": round(float((pb != y).mean()), 4),
                             "model_recall_damaged": round(float(pm[dmg].mean()), 4) if dmg.any() else None,
                             "model_recall_intact": round(float(1 - pm[ok].mean()), 4) if ok.any() else None,
                             "baseline_recall_damaged": round(float(pb[dmg].mean()), 4) if dmg.any() else None,
                             "baseline_recall_intact": round(float(1 - pb[ok].mean()), 4) if ok.any() else None})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ 5. failure cases, picked by a fixed rule
def failure_windows(P, shape):
    """For each date and each kind of mistake, find the 400 m window with the MOST mistakes of that kind.
    Kinds: "missed damage" (UN damaged, model says intact) and "false alarm" (UN intact, model says damaged).
    The rule is fixed in advance, so I cannot hand-pick flattering or dramatic examples."""
    out = []
    used = np.zeros(shape, bool)
    for d in LABELLED:
        s = P[P["date"] == d]
        r, c = s["row"].to_numpy(), s["col"].to_numpy()
        y, p = s["label"].to_numpy(), s["pred__" + MAIN_MODEL].to_numpy()
        grids = {"n": np.zeros(shape), "missed damage": np.zeros(shape), "false alarm": np.zeros(shape)}
        grids["n"][r, c] = 1
        grids["missed damage"][r, c] = (y == 1) & (p == 0)
        grids["false alarm"][r, c] = (y == 0) & (p == 1)
        # count inside every window with a summed-area table (fast)
        def window_sums(a):
            ii = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            return ii[WINDOW_PX:, WINDOW_PX:] - ii[:-WINDOW_PX, WINDOW_PX:] - ii[WINDOW_PX:, :-WINDOW_PX] + ii[:-WINDOW_PX, :-WINDOW_PX]
        n = window_sums(grids["n"])[::STRIDE_PX, ::STRIDE_PX]
        for kind in ("missed damage", "false alarm"):
            e = window_sums(grids[kind])[::STRIDE_PX, ::STRIDE_PX].copy()
            e[n < WINDOW_MIN_PX] = -1
            for flat in np.argsort(e, axis=None)[::-1]:
                i, j = np.unravel_index(flat, e.shape)
                r0, c0 = i * STRIDE_PX, j * STRIDE_PX
                if not used[r0:r0 + WINDOW_PX, c0:c0 + WINDOW_PX].any():     # do not show the same place twice
                    break
            used[r0:r0 + WINDOW_PX, c0:c0 + WINDOW_PX] = True
            w = s[(r >= r0) & (r < r0 + WINDOW_PX) & (c >= c0) & (c < c0 + WINDOW_PX)]
            wy, wp = w["label"].to_numpy(), w["pred__" + MAIN_MODEL].to_numpy()
            bad = (wy == 1) & (wp == 0) if kind == "missed damage" else (wy == 0) & (wp == 1)
            stats = lambda g: {  # noqa: E731
                "n_px": int(len(g)), "ndvi_2022": round(float(g["pre_ndvi"].mean()), 3),
                "ndvi_now": round(float(g["post_ndvi"].mean()), 3), "ndvi_change": round(float(g["d_ndvi"].mean()), 3),
                "bare_soil_change": round(float(g["d_bsi"].mean()), 3),
                "prewar_peak_ndvi": round(float(g["prewar_peak_mean"].mean()), 3),
                "landcover_shares": g["landcover"].value_counts(normalize=True).round(2).to_dict(),
                "timing_shares": g["timing"].value_counts(normalize=True).round(2).to_dict(),
                "edge_share": round(float((g["edge"] != "interior").mean()), 2)}
            out.append({"date": d, "kind": kind, "row0": int(r0), "col0": int(c0), "window_px": WINDOW_PX,
                        "block_ids": sorted(int(b) for b in w["block_id"].unique()),
                        "n_px": int(len(w)), "n_mistakes_of_this_kind": int(bad.sum()),
                        "un_damaged_share": round(float(wy.mean()), 3), "model_damaged_share": round(float(wp.mean()), 3),
                        "baseline_damaged_share": round(float(w["pred__" + MAIN_BASELINE].mean()), 3),
                        "mistaken_pixels": stats(w[bad]), "all_pixels_in_window": stats(w)})
    return out


# ------------------------------------------------------------------ figures
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_INTACT, C_DAMAGED = "#228833", "#aa3377"         # same colours as the label maps


def fig_scores(table, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.8))
    h = 0.8 / len(EXAMS)
    for i, (ex, col) in enumerate(zip(EXAMS, (C_BLUE, C_ORANGE, C_AQUA))):
        a = table[table["exam"] == ex].set_index("method").loc[ORDER]
        y = np.arange(len(ORDER)) + i * h
        err = [a["macro_f1"] - a["macro_f1_ci_low"], a["macro_f1_ci_high"] - a["macro_f1"]]
        ax.barh(y, a["macro_f1"], height=h * 0.9, xerr=err, color=col, label=ex,
                error_kw={"elinewidth": 1, "ecolor": MUTED, "capsize": 0})
        for yy, v, hi in zip(y, a["macro_f1"], a["macro_f1_ci_high"]):
            ax.text(hi + 0.012, yy, f"{v:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(np.arange(len(ORDER)) + h * (len(EXAMS) - 1) / 2)
    ax.set_yticklabels(ORDER)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro F1 (line = 95% interval from redrawing whole blocks)")
    ax.set_title(title, loc="left", fontsize=11)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=len(EXAMS), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_confusion(table, title, path):
    """Confusion matrices for the main baseline and the main model: of the pixels the UN gave each label, where did they go?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(EXAMS), figsize=(3.5 * len(EXAMS), 6.6))
    for i, m in enumerate((MAIN_BASELINE, MAIN_MODEL)):
        for j, ex in enumerate(EXAMS):
            ax = axes[i, j]
            a = table[(table["exam"] == ex) & (table["method"] == m)].iloc[0]
            cm = np.array([[a["tn"], a["fp"]], [a["fn"], a["tp"]]], float)
            share = cm / cm.sum(1, keepdims=True)
            ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
            for (u, v), val in np.ndenumerate(share):
                ax.text(v, u, f"{val:.0%}\n{cm[u, v] * HA_PER_PX:,.0f} ha", ha="center", va="center", fontsize=9,
                        color="white" if val > 0.55 else INK)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels([f"says {c}" for c in CLASSES], fontsize=8)
            ax.set_yticklabels([f"UN: {c}" for c in CLASSES], fontsize=8, rotation=90, va="center")
            ax.set_title(f"{m}\n{ex}  (macro F1 {a['macro_f1']:.2f})", fontsize=9)
            ax.tick_params(length=0)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_failures(P, fails, shape, title, path):
    """One row per failure window: 2022 image, recent image, UN map, model map, mistakes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, Rectangle
    pad = 15
    fig, axes = plt.subplots(len(fails), 5, figsize=(13, 2.62 * len(fails) + 0.9), squeeze=False)
    cache = {}
    for i, f in enumerate(fails):
        d = f["date"]
        if d not in cache:
            cache[d] = (raster(PROCESSED / "composites" / f"pre_{d}.tif"), raster(PROCESSED / "composites" / f"post_{d}.tif"))
        pre, post = cache[d]
        r0, r1 = max(f["row0"] - pad, 0), min(f["row0"] + WINDOW_PX + pad, shape[0])
        c0, c1 = max(f["col0"] - pad, 0), min(f["col0"] + WINDOW_PX + pad, shape[1])
        rgb = lambda img: np.clip(np.nan_to_num(np.moveaxis(img[[2, 1, 0], r0:r1, c0:c1], 0, -1)) / 0.30, 0, 1) ** 0.8  # noqa: E731
        s = P[(P["date"] == d) & P["row"].between(r0, r1 - 1) & P["col"].between(c0, c1 - 1)]
        lab = np.zeros((r1 - r0, c1 - c0)); mod = lab.copy(); err = lab.copy()
        rr, cc = s["row"].to_numpy() - r0, s["col"].to_numpy() - c0
        y, p = s["label"].to_numpy(), s["pred__" + MAIN_MODEL].to_numpy()
        lab[rr, cc] = y + 1
        mod[rr, cc] = p + 1
        err[rr, cc] = np.select([(y == 1) & (p == 0), (y == 0) & (p == 1)], [2, 3], 1)
        two = ListedColormap(["#ffffff", C_INTACT, C_DAMAGED])
        panels = [(rgb(pre), None, "same month in 2022"), (rgb(post), None, "recent image (date at left)"),
                  (lab, two, "UN map"), (mod, two, "model"),
                  (err, ListedColormap(["#ffffff", "#d9d8d2", C_ORANGE, C_BLUE]), "mistakes")]
        for ax, (img, cmap, name) in zip(axes[i], panels):
            ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=(cmap.N - 1) if cmap else None, interpolation="nearest")
            ax.add_patch(Rectangle((f["col0"] - c0 - 0.5, f["row0"] - r0 - 0.5), WINDOW_PX, WINDOW_PX, fill=False, ec=INK, lw=0.8))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=10)
        axes[i, 0].set_ylabel(f"{chr(65 + i)}: {f['kind']}\n{d}", fontsize=9)
    fig.legend(handles=[Patch(color=C_INTACT, label="non-affected / intact"), Patch(color=C_DAMAGED, label="damaged"),
                        Patch(color="#d9d8d2", label="model agrees with UN"),
                        Patch(color=C_ORANGE, label="missed damage (UN damaged, model intact)"),
                        Patch(color=C_BLUE, label="false alarm (UN intact, model damaged)")],
               loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.975))
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ everything together
def full_report(P, out_dir: Path, what: str, shape, order=None, main=None, baseline=None, gap_pairs=None):
    """Run every analysis on the predictions table P and save it in out_dir. `what` names the land in the titles.
    order / main / baseline / gap_pairs replace the v1 method names for the v2 report."""
    global ORDER, MAIN_MODEL, MAIN_BASELINE, GAP_PAIRS
    ORDER = order or ORDER
    MAIN_MODEL, MAIN_BASELINE = main or MAIN_MODEL, baseline or MAIN_BASELINE
    GAP_PAIRS = gap_pairs or [(MAIN_MODEL, MAIN_BASELINE)]
    out_dir.mkdir(parents=True, exist_ok=True)
    table, gaps = score_table(P)
    table.round(4).to_csv(out_dir / "scores.csv", index=False)
    (out_dir / "gap.json").write_text(json.dumps(gaps, indent=2))
    (out_dir / "plot_level.json").write_text(json.dumps(plot_level(P), indent=1))
    (out_dir / "area.json").write_text(json.dumps(area_totals(P), indent=2))
    calibration_table(P).to_csv(out_dir / "calibration.csv", index=False)
    PC = add_context(P)
    subgroup_table(PC).to_csv(out_dir / "subgroups.csv", index=False)
    fails = failure_windows(PC, shape)
    (out_dir / "failures.json").write_text(json.dumps(fails, indent=1))
    fig_scores(table, f"{what}: how well each method matches the UN damage map", out_dir / "scores.png")
    fig_confusion(table, f"{what}: where each UN class ends up (rows add to 100%)", out_dir / "confusion.png")
    fig_failures(PC, fails, shape, f"{what}: the places with the most mistakes, chosen by a fixed rule",
                 out_dir / "failures.png")
    return table, gaps
