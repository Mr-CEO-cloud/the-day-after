"""
07b_static_site.py - build the data, the facts and the pictures of the web page (folder web/).

Plain words:
- The page is "static": it runs entirely in the visitor's browser, with no server behind it.
  Hugging Face hosts such pages for free, and there is nothing that can crash.
- This script does not compute anything about the model. It only repackages results that
  already exist:
    web/data/plots.json      per plot: place, size, pre-war facts; per month: damaged share
                             (in 1/10,000), status / confidence / green-now code, UN share
    web/data/content.json    month list, boundary outline, and "facts": every number the page
                             prints, formatted here from the results files. The page's HTML
                             has empty slots (data-fact="...") that get filled from this file,
                             so no result is ever typed by hand.
    web/img/lapse/*.webp     half-size satellite pictures for the time-lapse at the top
    web/img/month/*.webp     full-size satellite picture for each month the page offers
    web/img/before/*.webp    full-size picture of the same calendar month in 2022
    web/assets/*.png         the figures from the test results
- The satellite pictures are the monthly Sentinel-2 composites made by 02b_monthly.py, shown
  in natural colour (red, green, blue bands), with the same brightness stretch everywhere.
  Credit: contains modified Copernicus Sentinel data, 2017-2026.
- The model itself and the sample of test pixels are exported by 07c_export_model.py.

Run:  .venv/Scripts/python.exe -I src/07b_static_site.py            (add --no-images to skip the pictures)
"""
import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, PROCESSED, RESULTS, DISCLAIMER, LABEL_MONTHS, V2_APP_MONTHS  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from PIL import Image  # noqa: E402
import app as gradio_app  # noqa: E402  (its packed data and helpers are reused, so both versions agree)

WEB = ROOT / "web"
REPO = "https://github.com/Mr-CEO-cloud/the-day-after"
MONTHS = LABEL_MONTHS + V2_APP_MONTHS
NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
# Colours measured from the Khan Younis images themselves (medians over the governorate, March 2022):
# bare soil, dark soil, vegetation, dense vegetation. The page's palette is built from these.
SAMPLED_PALETTE = {"sand": "#e9cfb0", "dark_soil": "#b39e89", "vegetation": "#61623e", "dense_vegetation": "#45532f"}


def label(m: str) -> str:
    return f"{NAMES[int(m[5:]) - 1]} {m[:4]}"


def ints(a, scale=1):
    return np.round(np.asarray(a, np.float64) * scale).astype(int).tolist()


# ------------------------------------------------------------------ pictures
def natural_colour(month: str) -> Image.Image:
    """One monthly composite as an ordinary colour picture. Bands 3, 2, 1 of the file = red, green, blue."""
    with rasterio.open(PROCESSED / "monthly" / f"s2_{month}.tif") as src:
        a = src.read([3, 2, 1])
    rgb = np.clip(np.nan_to_num(np.moveaxis(a, 0, -1)) / 0.30, 0, 1) ** 0.8       # same stretch as the failure figures
    return Image.fromarray((rgb * 255).astype(np.uint8))


def lapse_months():
    """Time-lapse frames: one per quarter while things change slowly, every month from mid-2023."""
    out = [f"{y}-{m:02d}" for y in range(2017, 2024) for m in (1, 4, 7, 10) if (y, m) < (2023, 7)]
    out += [f"{y}-{m:02d}" for y in range(2023, 2027) for m in range(1, 13) if (2023, 7) <= (y, m) <= (2026, 10)]
    return out


def build_images():
    for sub in ("lapse", "month", "before"):
        (WEB / "img" / sub).mkdir(parents=True, exist_ok=True)
    total = 0
    for m in lapse_months():
        im = natural_colour(m)
        im.resize((round(im.width * 0.42), round(im.height * 0.42)), Image.LANCZOS).save(WEB / "img" / "lapse" / f"{m}.webp", "WEBP", quality=50)
    for m in MONTHS:
        natural_colour(m).save(WEB / "img" / "month" / f"{m}.webp", "WEBP", quality=68)
    for k in range(1, 13):
        natural_colour(f"2022-{k:02d}").save(WEB / "img" / "before" / f"{k:02d}.webp", "WEBP", quality=68)
    for sub in ("lapse", "month", "before"):
        size = sum(p.stat().st_size for p in (WEB / "img" / sub).iterdir())
        total += size
        print(f"img/{sub}: {len(list((WEB / 'img' / sub).iterdir()))} pictures, {size / 1e6:.1f} MB")
    return total


# ------------------------------------------------------------------ facts: every number the page prints
def facts() -> dict:
    """Formatted strings, each computed from a results file. Keys are used as data-fact="..." in the page."""
    f = {}
    sc = pd.read_csv(RESULTS / "test_v2" / "scores.csv").set_index(["exam", "method"])
    exams = {"dec": "same-date 2024-12", "jul": "same-date 2025-07", "fwd": "time 2024-12->2025-07"}
    for e, exam in exams.items():
        for m in ("majority", "ndvi_drop", "rule_war", "rf_all", "v2_2m", "v2"):
            r = sc.loc[(exam, m)]
            f[f"f1_{m}_{e}"] = f"{r['macro_f1']:.2f}"
            f[f"ci_{m}_{e}"] = f"{r['macro_f1_ci_low']:.2f} to {r['macro_f1_ci_high']:.2f}"
            f[f"acc_{m}_{e}"] = f"{r['accuracy']:.0%}"
        f[f"clean_v2_{e}"] = f"{sc.loc[(exam, 'v2'), 'macro_f1_clean_pixels']:.2f}"
        f[f"recall_intact_v2_{e}"] = f"{sc.loc[(exam, 'v2'), 'recall_intact']:.0%}"
        f[f"precision_intact_v2_{e}"] = f"{sc.loc[(exam, 'v2'), 'precision_intact']:.0%}"
        f[f"precision_damaged_v2_{e}"] = f"{sc.loc[(exam, 'v2'), 'precision_damaged']:.0%}"
    gaps = json.loads((RESULTS / "test_v2" / "gap.json").read_text())
    for e, exam in exams.items():
        for rival, key in (("rule", exam), ("v1", f"{exam} | v2 vs rf_all")):
            g = gaps[key]
            f[f"gap_{rival}_{e}"] = f"{g['macro_f1_gap']:+.2f}"
            f[f"gap_{rival}_{e}_range"] = f"{g['ci_low']:+.3f} to {g['ci_high']:+.3f}"
    pl = json.loads((RESULTS / "test_v2" / "plot_level.json").read_text())
    for e, d in (("dec", "2024-12"), ("jul", "2025-07")):
        for k, name in (("model_calibrated", "v2"), ("v1_calibrated", "v1"), ("baseline_share", "rule")):
            f[f"plot_status_{name}_{e}"] = f"{pl[d]['estimators'][k]['status_accuracy']:.0%}"
            f[f"plot_mae_{name}_{e}"] = f"{pl[d]['estimators'][k]['share_mae']:.2f}"
    rk = json.loads((RESULTS / "planner_v2" / "validation.json").read_text())["test"]
    for e, d in (("dec", "2024-12"), ("jul", "2025-07")):
        for k, name in (("productivity only", "prod"), ("rule (worst month since Oct 2023)", "rule"), ("v1 planner", "v1"),
                        ("v2 planner", "v2"), ("UN map (oracle)", "oracle")):
            f[f"top_intact_{name}_{e}"] = f"{rk[d][k]['un_intact_share_of_top']:.0%}"
        f[f"all_intact_{e}"] = f"{rk[d]['all_plots_un_intact_share']:.0%}"
        # the benefit in hectares: UN-intact cropland inside the top-ranked 10% of test plots
        ha = {name: rk[d][k]["un_intact_ha_in_top"] for k, name in (("productivity only", "prod"), ("v2 planner", "v2"), ("UN map (oracle)", "oracle"))}
        for name, v in ha.items():
            f[f"reach_ha_{name}_{e}"] = f"{v:.0f}"
        f[f"reach_times_{e}"] = f"{ha['v2'] / ha['prod']:.1f}"
        f[f"reach_of_best_{e}"] = f"{ha['v2'] / ha['oracle']:.0%}"
        f[f"reach_plots_{e}"] = f"{rk[d]['top_k']:,}"
        f[f"reach_all_plots_{e}"] = f"{rk[d]['n_plots']:,}"
    again = json.loads((RESULTS / "test_v2" / "v1_failures_revisited.json").read_text())
    f["fail_v1"] = ", ".join(str(a["mistakes_v1"]) for a in again)
    f["fail_v2"] = ", ".join(str(a["mistakes_v2"]) for a in again)
    area = json.loads((RESULTS / "test_v2" / "area.json").read_text())
    for e, d in (("dec", "2024-12"), ("jul", "2025-07")):
        f[f"ha_un_{e}"] = f"{area[d]['damaged_ha_un']:,.0f}"
        f[f"ha_v2_{e}"] = f"{area[d]['damaged_ha_model_calibrated']:,.0f}"
    # the trained model and the tests
    fm = json.loads((RESULTS / "v2" / "final_models.json").read_text())
    u1 = json.loads((RESULTS / "test" / "unlock.json").read_text())
    u2 = json.loads((RESULTS / "test_v2" / "unlock.json").read_text())
    model = json.loads((WEB / "data" / "model.json").read_text()) if (WEB / "data" / "model.json").exists() else None
    sample = json.loads((WEB / "data" / "sample.json").read_text()) if (WEB / "data" / "sample.json").exists() else None
    f["n_test_rows"] = f"{u2['n_rows']:,}"
    f["n_test_blocks"] = str(u2["n_blocks"])
    with np.load(PROCESSED / "cv_v2" / "meta.npz", allow_pickle=True) as meta:      # one row per training pixel per UN map
        f["n_train_rows"] = f"{len(meta['label']):,}"
    f["n_train_months"] = str(len(fm.get("training_months", LABEL_MONTHS)))
    if model:
        f["n_trees"] = str(len(model["trees"]))
        f["n_splits"] = f"{sum(len(t['f']) for t in model['trees']):,}"
        f["n_features"] = str(len(model["features"]))
    if sample:
        f["n_sample"] = f"{sample['n']:,}"
        f["n_sample_month"] = f"{sample['n'] // len(sample['months']):,}"
        f["n_history"] = f"{len(sample['history']['pixels']):,}"
    f["unlock_v1_time"] = u1["opened_at"].replace("T", " ")[:16]
    f["unlock_v2_time"] = u2["opened_at"].replace("T", " ")[:16]
    f["unlock_v1_commit"] = u1["code_version_git_commit"][:7]
    f["unlock_v2_commit"] = u2["code_version_git_commit"][:7]
    # the land
    summ = json.loads((RESULTS / "planner_v2" / "summary.json").read_text())["months"]
    f["n_plots"] = f"{gradio_app.META['n_plots_rated']:,}"
    f["cropland_ha"] = f"{summ['2025-07']['cropland_ha_all_pixels']:,.0f}"
    first, last = summ[LABEL_MONTHS[0]], summ[LABEL_MONTHS[-1]]
    f["un_share_first"] = f"{first['damaged_ha_un_all_pixels'] / first['cropland_ha_all_pixels']:.0%}"
    f["un_share_last"] = f"{last['damaged_ha_un_all_pixels'] / last['cropland_ha_all_pixels']:.0%}"
    f["un_ha_last"] = f"{last['damaged_ha_un_all_pixels']:,.0f}"
    f["n_un_maps"] = str(len(LABEL_MONTHS))
    hist = json.loads((RESULTS / "label_history.json").read_text())["fao_cross_check_march_2025"]
    f["fao_ours"] = f"{hist['ours_khan_younis']['damaged_pct']:.1f}%"
    f["fao_theirs"] = f"{hist['fao_khan_younis']['damaged_pct']:.1f}%"
    return f


def un_series():
    """UN-mapped damaged share of Khan Younis cropland for each of the 18 UN maps (for the step-3 chart)."""
    summ = json.loads((RESULTS / "planner_v2" / "summary.json").read_text())["months"]
    return [{"month": m, "label": label(m), "un_share": round(summ[m]["damaged_ha_un_all_pixels"] / summ[m]["cropland_ha_all_pixels"], 4),
             "model_share": round(summ[m]["damaged_ha_est_all_pixels"] / summ[m]["cropland_ha_all_pixels"], 4)} for m in LABEL_MONTHS]


def region_test():
    """The new-place test (a second governorate the model never saw), if it has been run."""
    p = RESULTS / "area_deir_al_balah" / "region_test.json"
    return json.loads(p.read_text()) if p.exists() else None


def main():
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "assets").mkdir(parents=True, exist_ok=True)
    D, META = gradio_app.D, gradio_app.META
    months = [m["value"] for m in META["months"]]
    code = D["status"].astype(int) + 3 * D["conf"].astype(int) + 6 * D["green"].astype(int)      # 0..11
    un = np.where(np.isfinite(D["un_share"]), np.round(D["un_share"] * 10000), -1).astype(int)
    plots = {
        "n": int(len(D["plot_id"])),
        "plot_id": D["plot_id"].astype(int).tolist(),
        "row": D["plot_row"].astype(int).tolist(), "col": D["plot_col"].astype(int).tolist(),
        "lat": np.round(D["lat"].astype(np.float64), 5).tolist(), "lon": np.round(D["lon"].astype(np.float64), 5).tolist(),
        "px": D["cropland_px"].astype(int).tolist(),                         # hectares = pixels x 0.01
        "prod": np.round(D["productivity_pct"].astype(np.float64), 4).tolist(),
        "built": np.round(D["built_share"].astype(np.float64), 4).tolist(),
        "use": D["use_code"].astype(int).tolist(),
        "land": D["land"].astype(int).tolist(),                              # 0 training land, 1 test land, 2 buffer
        "months": months,
        "share": [ints(s, 10000) for s in D["share"]],                       # damaged share x 10,000
        "code": [c.tolist() for c in code],                                  # status + 3 x lower-confidence + 6 x green-now
        "un": [u.tolist() for u in un],                                      # UN share x 10,000, -1 where there is no UN map
    }
    (WEB / "data" / "plots.json").write_text(json.dumps(plots, separators=(",", ":")))

    with rasterio.open(PROCESSED / "split" / "split.tif") as src:
        image_px = {"width": src.width, "height": src.height}                # the satellite pictures cover this many 10 m pixels
    content = {
        "disclaimer": DISCLAIMER,
        "repo": REPO,
        "months": META["months"],
        "grid": META["grid"],
        "image_px": image_px,
        "boundary": META["boundary"],
        "lapse": [{"month": m, "label": label(m)} for m in lapse_months()],
        "facts": facts(),
        "un_series": un_series(),
        "region_test": region_test(),
        "sampled_palette": SAMPLED_PALETTE,
        "image_credit": "Contains modified Copernicus Sentinel data, 2017-2026 (Sentinel-2 L2A via Microsoft Planetary Computer).",
    }
    (WEB / "data" / "content.json").write_text(json.dumps(content, separators=(",", ":")))
    for name in ("scores", "confusion", "failures"):
        shutil.copy(RESULTS / "test_v2" / f"{name}.png", WEB / "assets" / f"test_{name}.png")
    for f in ("data/plots.json", "data/content.json"):
        print(f"{f}: {(WEB / f).stat().st_size / 1e6:.2f} MB")
    print(f"facts: {len(content['facts'])} formatted numbers from the results files")
    if "--no-images" not in sys.argv:
        build_images()


if __name__ == "__main__":
    main()
