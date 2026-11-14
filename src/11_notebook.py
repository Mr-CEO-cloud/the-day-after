"""
11_notebook.py - build and run notebooks/report.ipynb, the notebook with every result visible.

Plain words: the submission asks for a notebook with outputs visible. This script writes that
notebook and runs it, so each table and figure in it is produced by code at that moment. The
notebook only needs the small files that are in the repository (results/ and the saved test
answers), so anyone who clones the repository can run it again in a minute, without downloading
satellite images. Its key cells RECOMPUTE the test scores from the saved per-pixel answers.

Run:  .venv/Scripts/python.exe -I src/11_notebook.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, DISCLAIMER  # noqa: E402

import nbformat  # noqa: E402
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

CELLS = [
    ("md", f"""# The Day After: results notebook

**Which farmland in Khan Younis (Gaza) could feed people first after a ceasefire?** Satellite AI, checked against UN expert damage maps.

*{DISCLAIMER}*

This notebook shows every result of the project with its output visible. It reads only small files stored in the repository
(`results/`). The scores are **recomputed here** from the per-pixel answers that were saved when each test was run, so you can check them
without trusting our tables. The full pipeline (satellite download, features, training) is in `src/`, and the plain-words explanation of
every step is in `docs/explainers.md`."""),
    ("code", """import json, gzip, io
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
ROOT = Path.cwd() if (Path.cwd() / "results").exists() else Path.cwd().parent
R = ROOT / "results"
pd.set_option("display.max_colwidth", 140); pd.set_option("display.width", 200)
load = lambda p: json.loads((R / p).read_text())

def counts(y, p):
    \"\"\"tn, fp, fn, tp: y is the UN label (1 = damaged), p the method's answer.\"\"\"
    y, p = np.asarray(y), np.asarray(p)
    return int(((y == 0) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum()), int(((y == 1) & (p == 0)).sum()), int(((y == 1) & (p == 1)).sum())

def macro_f1(tn, fp, fn, tp):
    \"\"\"F1 for 'damaged' and for 'intact', averaged. 1 = perfect; always answering 'damaged' gives under 0.5.\"\"\"
    f = lambda a, b, c: 2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0
    return (f(tp, fp, fn) + f(tn, fn, fp)) / 2

def rescore(file, methods):
    \"\"\"Recompute every score from a file of saved per-pixel answers.\"\"\"
    A = pd.read_csv(io.BytesIO(gzip.decompress((R / file).read_bytes())))
    rows = []
    for exam, prefix, sel in (("Dec 2024", "pred__", A.date == "2024-12"), ("Jul 2025", "pred__", A.date == "2025-07"), ("Forward in time", "time__", A.date == "2025-07")):
        for m in methods:
            tn, fp, fn, tp = counts(A.label[sel], A[prefix + m][sel])
            rows.append({"exam": exam, "method": m, "pixels": int(sel.sum()), "UN intact, says intact": tn, "UN intact, says damaged": fp,
                         "UN damaged, says intact": fn, "UN damaged, says damaged": tp, "accuracy": round((tn + tp) / sel.sum(), 3), "macro F1": round(macro_f1(tn, fp, fn, tp), 3)})
    return pd.DataFrame(rows)
print("results folder:", R.resolve().name, "| files:", len(list(R.rglob("*.*"))))"""),

    ("md", """## 1. The answer key: UN damage maps
UNOSAT and FAO mapped Gaza's cropland as *damaged* or *non-affected* 18 times (Oct 2023 to Jul 2025). We put each map on our 10 m grid.
The table shows the damaged area in Khan Younis governorate for each map. The last cell compares our hectares with the figure FAO published."""),
    ("code", """h = load("label_history.json")
t = pd.DataFrame(h["snapshots"])[["snapshot", "cropland_ha", "damaged_ha", "damaged_pct"]]
display(t)
x = h["fao_cross_check_march_2025"]
print("March 2025, Khan Younis damaged share. FAO published:", x["fao_khan_younis"]["damaged_pct"], "%   ours:", x["ours_khan_younis"]["damaged_pct"], "%")"""),

    ("md", """## 2. The split: land the model never sees
Neighbouring pixels look alike, so a random split would let a model copy its neighbours. We cut the governorate into 1 km blocks and locked
whole blocks away for testing. Training pixels within 100 m of a test block were dropped."""),
    ("code", """s = load("split_summary.json")
display(pd.DataFrame({k: {"blocks": s["n_blocks"].get(k, ""), "pixels": v["n_px"], "hectares": v["hectares"], "share of cropland": v["share_of_cropland"],
                          "damaged share Dec 2024": v["2024-12"]["damaged_share"], "damaged share Jul 2025": v["2025-07"]["damaged_share"]} for k, v in s["pixels"].items()}).T)
print("closest training pixel to test land:", s["min_train_to_test_distance_m"], "m")
display(Image(str(R / "figures" / "split_map.png"), width=950))"""),

    ("md", """## 3. First model (v1): one month of images
v1 compared one month with the same month of 2022. Before the test land was opened, we wrote the rules in `docs/test_plan.md`.
The next cell **recomputes** the v1 test scores from the saved answers (`results/test/predictions_main.csv.gz`)."""),
    ("code", """u = load("test/unlock.json"); print("test land opened once at", u["opened_at"], "| code version", u["code_version_git_commit"][:7], "|", u["n_rows"], "pixel-dates in", u["n_blocks"], "blocks")
v1 = rescore("test/predictions_main.csv.gz", ["ndvi_drop", "rf_all"])
display(v1)
saved = pd.read_csv(R / "test" / "scores.csv").set_index(["exam", "method"])
print("largest difference from the scores we reported:", max(abs(r["macro F1"] - saved.loc[({"Dec 2024": "same-date 2024-12", "Jul 2025": "same-date 2025-07", "Forward in time": "time 2024-12->2025-07"}[r["exam"]], r["method"]), "macro_f1"]) for _, r in v1.iterrows()).round(4))"""),

    ("md", """## 4. Improved model (v2): each field against its own normal, month after month
v1 failed on fields that are bare every December or July. v2 gives the model each pixel's monthly history against its own 2017-2022 normal,
and it learns from all 18 UN maps. It was compared with other candidates **on training land only** (cross-validation):"""),
    ("code", """c = pd.read_csv(R / "v2" / "cv_candidates.csv").rename(columns={"f1_2024-12": "Dec 2024", "f1_2025-07": "Jul 2025", "headline_mean": "mean of the two",
        "clean_pixels_mean": "on clean pixels", "all_18_months_mean": "mean of all 18 months"})
display(c)
ch = load("v2/chosen.json")
print("our pre-set rule picked:", ch["rule_pick"], "| we chose:", ch["chosen"], "| departed from the rule:", ch["departed_from_rule"])
print("why: the rule's pick learned from two months only. Mean score over all 18 months:", {k: round(v, 3) for k, v in ch["all_18_months_mean"].items() if k in ("hist_2m", "hist_18m")})"""),
    ("md", """The rules for the second look at the test land were written in `docs/test_plan_v2.md` and pushed first. The next cell **recomputes**
the scores from the saved answers. `rule_war` is the simple rival: "damaged if the worst monthly greenness drop since Oct 2023 is below a threshold"."""),
    ("code", """u = load("test_v2/unlock.json"); print("second and last look at", u["opened_at"], "| code version", u["code_version_git_commit"][:7])
v2 = rescore("test_v2/predictions_main.csv.gz", ["rule_war", "rf_all", "v2_2m", "v2"])
display(v2)
g = load("test_v2/gap.json")
display(pd.DataFrame([{"comparison": v["model"] + " minus " + v["baseline"], "exam": k.split(" | ")[0], "gap in macro F1": v["macro_f1_gap"], "95% range, low": v["ci_low"], "95% range, high": v["ci_high"],
                       "whole range above zero": v["ci_low"] > 0} for k, v in g.items()]))"""),
    ("code", """display(Image(str(R / "test_v2" / "scores.png"), width=800))
display(Image(str(R / "test_v2" / "confusion.png"), width=900))"""),

    ("md", """## 5. A new place: Deir al-Balah
The models learned only from Khan Younis. Without any change they were scored once on **all** the cropland of the governorate to the north
(rules in `docs/test_plan_region.md`, pushed first). Recomputed here from the saved answers:"""),
    ("code", """u = load("area_deir_al_balah/unlock.json"); print("scored once at", u["scored_at"], "| code version", u["code_version_git_commit"][:7], "|", u["n_rows"], "pixel-dates in", u["n_blocks"], "blocks")
reg = rescore("area_deir_al_balah/predictions_main.csv.gz", ["rule_war", "rf_all", "v2"])
display(reg)
both = v2[v2.method == "v2"][["exam", "macro F1"]].rename(columns={"macro F1": "v2 in Khan Younis (test land)"}).merge(
    reg[reg.method == "v2"][["exam", "macro F1"]].rename(columns={"macro F1": "v2 in Deir al-Balah (never seen)"}), on="exam")
display(both)
display(pd.read_csv(R / "area_deir_al_balah" / "by_month.csv").rename(columns={"v2_macro_f1": "v2", "rule_war_macro_f1": "simple rule"}))"""),

    ("md", """## 6. Where it fails
Error rates of v2 on Khan Younis test land, by kind of land. Then the four places with the most mistakes, picked by a fixed rule (not by hand)."""),
    ("code", """sg = pd.read_csv(R / "test_v2" / "subgroups.csv")
display(sg[["date", "dimension", "group", "share_of_pixels", "un_damaged_share", "model_error_rate", "baseline_error_rate", "model_recall_damaged", "model_recall_intact"]])
display(Image(str(R / "test_v2" / "failures.png"), width=950))
print("the four worst places of the FIRST model, looked at again (mistakes v1 -> v2):", [(a["mistakes_v1"], a["mistakes_v2"]) for a in load("test_v2/v1_failures_revisited.json")])"""),

    ("md", """## 7. From damage to a shortlist (the planner)
The AI estimates how much of each 1-hectare plot was damaged. A plain rule ranks the plots:
**priority = pre-war productivity x (1 - 0.6 x damaged share)**. First: does the plot-level estimate agree with the UN map on test land?
Then: of the top 10% of plots under different rankings, how much does the UN map call intact?"""),
    ("code", """pl = load("test_v2/plot_level.json")
display(pd.DataFrame([{"month": d, "estimate": k, "plots": e["n_plots"], "status agrees with UN": v["status_accuracy"], "average error of damaged share": v["share_mae"],
                       "bias": v["share_bias"], "rank correlation": v["spearman_r"]} for d, e in pl.items() for k, v in e["estimators"].items()]))
val = load("planner_v2/validation.json")["test"]
display(pd.DataFrame({d: {k: v[k]["un_intact_share_of_top"] for k in ("productivity only", "rule (worst month since Oct 2023)", "v1 planner", "v2 planner", "UN map (oracle)")} | {"all plots": v["all_plots_un_intact_share"]}
                      for d, v in val.items()}).rename_axis("share of the top 10% of plots that the UN map calls intact"))"""),
    ("code", """for month in ("2025-07", "2026-04"):
    t = pd.read_csv(R / "planner_v2" / f"plots_{month}.csv")
    print(month, "| plots by status:", t.status.value_counts().to_dict(), "| estimated damaged ha:", round(t.damaged_ha_est.sum()))
    display(t.head(5)[["rank", "lat", "lon", "cropland_ha", "status", "damaged_share", "prewar_use", "green_now", "reason", "suggested_use"]])"""),

    ("md", f"""## 8. Sources and tools
- **Data:** Copernicus Sentinel-2 L2A (ESA) through the Microsoft Planetary Computer; UNOSAT-FAO Gaza Strip Cropland Damage Assessment (HDX, CC BY-SA);
  ESA WorldCover 2021; OCHA COD-AB boundaries; FAO (2025), *Land availability for cultivation in the Gaza Strip as of April 2025* (CD5442EN/1/05.25).
- **Software:** Python, NumPy, pandas, GeoPandas, rasterio, odc-stac, scikit-learn, LightGBM, matplotlib.
- **AI tools:** Claude Code (Anthropic: Claude Fable 5.2, Opus 5.5, Sonnet 5.5) wrote most of the code with the author; OpenAI Codex reviewed it and recomputed the test numbers.
  Every step is logged in `docs/ai_log.md`.

*{DISCLAIMER}*"""),
]


def main():
    nb = new_notebook()
    nb.cells = [new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    out = ROOT / "notebooks" / "report.ipynb"
    NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(out.parent)}}).execute()
    nbformat.write(nb, out)
    n_out = sum(len(c.get("outputs", [])) for c in nb.cells if c.cell_type == "code")
    print(f"{out.relative_to(ROOT)} written and run: {len(nb.cells)} cells, {n_out} outputs, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
