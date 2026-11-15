"""
app.py - "The Day After": a Gradio demo that helps decide which cropland in Khan Younis
(Gaza) could be brought back first.

Plain words:
- This app only READS small precomputed files in app/data (made by src/07_precompute.py).
  It never calls a satellite service and never trains a model, so it cannot break
  while someone is using it.
- The user picks a month and how strongly damage should count against a plot. The app
  redraws two maps (status and priority), the hectare table, and the top plots with a
  one-line reason and a suggested first step.
- Every number about how well the model works is read from meta.json, which was copied
  from the results files. Nothing is typed in by hand.

This project is a prototype, not agricultural or humanitarian advice.
"""
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))      # so planner_rules.py is found however the app is started
import planner_rules as R  # noqa: E402
D = dict(np.load(HERE / "data" / "plots.npz"))
META = json.loads((HERE / "data" / "meta.json").read_text())
DISCLAIMER = META["disclaimer"]
MONTHS = META["months"]
LABEL = {m["value"]: m["label"] for m in MONTHS}
INDEX = {m["value"]: i for i, m in enumerate(MONTHS)}
HAS_UN = {m["value"]: m["has_un_map"] for m in MONTHS}
ROWS, COLS = META["grid"]["rows"], META["grid"]["cols"]
PR, PC = D["plot_row"].astype(int), D["plot_col"].astype(int)
CROP_HA = D["cropland_px"].astype(np.float64) * 0.01        # one 10 m pixel = 0.01 ha
USE_OPTIONS = ["seasonal crops", "mixed trees and crops", "orchard"]

# colour-blind-safe colours (Okabe-Ito): intact, partly damaged, mostly damaged
STATUS_COLORS = ["#009E73", "#E69F00", "#CC79A7"]
INK, MUTED = "#222222", "#555555"


# ------------------------------------------------------------------ the computation for one choice of month / penalty / filter
def current(month: str, penalty: float, uses):
    i = INDEX[month]
    share = D["share"][i]
    keep = np.isin(D["use_code"], [USE_OPTIONS.index(u) for u in uses]) if uses else np.zeros(len(share), bool)
    prio = R.priority(D["productivity_pct"], share, penalty)
    shown = np.where(keep, prio, np.nan)
    order = np.argsort(-np.nan_to_num(shown, nan=-1))[: int(keep.sum())]      # best first, filtered plots only
    return i, share, keep, shown, order


def grid_of(values, fill=np.nan):
    g = np.full((ROWS, COLS), fill, np.float32)
    g[PR, PC] = values
    return g


def base_axes(title):
    fig, ax = plt.subplots(figsize=(5.0, 6.0))
    for ring in META["boundary"]:
        ax.plot(ring["x"], ring["y"], color="#888888", lw=0.8)
    ax.set_xlim(0, COLS)
    ax.set_ylim(ROWS, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.plot([4, 24], [ROWS - 3, ROWS - 3], color=INK, lw=2)             # 20 plots x 100 m = 2 km
    ax.text(14, ROWS - 5, "2 km", ha="center", va="bottom", fontsize=8, color=INK)
    ax.annotate("N", xy=(8, 9), xytext=(8, 22), ha="center", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))
    return fig, ax


def status_map(codes, keep, title, legend=True):
    fig, ax = base_axes(title)
    g = np.full((ROWS, COLS), np.nan, np.float32)
    shown = np.where(keep, codes, np.nan)
    g[PR, PC] = shown
    ax.imshow(g, cmap=ListedColormap(STATUS_COLORS), vmin=-0.5, vmax=2.5, extent=(0, COLS, ROWS, 0), interpolation="nearest")
    if legend:
        ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(STATUS_COLORS, R.STATUS_NAMES)],
                  loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
    fig.tight_layout()
    return fig


def priority_map(shown, order, title):
    fig, ax = base_axes(title)
    g = grid_of(shown)
    im = ax.imshow(g, cmap="Blues", vmin=0, vmax=1, extent=(0, COLS, ROWS, 0), interpolation="nearest")
    top = order[:15]
    ax.scatter(D["plot_col"][top] + 0.5, D["plot_row"][top] + 0.5, s=55, facecolors="none", edgecolors="#D55E00", linewidths=1.4,
               label="top 15 plots")
    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.9)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("priority (0 low, 1 high)", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ text pieces
def banner(month: str) -> str:
    if HAS_UN[month]:
        return (f"**{LABEL[month]}: a UN damage map exists for this month.** The model's estimate for every plot comes from a model "
                "that had not seen that land, so you can compare it with the UN map shown on the right.")
    fwd = next(s for s in META["scores"] if s["exam"].startswith("time") and s["method"] == "v2")
    return (f"**{LABEL[month]}: no UN damage map exists for this month, so nothing here can be checked.** "
            "This is the model's reading of the satellite images only. "
            f"In the one test that mimics this situation (learn from maps up to Dec 2024, then judge Jul 2025 on unseen land) "
            f"the model scored {fwd['macro_f1']:.2f} out of 1, and was **not clearly better** than a simple rule. "
            "'Damaged' means damaged at some point since October 2023: it does not go away when land turns green again, "
            "so a falling damage estimate means land *looks greener*, not that it was repaired.")


def hectare_table(share, keep):
    ha = CROP_HA
    codes = R.status_of(share)
    rows = []
    for k, name in enumerate(R.STATUS_NAMES):
        m = keep & (codes == k)
        rows.append([name.capitalize(), int(m.sum()), round(float(ha[m].sum()), 1), 0.0, round(float((share[m] * ha[m]).sum()), 1)])
    total = float(ha[keep].sum())
    for r in rows:
        r[3] = round(100 * r[2] / total, 1) if total else 0.0
    rows.append(["All rated plots", int(keep.sum()), round(total, 1), 100.0 if total else 0.0, round(float((share[keep] * ha[keep]).sum()), 1)])
    return pd.DataFrame(rows, columns=["Status", "Plots (1 ha squares)", "Cropland (ha)", "Share of rated cropland (%)",
                                       "Estimated damaged since Oct 2023 (ha)"])


def top_table(i, share, order, n=15):
    rows = []
    for rank, p in enumerate(order[:n], 1):
        s = float(share[p])
        status = R.STATUS_NAMES[int(R.status_of(s))]
        use = R.USE_NAMES[int(D["use_code"][p])]
        conf = R.CONFIDENCE_NAMES[int(D["conf"][i][p])]
        green = R.GREEN_NAMES[int(D["green"][i][p])]
        rows.append([rank, f"{D['lat'][p]:.4f}, {D['lon'][p]:.4f}", round(float(CROP_HA[p]), 1), status.capitalize(),
                     f"{s:.0%}", use.capitalize(), R.productivity_label(float(D["productivity_pct"][p])).capitalize(), green.capitalize(),
                     R.reason(use, float(D["productivity_pct"][p]), s, float(CROP_HA[p]), status),
                     R.suggestion(status, use, float(D["built_share"][p]), green, conf)])
    return pd.DataFrame(rows, columns=["Rank", "Location (lat, lon)", "Size (ha)", "Status", "Est. damaged", "Pre-war use",
                                       "Pre-war greenness", "Green now vs normal?", "Why it ranks here", "Suggested first step"])


def download_file(month, i, share, shown, order):
    rows = []
    for rank, p in enumerate(order, 1):
        s = float(share[p])
        status = R.STATUS_NAMES[int(R.status_of(s))]
        use = R.USE_NAMES[int(D["use_code"][p])]
        conf = R.CONFIDENCE_NAMES[int(D["conf"][i][p])]
        green = R.GREEN_NAMES[int(D["green"][i][p])]
        rows.append({"rank": rank, "lat": round(float(D["lat"][p]), 5), "lon": round(float(D["lon"][p]), 5),
                     "cropland_ha": round(float(CROP_HA[p]), 2), "status": status, "estimated_damaged_share": round(s, 3),
                     "priority": round(float(shown[p]), 4), "pre_war_use": use,
                     "pre_war_greenness_percentile": round(float(D["productivity_pct"][p]), 3), "green_now_vs_normal": green,
                     "confidence": conf,
                     "reason": R.reason(use, float(D["productivity_pct"][p]), s, float(CROP_HA[p]), status),
                     "suggested_first_step": R.suggestion(status, use, float(D["built_share"][p]), green, conf),
                     "note": DISCLAIMER})
    path = Path(tempfile.mkdtemp()) / f"day_after_{month}_ranked_plots.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


NEED_ONE = " **Tick at least one pre-war use above to see plots.**"


def update(month, penalty, uses):
    if not uses:
        empty = pd.DataFrame()
        return (banner(month) + NEED_ONE, None, None, None, empty, empty, None)
    i, share, keep, shown, order = current(month, penalty, uses)
    codes = R.status_of(share)
    f1 = status_map(codes, keep, f"Estimated status, {LABEL[month]}")
    f2 = priority_map(shown, order, f"Recovery priority, {LABEL[month]} (damage penalty {penalty:.1f})")
    f3 = None
    if HAS_UN[month]:
        un = D["un_share"][i]
        f3 = status_map(np.nan_to_num(R.status_of(np.nan_to_num(un)).astype(float)), keep & np.isfinite(un), f"UN damage map for the same plots, {LABEL[month]}")
    return (banner(month), f1, f2, f3, hectare_table(share, keep), top_table(i, share, order), download_file(month, i, share, shown, order))


# ------------------------------------------------------------------ the "how well does it work" tab
METHOD_NAMES = {
    "majority": "Always say 'damaged'",
    "ndvi_drop": "v1 rule: greenness drop vs same month of 2022",
    "rule_war": "Rule: worst monthly greenness drop since Oct 2023",
    "rf_all": "v1 model (one month of images)",
    "v2_2m": "Side result: v2 features, learned from 2 months only",
    "v2": "v2 model (season history, learned from 18 UN maps)",
}
EXAM_NAMES = {"same-date 2024-12": "Dec 2024", "same-date 2025-07": "Jul 2025", "time 2024-12->2025-07": "Forward in time"}


def score_table():
    rows = []
    for m, name in METHOD_NAMES.items():
        row = {"Method": name}
        for e, en in EXAM_NAMES.items():
            s = next(x for x in META["scores"] if x["method"] == m and x["exam"] == e)
            row[en] = f"{s['macro_f1']:.2f}  ({s['macro_f1_ci_low']:.2f}-{s['macro_f1_ci_high']:.2f})"
        rows.append(row)
    return pd.DataFrame(rows)


def works_text() -> str:
    return """
### The test
The model was scored on **40 blocks of land (1 km by 1 km each) that it never saw while learning**, against the UN damage maps
(UNOSAT-FAO, Dec 2024 and Jul 2025). The score is *macro F1*: 1 is perfect, and always guessing 'damaged' scores about 0.4 to 0.5.
Brackets show a 95% range from re-drawing the test blocks 2,000 times.
"""


def works_text_after() -> str:
    g = META["gaps"]
    d, j, f = g["same-date 2024-12"], g["same-date 2025-07"], g["time 2024-12->2025-07"]
    v1d, v1j, v1f = (g["same-date 2024-12 | v2 vs rf_all"], g["same-date 2025-07 | v2 vs rf_all"], g["time 2024-12->2025-07 | v2 vs rf_all"])
    pl = META["plot_level"]
    r1, r2 = META["ranking"]["2024-12"], META["ranking"]["2025-07"]
    clean = {e: next(x for x in META["scores"] if x["method"] == "v2" and x["exam"] == e)["macro_f1_clean_pixels"] for e in EXAM_NAMES}
    fr = META["failures_revisited"]
    return f"""
### What the numbers say
- **v2 beats the earlier model (v1) on unseen land** by {v1d['macro_f1_gap']:.2f} (Dec 2024), {v1j['macro_f1_gap']:.2f} (Jul 2025) and {v1f['macro_f1_gap']:.2f} (forward in time). In all three, the whole 95% range is above zero.
- **Against the simple season-history rule** the lead is {d['macro_f1_gap']:.2f} (Dec 2024) and {j['macro_f1_gap']:.2f} (Jul 2025), both clearly above zero. **Looking forward in time it is only {f['macro_f1_gap']:.3f}, and the range ({f['ci_low']:+.3f} to {f['ci_high']:+.3f}) includes zero: I do not claim a win there.** That is the situation for the months with no UN map.
- **Most of the gain comes from information, not from a fancier model:** the simple rule with season history already beats the whole v1 model.
- **At the planner's unit (1-hectare plots)** the estimated status matches the UN map for {pl['2024-12']['model_calibrated']['status_accuracy']:.0%} (Dec 2024) and {pl['2025-07']['model_calibrated']['status_accuracy']:.0%} (Jul 2025) of plots, against {pl['2024-12']['v1_calibrated']['status_accuracy']:.0%} and {pl['2025-07']['v1_calibrated']['status_accuracy']:.0%} for v1. The estimated share of a plot that is damaged is off by {pl['2024-12']['model_calibrated']['share_mae']:.2f} on average (Dec) and {pl['2025-07']['model_calibrated']['share_mae']:.2f} (Jul).
- **Does the shortlist point to good land?** Of the 10% of plots ranked highest, the UN map calls this share intact: **{r1['v2 planner']['un_intact_share_of_top']:.0%}** (Dec 2024) and **{r2['v2 planner']['un_intact_share_of_top']:.0%}** (Jul 2025). Ignoring damage would give {r1['productivity only']['un_intact_share_of_top']:.0%} and {r2['productivity only']['un_intact_share_of_top']:.0%}; a ranking built from the UN map itself would give {r1['UN map (oracle)']['un_intact_share_of_top']:.0%} and {r2['UN map (oracle)']['un_intact_share_of_top']:.0%}. So the list is a shortlist for field teams, not an answer.

### Where it still fails
- **Field borders.** About a quarter of pixels sit next to the other UN class, and the error there is several times higher than inside fields. The UN map is drawn on a finer grid (5 m) than the satellite pixels (10 m). On pixels that are fully one UN class, the v2 score is {clean['same-date 2024-12']:.2f} (Dec) and {clean['same-date 2025-07']:.2f} (Jul), against the headline numbers above.
- **Bare land that is intact** is often called damaged, and land that was hardly green before the war gives little signal.
- **Four places where v1 failed worst, looked at again:** v1 made {fr[0]['mistakes_v1']}, {fr[1]['mistakes_v1']}, {fr[2]['mistakes_v1']} and {fr[3]['mistakes_v1']} mistakes there; v2 makes {fr[0]['mistakes_v2']}, {fr[1]['mistakes_v2']}, {fr[2]['mistakes_v2']} and {fr[3]['mistakes_v2']}.
- **The reference is not perfect.** The UN map was made from the same satellite's greenness history, so a high score means 'agrees with the UN analysts'. Its 'damaged' label also stays on land that later turns green.

### How honest is this test?
The test blocks were opened once for the first model (v1) and **once more for v2**. v2 was chosen on training land only, and its rules were written and published before the second look. The idea for v2 came partly from v1's mistakes on those same blocks, so a fully independent check would need fresh land. I also departed from my own pre-set selection rule, for a reason recorded before the test: the rule's pick scored well on its two months but poorly looking forward (side result in the table).
"""


# ------------------------------------------------------------------ about tab
ABOUT = """
### What this is
After a ceasefire, agronomists (FAO, NGOs) have limited seeds, water and staff. **The Day After** estimates which cropland in Khan Younis could feed people fastest, using satellite images checked against UN expert damage maps, and turns that into a ranked shortlist of 1-hectare plots with a reason and a suggested first step for each.

### How the priority is computed
For each 1-hectare plot (a 100 m square with at least 0.2 ha of UN-mapped cropland):
1. **Damaged share**: the average chance that its 10 m pixels were damaged since October 2023 (model v2, calibrated).
2. **Status**: intact (under 25% damaged), partly damaged (25-75%), mostly damaged (75% or more).
3. **Productivity**: how green the plot was at its yearly peak in 2017-2022, as a percentile among all plots.
4. **Priority = productivity x (1 - penalty x damaged share).** The **damage penalty** is a *values choice, not a measured fact*: 0 ignores damage, 1 makes a fully damaged plot worth nothing. The default of 0.6 means an intact plot keeps its full score, a half-damaged plot keeps 70%, and a fully damaged plot keeps 40%. Move the slider to see how the ranking changes.
5. **Pre-war use** comes from ESA WorldCover 2021: orchard (50% or more trees), mixed (20-50%), or seasonal crops.

### What it does not know
- **Access.** FAO reports that most Khan Younis cropland was not accessible in April 2025. The planner does not know which land can be reached, or which land is safe to enter.
- **Soil, water, ownership, unexploded ordnance, or what farmers want to plant.**
- **Whether a field is cultivated now.** 'Damaged' follows the UN's meaning: damaged at some point since October 2023.
- **Months after July 2025** have no UN map to check against.

### Data, tools and AI help (all cited)
- **Sentinel-2 L2A** satellite images (ESA, Copernicus programme), accessed through the Microsoft Planetary Computer.
- **ESA WorldCover 2021** land cover (CC BY 4.0).
- **UNOSAT-FAO cropland damage assessment** for the Gaza Strip, Oct 2023 to Jul 2025, from the Humanitarian Data Exchange (CC BY-SA). Used as the answer key.
- **OCHA COD-AB** administrative boundaries (Khan Younis governorate outline).
- **FAO (2025)**, *Land availability for cultivation in the Gaza Strip as of April 2025* (CD5442EN/1/05.25), used only to cross-check hectare totals.
- **Software:** Python, NumPy, pandas, GeoPandas, rasterio, odc-stac, pystac-client, planetary-computer, scikit-learn, LightGBM, matplotlib, Gradio.
- **AI tools:** built with Claude Code (Claude Fable 5.2, Opus 5.5, Sonnet 5.5 by Anthropic), with independent code review by OpenAI Codex.

### Where the code is
https://github.com/Mr-CEO-cloud/the-day-after (the test plans, results and the log of every decision are in the repository).

*Built by a high-school student for the Veritas AI International High School Hackathon 2026 (AI + Climate track). The framing is humanitarian and neutral: the aim is to support recovery planning for the people who farm this land.*
"""


# ------------------------------------------------------------------ the interface
def build():
    with gr.Blocks(title="The Day After") as demo:
        gr.Markdown("# The Day After\n**Which Gaza cropland could feed people fastest? Satellite AI checked against UN expert damage maps.** "
                    "Study area: Khan Younis governorate.")
        gr.Markdown(f"> **{DISCLAIMER}**")
        with gr.Tabs():
            with gr.Tab("Plan"):
                with gr.Row():
                    month = gr.Dropdown(choices=[(f"{m['label']}" + ("  (UN map available)" if m["has_un_map"] else "  (no UN map: model estimate only)"), m["value"])
                                                 for m in MONTHS], value="2025-07", label="Month", scale=2)
                    penalty = gr.Slider(0.0, 1.0, value=R.DEFAULT_PENALTY, step=0.1, label="Damage penalty (a values choice: how much damage counts against a plot)", scale=3)
                    uses = gr.CheckboxGroup(choices=[u.capitalize() for u in USE_OPTIONS], value=[u.capitalize() for u in USE_OPTIONS],
                                            label="Show these pre-war uses", scale=3)
                note = gr.Markdown()
                with gr.Row():
                    m1, m2 = gr.Plot(label="Status map"), gr.Plot(label="Priority map")
                    m3 = gr.Plot(label="UN map for comparison")
                gr.Markdown("#### Hectares by status")
                hect = gr.Dataframe(interactive=False)
                gr.Markdown("#### Top 15 plots to look at first (a shortlist for field teams, not a decision)")
                top = gr.Dataframe(interactive=False, wrap=True)
                csv = gr.File(label="Download the full ranked list (CSV)", interactive=False)
                gr.Markdown(f"*{DISCLAIMER} Every suggestion says to verify on the ground.*")

                def go(mo, pen, us):
                    return update(mo, pen, [u.lower() for u in us])
                for c in (month, penalty, uses):
                    c.change(go, [month, penalty, uses], [note, m1, m2, m3, hect, top, csv])
                demo.load(go, [month, penalty, uses], [note, m1, m2, m3, hect, top, csv])
            with gr.Tab("How well does it work?"):
                gr.Markdown(works_text())
                gr.Dataframe(score_table(), interactive=False, wrap=True)
                gr.Markdown("*Rows: methods. Columns: the three tests. 'Forward in time' means learning from UN maps up to Dec 2024 and judging Jul 2025 on unseen land.*")
                gr.Markdown(works_text_after())
                gr.Image(str(HERE / "assets" / "test_scores.png"), label="Scores on unseen land", show_label=True)
                gr.Image(str(HERE / "assets" / "test_confusion.png"), label="Where each UN class ends up", show_label=True)
                gr.Image(str(HERE / "assets" / "test_failures.png"), label="The four places with the most mistakes, chosen by a fixed rule", show_label=True)
                gr.Markdown(f"*{DISCLAIMER}*")
            with gr.Tab("About, sources and limits"):
                gr.Markdown(ABOUT)
                gr.Markdown(f"**{DISCLAIMER}**")
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch()
