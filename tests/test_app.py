"""
test_app.py - checks that the web app shows exactly what the planner produced.

Plain words: the app rebuilds the one-line reasons, suggested steps and status labels from
small saved codes. This test compares, for every plot in all 21 months, the app's text with
the text in the planner's own tables (data/processed/planner_v2/plots_<month>.csv). It also
checks the ranking at the default damage penalty, that every month runs, and that the
disclaimer is present.

Run:  .venv/Scripts/python.exe -I tests/test_app.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import app  # noqa: E402
import planner_rules as R  # noqa: E402
from config import PROCESSED, DISCLAIMER  # noqa: E402

ALL_USES = ["seasonal crops", "mixed trees and crops", "orchard"]


def main():
    assert app.DISCLAIMER == DISCLAIMER == "This project is a prototype, not agricultural or humanitarian advice."
    source = (ROOT / "app" / "app.py").read_text(encoding="utf-8")
    assert source.count("{DISCLAIMER}") >= 4, "the disclaimer must appear on every tab"
    print("ok 1: the disclaimer text matches the project's wording and is on every tab")

    n_text, n_diff = 0, 0
    for month in [m["value"] for m in app.MONTHS]:
        csv = pd.read_csv(PROCESSED / "planner_v2" / f"plots_{month}.csv").set_index("plot_id").sort_index()
        i = app.INDEX[month]
        assert (csv.index.to_numpy() == app.D["plot_id"]).all()
        share = app.D["share"][i]
        # status, confidence, green flag
        assert (np.array(R.STATUS_NAMES)[app.D["status"][i]] == csv["status"].to_numpy()).all(), month
        assert (np.array(R.CONFIDENCE_NAMES)[app.D["conf"][i]] == csv["confidence"].to_numpy()).all(), month
        assert (np.array(R.GREEN_NAMES)[app.D["green"][i]] == csv["green_now"].to_numpy()).all(), month
        # the status the app would compute from the share agrees too (up to the 4-decimal rounding in the CSV)
        agree = (R.status_of(share) == app.D["status"][i]).mean()
        assert agree > 0.9995, (month, agree)
        # text: the same sentence for every plot
        for k in range(len(csv)):
            r = csv.iloc[k]
            status = R.STATUS_NAMES[app.D["status"][i][k]]
            use = R.USE_NAMES[app.D["use_code"][k]]
            reason = R.reason(use, float(app.D["productivity_pct"][k]), float(share[k]), float(app.CROP_HA[k]), status)
            sug = R.suggestion(status, use, float(app.D["built_share"][k]), R.GREEN_NAMES[app.D["green"][i][k]],
                               R.CONFIDENCE_NAMES[app.D["conf"][i][k]])
            n_text += 2
            assert sug == r["suggested_use"], (month, csv.index[k], sug, r["suggested_use"])     # exact
            if reason != r["reason"]:
                # only a percentage printed from a value on a rounding boundary (e.g. 94.5%) may differ, and by one point at most
                nums_a, nums_b = re.findall(r"\d+", reason), re.findall(r"\d+", r["reason"])
                assert re.sub(r"\d+", "#", reason) == re.sub(r"\d+", "#", r["reason"]), (month, reason, r["reason"])
                assert all(abs(int(x) - int(y)) <= 1 for x, y in zip(nums_a, nums_b)), (month, reason, r["reason"])
                n_diff += 1
        # ranking at the default penalty: the top 50 plots are the planner's top 50
        _, _, keep, shown, order = app.current(month, R.DEFAULT_PENALTY, ALL_USES)
        mine = set(app.D["plot_id"][order[:50]])
        theirs = set(csv.sort_values("rank").index[:50])
        assert len(mine & theirs) >= 48, (month, len(mine & theirs))
    # The planner printed its text from unrounded values; the app only has the 4-decimal copies in the tables.
    # A sentence can differ only in a percentage that sits exactly on a rounding boundary (e.g. 94.5%), by one point.
    assert n_diff <= 0.01 * (n_text // 2), f"{n_diff} plot reasons differ"
    print(f"ok 2: {n_text:,} reasons and suggestions in 21 months match the planner's (suggestions exactly; {n_diff} reasons differ by one rounding point in a percentage); statuses, confidence and green flags match; "
          "the top 50 plots agree")

    # every month, several penalties and filters run without error, and filters really filter
    for month in [m["value"] for m in app.MONTHS]:
        for pen in (0.0, 0.6, 1.0):
            out = app.update(month, pen, ALL_USES)
            assert out[1] is not None and len(out[5]) == 15 and out[4]["Plots (1 ha squares)"].iloc[-1] == app.D["plot_id"].size
            for f in out[1:4]:
                if f is not None:
                    app.plt.close(f)
    orchards = app.update("2025-07", 0.6, ["orchard"])
    assert (orchards[5]["Pre-war use"] == "Orchard").all()
    assert orchards[4]["Plots (1 ha squares)"].iloc[-1] == int((app.D["use_code"] == 2).sum())
    assert app.update("2025-07", 0.6, [])[1] is None
    print("ok 3: all 21 months x 3 penalties run; the pre-war-use filter filters; an empty filter is handled")

    # a stronger damage penalty must push damaged plots down the list
    def mean_share_of_top(pen):
        _, share, keep, shown, order = app.current("2025-07", pen, ALL_USES)
        return float(share[order[:200]].mean())
    assert mean_share_of_top(1.0) < mean_share_of_top(0.6) < mean_share_of_top(0.0)
    print("ok 4: a higher damage penalty puts less-damaged plots at the top")

    # numbers printed in the "How well" tab come from meta.json, which was copied from results/test_v2
    import json
    res = pd.read_csv(ROOT / "results" / "test_v2" / "scores.csv")
    for s in app.META["scores"]:
        r = res[(res["method"] == s["method"]) & (res["exam"] == s["exam"])].iloc[0]
        assert abs(r["macro_f1"] - s["macro_f1"]) < 1e-3
    gaps = json.loads((ROOT / "results" / "test_v2" / "gap.json").read_text())
    assert gaps == app.META["gaps"]
    assert "includes zero" in app.works_text_after() and "we do not claim a win there" in app.works_text_after()
    print("ok 5: test numbers in the app equal results/test_v2; the forward-exam caveat is printed")
    print("ALL APP CHECKS PASSED")


if __name__ == "__main__":
    main()
