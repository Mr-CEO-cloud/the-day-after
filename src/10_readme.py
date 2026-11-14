"""
10_readme.py - write README.md from a template, filling in every result from the results files.

Plain words: the rule for this project is "every number in the write-up comes from code output".
So the write-up is a template (docs/README.template.md, and docs/writeup.template.md for the
one-page version) with slots like {{f1_v2_dec}}. This script
fills each slot from the results files and stops with an error if a slot has no value. No result
in README.md is typed by hand. (Figures quoted from FAO's published table are cited in the text.)

Run:  .venv/Scripts/python.exe -I src/10_readme.py
"""
import re
import sys
import json
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, RESULTS  # noqa: E402

import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location("site07b", Path(__file__).resolve().parent / "07b_static_site.py")
site = importlib.util.module_from_spec(_spec)      # the web page's facts are reused, so page and write-up agree
_spec.loader.exec_module(site)

REGION = RESULTS / "area_deir_al_balah"


def region_facts() -> dict:
    """The new-place test (Deir al-Balah), formatted like the other facts."""
    f = {}
    sc = pd.read_csv(REGION / "region_test" / "scores.csv").set_index(["exam", "method"])
    gaps = json.loads((REGION / "region_test" / "gap.json").read_text())
    exams = {"dec": "same-date 2024-12", "jul": "same-date 2025-07", "fwd": "time 2024-12->2025-07"}
    for e, exam in exams.items():
        for m in ("majority", "ndvi_drop", "rule_war", "rf_all", "v2"):
            f[f"reg_f1_{m}_{e}"] = f"{sc.loc[(exam, m), 'macro_f1']:.2f}"
        f[f"reg_ci_v2_{e}"] = f"{sc.loc[(exam, 'v2'), 'macro_f1_ci_low']:.2f} to {sc.loc[(exam, 'v2'), 'macro_f1_ci_high']:.2f}"
        f[f"reg_gap_rule_{e}"] = f"{gaps[exam]['macro_f1_gap']:+.2f}"
        f[f"reg_gap_rule_{e}_range"] = f"{gaps[exam]['ci_low']:+.3f} to {gaps[exam]['ci_high']:+.3f}"
    area = json.loads((REGION / "region_test" / "area.json").read_text())
    f["reg_ha_un_dec"] = f"{area['2024-12']['damaged_ha_un']:,.0f}"
    f["reg_ha_v2_dec"] = f"{area['2024-12']['damaged_ha_model_calibrated']:,.0f}"
    return f


def subgroup_facts() -> dict:
    s = pd.read_csv(RESULTS / "test_v2" / "subgroups.csv")
    e = s[(s["dimension"] == "edge") & (s["date"] == "2024-12")].set_index("group")["model_error_rate"]
    return {"edge_err_dec": f"{e[[g for g in e.index if g.startswith('edge')][0]]:.0%}", "interior_err_dec": f"{e['interior']:.0%}"}


def fill(template: Path, out: Path, facts: dict) -> str:
    text = template.read_text(encoding="utf-8")
    slots = set(re.findall(r"\{\{([a-z0-9_]+)\}\}", text))
    missing = sorted(k for k in slots if not facts.get(k))
    assert not missing, f"{template.name}: no value for {missing}"
    for k in slots:
        text = text.replace("{{" + k + "}}", facts[k])
    assert "{{" not in text
    out.write_text(text, encoding="utf-8")
    words = len(re.sub(r"```.*?```", "", text, flags=re.S).split())
    print(f"{out.name} written: {len(slots)} numbers filled from the results files, about {words} words")
    return text


def main():
    facts = {**site.facts(), **region_facts(), **subgroup_facts()}
    fill(ROOT / "docs" / "README.template.md", ROOT / "README.md", facts)                 # the full write-up
    fill(ROOT / "docs" / "writeup.template.md", ROOT / "docs" / "writeup.md", facts)      # the one-page version for the submission form
    fill(ROOT / "docs" / "video_script.template.md", ROOT / "docs" / "video_script.md", facts)   # what to say in the 2-minute video


if __name__ == "__main__":
    main()
