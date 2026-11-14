"""
07b_static_site.py - build the free, static version of the app (folder web/).

Plain words:
- Hugging Face no longer hosts free Gradio apps, but it hosts static web pages for free.
  A static page runs entirely in the visitor's browser: no server, nothing that can crash.
- This script prepares the page's data and text from the same precomputed files as the
  Gradio app (app/data, results/): it does not recompute anything about the model.
    web/data/plots.json      per plot: place, size, pre-war facts; per month: damaged share
                             (in 1/10,000), status / confidence / green-now code, UN share
    web/data/content.json    month list, boundary outline, banner text per month, and the
                             text of the "How well does it work?" and "About" tabs
    web/assets/*.png         the figures from the test results
- The page (web/index.html, app.js, rules.js, style.css) is written by hand.

Run:  .venv/Scripts/python.exe -I src/07b_static_site.py
"""
import re
import sys
import json
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ROOT, DISCLAIMER  # noqa: E402

sys.path.insert(0, str(ROOT / "app"))
import numpy as np  # noqa: E402
import app as gradio_app  # noqa: E402  (its text helpers are reused, so both versions say the same thing)
from markdown_it import MarkdownIt  # noqa: E402

WEB = ROOT / "web"
md = MarkdownIt("commonmark")


def html(markdown_text: str) -> str:
    out = md.render(markdown_text.strip())
    return re.sub(r"(https?://[^\s<]+[^\s<.,)])", r'<a href="\1" rel="noopener">\1</a>', out)


def ints(a, scale=1):
    return np.round(np.asarray(a, np.float64) * scale).astype(int).tolist()


def main():
    (WEB / "data").mkdir(parents=True, exist_ok=True)
    (WEB / "assets").mkdir(parents=True, exist_ok=True)
    D, META = gradio_app.D, gradio_app.META
    months = [m["value"] for m in META["months"]]
    share = D["share"]
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
        "months": months,
        "share": [ints(s, 10000) for s in share],                            # damaged share x 10,000
        "code": [c.tolist() for c in code],                                  # status + 3 x lower-confidence + 6 x green-now
        "un": [u.tolist() for u in un],                                      # UN share x 10,000, -1 where there is no UN map
    }
    (WEB / "data" / "plots.json").write_text(json.dumps(plots, separators=(",", ":")))

    works = (html(gradio_app.works_text()) + gradio_app.score_table().to_html(index=False, border=0, classes="scores", escape=True)
             + html("*Rows: methods. Columns: the three tests, macro F1 with a 95% range in brackets. 'Forward in time' means learning from UN maps up to Dec 2024 and judging Jul 2025 on unseen land.*")
             + html(gradio_app.works_text_after()))
    content = {
        "disclaimer": DISCLAIMER,
        "months": META["months"],
        "grid": META["grid"],
        "boundary": META["boundary"],
        "banners": {m: html(gradio_app.banner(m)) for m in months},
        "works_html": works,
        "about_html": html(gradio_app.ABOUT),
    }
    (WEB / "data" / "content.json").write_text(json.dumps(content, separators=(",", ":")))
    for name in ("scores", "confusion", "failures"):
        shutil.copy(ROOT / "app" / "assets" / f"test_{name}.png", WEB / "assets" / f"test_{name}.png")
    for f in ("data/plots.json", "data/content.json"):
        print(f"{f}: {(WEB / f).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
