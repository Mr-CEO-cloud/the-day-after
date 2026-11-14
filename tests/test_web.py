"""
test_web.py - checks the static web page (folder web/) against independent Python code.

Plain words: the page does its own calculations in JavaScript. This test makes sure they are right:
  1. the page's data files equal the app's packed data (which equals the planner's results);
  2. the page's rules.js prints exactly the same sentence as the Python rules, for every plot in every month;
  3. the page, opened in a real (headless) browser, shows the same hectare table and the same top-15
     plots, with the same texts, as a separate Python calculation, for several choices of month,
     penalty and pre-war-use filter;
  4. the disclaimer is on the page.
Check 3 needs Microsoft Edge (installed with Windows) and starts a tiny local web server.

Run:  .venv/Scripts/python.exe -I tests/test_web.py
"""
import sys
import json
import hashlib
import subprocess
import threading
import http.server
import socketserver
from functools import partial
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import planner_rules as R  # noqa: E402
from config import DISCLAIMER  # noqa: E402

WEB = ROOT / "web"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def load():
    return json.loads((WEB / "data" / "plots.json").read_text())


def check_data(P):
    app_data = np.load(ROOT / "app" / "data" / "plots.npz")
    assert P["n"] == len(app_data["plot_id"]) and P["plot_id"] == app_data["plot_id"].tolist()
    assert np.allclose(np.array(P["share"]) / 10000, app_data["share"], atol=6e-5)
    code = app_data["status"].astype(int) + 3 * app_data["conf"].astype(int) + 6 * app_data["green"].astype(int)
    assert (np.array(P["code"]) == code).all()
    assert (np.array(P["px"]) == app_data["cropland_px"]).all() and (np.array(P["use"]) == app_data["use_code"]).all()
    un = np.array(P["un"], float)
    assert np.allclose(np.where(un < 0, np.nan, un / 10000), app_data["un_share"], atol=6e-5, equal_nan=True)
    print("ok 1: the page's data equals the app's packed data (shares within 0.00006)")


def check_rules(P):
    out = subprocess.run(["node", str(ROOT / "tests" / "web_rules_check.js")], capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    h = hashlib.sha256()
    n = 0
    for i in range(len(P["months"])):
        for p in range(P["n"]):
            c = P["code"][i][p]
            status, conf, green = R.STATUS_NAMES[c % 3], R.CONFIDENCE_NAMES[(c // 3) % 2], R.GREEN_NAMES[c // 6]
            use = R.USE_NAMES[P["use"][p]]
            share = P["share"][i][p] / 10000
            h.update((R.reason(use, P["prod"][p], share, P["px"][p] * 0.01, status) + "\n"
                      + R.suggestion(status, use, P["built"][p], green, conf) + "\n").encode())
            n += 2
    assert n == js["texts"] and h.hexdigest() == js["sha256"], "rules.js and the Python rules print different text"
    print(f"ok 2: rules.js and the Python rules print identical text for all {n:,} sentences (21 months x {P['n']:,} plots)")


class Table(HTMLParser):
    """Collect the rows of the <table id=...> elements of a rendered page."""
    def __init__(self):
        super().__init__()
        self.tables, self._t, self._row, self._cell = {}, None, None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._t = a.get("id")
            self.tables[self._t] = []
        elif tag == "tr" and self._t:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.tables[self._t].append(self._row)
            self._row = None
        elif tag == "table":
            self._t = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def expected(P, month, penalty, uses):
    """What the page should show, computed separately in Python from the page's own data."""
    i = P["months"].index(month)
    prod, px, built = np.array(P["prod"]), np.array(P["px"]), np.array(P["built"])
    share = np.array(P["share"][i]) / 10000
    use = np.array(P["use"])
    keep = np.isin(use, uses)
    prio = prod * (1 - penalty * share)
    idx = [p for p in np.flatnonzero(keep)]
    idx.sort(key=lambda p: (-prio[p], p))
    code = np.array(P["code"][i])
    status = code % 3
    ha = px * 0.01
    hect = []
    for k in range(3):
        m = keep & (status == k)
        hect.append([int(m.sum()), float(ha[m].sum()), float((share[m] * ha[m]).sum())])
    top = []
    for rank, p in enumerate(idx[:15], 1):
        st, u = R.STATUS_NAMES[status[p]], R.USE_NAMES[use[p]]
        conf, green = R.CONFIDENCE_NAMES[(code[p] // 3) % 2], R.GREEN_NAMES[code[p] // 6]
        top.append([str(rank), f"{P['lat'][p]:.4f}, {P['lon'][p]:.4f}", f"{ha[p]:.1f}", st.capitalize(), f"{share[p]:.0%}", u.capitalize(),
                    R.productivity_label(P["prod"][p]).capitalize(), green.capitalize(),
                    R.reason(u, P["prod"][p], share[p], ha[p], st), R.suggestion(st, u, built[p], green, conf)])
    return hect, float(ha[keep].sum()), float((share[keep] * ha[keep]).sum()), top


def render(port, month, penalty, uses):
    url = f"http://127.0.0.1:{port}/index.html?month={month}&penalty={penalty}&uses={','.join(map(str, uses))}"
    out = subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--virtual-time-budget=9000", "--dump-dom", url],
                         capture_output=True, text=True, timeout=120).stdout
    t = Table()
    t.feed(out)
    return out, t.tables


def num(s):
    return float(s.replace(",", ""))


def check_browser(P):
    if not EDGE.exists():
        print("skip 3: Microsoft Edge not found, the browser check was not run")
        return
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args, **kwargs):
            pass
    handler = partial(Quiet, directory=str(WEB))
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as srv:
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        states = [("2025-07", 0.6, [0, 1, 2]), ("2026-04", 0.0, [2]), ("2023-10", 1.0, [0, 1]), ("2024-12", 0.3, [0, 1, 2]), ("2025-11", 0.6, [1])]
        for month, pen, uses in states:
            html, tables = render(port, month, pen, uses)
            hect, tot_ha, tot_dmg, top = expected(P, month, pen, uses)
            got_h, got_top = tables["hect"][1:], tables["top"][1:]
            for k in range(3):
                assert int(num(got_h[k][1])) == hect[k][0], (month, k, got_h[k], hect[k])
                assert abs(num(got_h[k][2]) - hect[k][1]) < 0.06 and abs(num(got_h[k][4]) - hect[k][2]) < 0.06, (month, k, got_h[k], hect[k])
            assert int(num(got_h[3][1])) == sum(h[0] for h in hect) and abs(num(got_h[3][2]) - tot_ha) < 0.06 and abs(num(got_h[3][4]) - tot_dmg) < 0.06
            assert len(got_top) == len(top), (month, len(got_top), len(top))
            for g, e in zip(got_top, top):
                assert g == e, (month, pen, uses, g, e)
            assert DISCLAIMER in html
            assert 'id="error" class="note error" hidden' in html, "the page reported an error"
            print(f"ok 3 ({month}, penalty {pen}, uses {uses}): the page's hectare table and top {len(top)} plots, with their texts, equal the Python calculation")

        # the "check it yourself" panel, run for real in the browser: its counts must equal a separate Python count
        S, counts = sample_counts()
        url = f"http://127.0.0.1:{port}/index.html?autorun=1"
        html = subprocess.run([str(EDGE), "--headless", "--disable-gpu", "--virtual-time-budget=20000", "--dump-dom", url],
                              capture_output=True, text=True, timeout=180).stdout
        t = Table()
        t.feed(html)
        rows = t.tables["proof"][1:]
        assert 'data-proof="done"' in html and len(rows) == len(S["months"])
        for k, row in enumerate(rows):
            got = [int(v) for v in row[2:6]]
            assert got == counts[k], (row, counts[k])
            assert abs(float(row[6]) - macro_f1(*counts[k])) < 0.0006
        n = f"{S['n']:,}"
        assert f"on {n} of {n} patches" in html, "the browser's answers do not all match the saved Python answers"
        print(f"ok 6: in a real browser the live model run gives the same confusion counts as Python on all {n} sampled test patches")
        srv.shutdown()


def macro_f1(tn, fp, fn, tp):
    f = lambda a, b, c: 2 * a / (2 * a + b + c) if 2 * a + b + c else 0.0  # noqa: E731
    return (f(tp, fp, fn) + f(tn, fn, fp)) / 2


def sample_counts():
    """Confusion counts per month for the page's sample, from the answers PYTHON saved at the test look."""
    S = json.loads((WEB / "data" / "sample.json").read_text())
    counts = [[0, 0, 0, 0] for _ in S["months"]]
    for m, y, p in zip(S["month"], S["label"], S["py_prob"]):
        counts[m][y * 2 + int(p >= 0.5)] += 1                 # tn, fp, fn, tp
    return S, counts


def check_model():
    """The model exported for the page must behave exactly like the Python model."""
    js = json.loads(subprocess.run(["node", str(ROOT / "tests" / "web_model_check.js")], capture_output=True, text=True, check=True).stdout)
    S, counts = sample_counts()
    assert js["n"] == S["n"] == 6000 and js["max_prob_diff"] < 1e-6 and js["max_cal_diff"] < 1e-5
    assert js["decision_diff"] == 0 and js["rule_diff"] == 0
    assert [c["model"] for c in js["confusion"]] == counts
    # the sample really is test land scored at the second look: its saved answers are in the full results file
    import gzip
    import io
    import pandas as pd
    full = pd.read_csv(io.BytesIO(gzip.decompress((ROOT / "results" / "test_v2" / "predictions_main.csv.gz").read_bytes())))
    key = full.set_index(["date", "row", "col"])
    idx = list(zip([S["months"][m] for m in S["month"]], S["row"], S["col"]))
    hit = key.loc[idx]
    assert (hit["label"].to_numpy() == np.array(S["label"])).all() and (hit["pred__v2"].to_numpy() == (np.array(S["py_prob"]) >= 0.5)).all()
    print(f"ok 5: the exported model gives Python's probability on all {js['n']:,} sampled test patches "
          f"(largest difference {js['max_prob_diff']:.1e}, 0 different decisions); the sample is part of the saved test answers")


def check_facts():
    """Every number slot in the page is filled from content.json, and content.json agrees with the results files."""
    import re
    import pandas as pd
    index = (WEB / "index.html").read_text(encoding="utf-8")
    content = json.loads((WEB / "data" / "content.json").read_text())
    facts = content["facts"]
    used = set(re.findall(r'data-fact="([a-z0-9_]+)"', index))
    js_used = set(re.findall(r"F\.([a-z0-9_]+)", (WEB / "proof.js").read_text(encoding="utf-8") + (WEB / "app.js").read_text(encoding="utf-8")))
    missing = [k for k in used | js_used if not facts.get(k)]
    assert not missing, f"empty number slots: {missing}"
    sc = pd.read_csv(ROOT / "results" / "test_v2" / "scores.csv").set_index(["exam", "method"])
    assert facts["f1_v2_dec"] == f"{sc.loc[('same-date 2024-12', 'v2'), 'macro_f1']:.2f}"
    assert facts["f1_rule_war_fwd"] == f"{sc.loc[('time 2024-12->2025-07', 'rule_war'), 'macro_f1']:.2f}"
    assert facts["gap_rule_fwd_range"].startswith("-"), "the forward-exam range must show that it includes zero"
    assert index.count(DISCLAIMER) >= 2 and content["disclaimer"] == DISCLAIMER
    assert "not clearly better than the simple rule" in index and "nothing here can be checked" in (WEB / "app.js").read_text(encoding="utf-8")
    assert "sdk: static" in (WEB / "README.md").read_text(encoding="utf-8")
    print(f"ok 4: all {len(used | js_used)} number slots are filled from the results files; the disclaimer and both caveats are on the page")


def main():
    P = load()
    check_data(P)
    check_rules(P)
    check_facts()
    check_model()
    check_browser(P)
    print("ALL WEB CHECKS PASSED")


if __name__ == "__main__":
    main()
