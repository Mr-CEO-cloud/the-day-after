/*
 * proof.js - step 4 of the page: "What our AI says, and how to check it".
 *
 * Plain words: this is the part that lets a visitor check our scores instead of believing them.
 *   - "Run the model" downloads the real trained model (data/model.json) and a random sample of test
 *     patches it never saw (data/sample.bin, data/sample.json), runs the model on each patch in this
 *     browser (model.js), compares each answer with the UN's, and counts the results while you watch.
 *     The model has no randomness, so this gives the same counts every time.
 *   - "Run it on a random 1,000 of them" does the same on a fresh random draw of those patches, so the
 *     counts change a little each time. That shows the numbers are computed here, not replayed.
 *   - "Show a patch" draws one patch's greenness history and the model's answer for it.
 * The full-test scores and the audit trail below come from data/content.json, which is generated
 * from the results files in the code repository.
 *
 * This project is a prototype, not agricultural or humanitarian advice.
 */
(function () {
  "use strict";
  const DA = window.DA, $ = DA.$;

  DA.ready.push(function () {
    const { C, P, el } = DA, R = window.Rules, F = C.facts;
    const monthLabel = (m) => C.months.find((x) => x.value === m).label;

    // ------------------------------------------------------------------ AI map next to the UN map
    (function sideBySide() {
      let month = "2025-07", seen = false, token = 0;
      function show() {
        const i = C.months.findIndex((m) => m.value === month), un = P.un[i], mine = ++token;
        const draw = (photo) => {                                // first without the satellite picture, then on top of it
          const base = photo ? { cell: 4, photo, alpha: 0.85 } : { cell: 4, dark: true };
          DA.drawMap($("map-ai"), Object.assign({ colorOf: (p) => DA.STATUS_COLORS[DA.statusOf(i, p)] }, base));
          DA.drawMap($("map-ai-un"), Object.assign({ colorOf: (p) => (un[p] >= 0 ? DA.STATUS_COLORS[R.statusOf(un[p] / 10000)] : null) }, base));
        };
        draw(null);
        DA.loadImage(DA.monthPhoto(month)).then((img) => { if (mine === token) draw(img); }).catch(() => {});
        document.querySelectorAll("[data-ai]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.ai === month)));
      }
      document.querySelectorAll("[data-ai]").forEach((b) => b.addEventListener("click", () => { month = b.dataset.ai; show(); }));
      const io = new IntersectionObserver((entries) => { if (!seen && entries.some((e) => e.isIntersecting)) { seen = true; io.disconnect(); show(); } }, { rootMargin: "600px" });
      io.observe($("map-ai"));
    })();

    // ------------------------------------------------------------------ the model and the sample, downloaded once
    let kit = null;
    function loadKit() {
      if (!kit) {
        kit = Promise.all([fetch("data/model.json").then((r) => r.json()), fetch("data/sample.bin").then((r) => r.arrayBuffer()), DA.sample()])
          .then(([json, buf, S]) => ({ model: window.Model.create(json), X: new Float32Array(buf), S }));
      }
      return kit;
    }

    // ------------------------------------------------------------------ run the model on the sample
    (function runPanel() {
      const button = $("run-model"), randomButton = $("run-random"), status = $("run-status"), bar = $("run-progress"), boxes = $("matrices"), table = $("proof");
      const RANDOM_DRAW = 1000;                                 // how many patches the second button draws
      const cellNames = ["UN: not damaged, AI: not damaged", "UN: not damaged, AI: damaged", "UN: damaged, AI: not damaged", "UN: damaged, AI: damaged"];

      function drawMatrices(S, counts) {
        boxes.replaceChildren(...S.months.map((m, k) => {
          const c = counts[k].model, card = el("div", "matrix"), grid = el("div", "grid");
          card.append(el("h4", "", monthLabel(m)));
          grid.append(el("span"), el("span", "small", "AI says not damaged"), el("span", "small", "AI says damaged"),
            el("span", "small", "UN: not damaged"), el("span", "cell ok", c[0].toLocaleString("en-US")), el("span", "cell bad", c[1].toLocaleString("en-US")),
            el("span", "small", "UN: damaged"), el("span", "cell bad", c[2].toLocaleString("en-US")), el("span", "cell ok", c[3].toLocaleString("en-US")));
          const done = c[0] + c[1] + c[2] + c[3];
          card.append(grid, el("p", "score", done ? "AI score " + window.Model.macroF1(...c).toFixed(2) + ". Simple rule " + window.Model.macroF1(...counts[k].rule).toFixed(2) + "." : ""));
          return card;
        }));
      }

      function finish(S, counts, same, maxDiff, total, random) {
        const head = ["Month", "Patches"].concat(cellNames, ["AI score", "Simple rule score"]);
        table.replaceChildren();
        const thead = el("thead"), hr = el("tr");
        head.forEach((h, k) => { const c = el("th", k ? "num" : "", h); c.scope = "col"; hr.append(c); });
        thead.append(hr);
        const tbody = el("tbody");
        S.months.forEach((m, k) => {
          const c = counts[k].model, tr = el("tr");
          [monthLabel(m), String(c[0] + c[1] + c[2] + c[3]), String(c[0]), String(c[1]), String(c[2]), String(c[3]),
            window.Model.macroF1(...c).toFixed(3), window.Model.macroF1(...counts[k].rule).toFixed(3)].forEach((v, n) => tr.append(el("td", n ? "num" : "", v)));
          tbody.append(tr);
        });
        table.append(thead, tbody);
        table.hidden = false;
        const m = $("run-match");
        m.replaceChildren(el("strong", "", "Your browser's answers match the answers Python saved at the test on " + same.toLocaleString("en-US") + " of " + total.toLocaleString("en-US") + " patches. "),
          document.createTextNode((random ? "This was a fresh random draw: run it again and the counts change a little. " : "") +
            "On the full test of " + F.n_test_rows + " patches the scores are " + F.f1_v2_dec + " (Dec 2024) and " + F.f1_v2_jul + " (Jul 2025)."));
        status.textContent = random ? "Done: a random " + total.toLocaleString("en-US") + " of the " + S.n.toLocaleString("en-US") + " patches." : "Done: all " + total.toLocaleString("en-US") + " patches.";
        button.disabled = randomButton.disabled = false;
        if (!random) button.textContent = "Run all " + S.n.toLocaleString("en-US") + " again";
        document.body.dataset.proof = "done";
      }

      // draw: 0 = every patch, in order. Otherwise that many patches, picked at random now (a different set on every click).
      function run(draw) {
        button.disabled = randomButton.disabled = true;
        status.textContent = "Downloading the model and the test patches...";
        loadKit().then(({ model, X, S }) => {
          let ids = null, total = S.n;
          if (draw) {
            ids = Array.from({ length: S.n }, (_, k) => k);
            total = Math.min(draw, S.n);
            for (let k = 0; k < total; k++) { const j = k + Math.floor(Math.random() * (S.n - k)); [ids[k], ids[j]] = [ids[j], ids[k]]; }   // shuffle the first `total`
          }
          const counts = S.months.map(() => ({ model: [0, 0, 0, 0], rule: [0, 0, 0, 0] }));
          let n = 0, same = 0, maxDiff = 0;
          bar.hidden = false; boxes.hidden = false;
          const step = DA.query.has("autorun") ? total : Math.ceil(total / 60);     // the automatic test does it in one go
          (function chunk() {
            const end = Math.min(total, n + step);
            for (; n < end; n++) {
              const i = ids ? ids[n] : n;
              const off = i * S.n_features, p = model.probability(X, off), pred = p >= model.threshold ? 1 : 0;
              counts[S.month[i]].model[S.label[i] * 2 + pred]++;
              counts[S.month[i]].rule[S.label[i] * 2 + model.rule(X, off, S.months[S.month[i]])]++;
              if (pred === (S.py_prob[i] >= model.threshold ? 1 : 0)) same++;
              maxDiff = Math.max(maxDiff, Math.abs(p - S.py_prob[i]));
            }
            bar.value = n / total;
            status.textContent = "Running the model: " + n.toLocaleString("en-US") + " of " + total.toLocaleString("en-US") + " patches";
            drawMatrices(S, counts);
            if (n < total) requestAnimationFrame(chunk); else finish(S, counts, same, maxDiff, total, Boolean(draw));
          })();
        }).catch((e) => { status.textContent = "Could not load the model: " + e.message; button.disabled = randomButton.disabled = false; });
      }
      button.addEventListener("click", () => run(0));
      randomButton.addEventListener("click", () => run(RANDOM_DRAW));
      if (DA.query.has("autorun")) run(DA.query.get("autorun") === "random" ? RANDOM_DRAW : 0);
    })();

    // ------------------------------------------------------------------ look at one patch
    (function onePatch() {
      const box = $("pixel");
      function show(wrongOnly) {
        box.replaceChildren(el("p", "small", "Loading..."));
        loadKit().then(({ model, X, S }) => {
          const H = S.history;
          let ks = H.pixels.map((_, k) => k);
          const answer = (k) => { const pix = H.pixels[k], p = model.probability(X, pix * S.n_features); return { pix, p, pred: p >= model.threshold ? 1 : 0 }; };
          if (wrongOnly) ks = ks.filter((k) => { const a = answer(k); return a.pred !== S.label[a.pix]; });
          const k = ks[Math.floor(Math.random() * ks.length)], a = answer(k), un = S.label[a.pix], word = (v) => (v ? "damaged" : "not damaged");
          box.replaceChildren();
          const chart = el("div", "chart");
          DA.rhythmChart(chart, S, k, "Greenness history of one test patch");
          const v = el("p", "verdict");
          v.append(el("span", "", "AI: "), el("b", "", Math.round(a.p * 100) + "% chance damaged"), el("span", "", "so it says " + word(a.pred) + "."),
            el("span", "", "UN: " + word(un) + "."), el("span", a.pred === un ? "agree" : "disagree", a.pred === un ? "They agree." : "They disagree."));
          box.append(el("p", "small", "Test patch at " + S.lat[a.pix].toFixed(4) + ", " + S.lon[a.pix].toFixed(4) + ", judged for " + monthLabel(S.months[S.month[a.pix]]) + "."), chart, v);
        }).catch((e) => { box.replaceChildren(el("p", "small", "Could not load the model: " + e.message)); });
      }
      $("pixel-random").addEventListener("click", () => show(false));
      $("pixel-wrong").addEventListener("click", () => show(true));
    })();

    // ------------------------------------------------------------------ full-test scores as bars
    (function scores() {
      const methods = [["majority", "Always \"damaged\""], ["ndvi_drop", "First rule (one month)"], ["rule_war", "Season rule"], ["rf_all", "First model"], ["v2", "This model"]];
      const exams = [["dec", "Dec 2024"], ["jul", "Jul 2025"], ["fwd", "Forward in time (learn up to Dec 2024, judge Jul 2025)"]];
      $("scores").replaceChildren(...exams.map(([e, name]) => {
        const block = el("div");
        block.append(el("h4", "", name));
        methods.forEach(([m, label]) => {
          const v = Number(F["f1_" + m + "_" + e]), ci = F["ci_" + m + "_" + e].split(" to ").map(Number);
          const row = el("div", "bar-row" + (m === "v2" ? " main" : "")), track = el("div", "bar-track"), fill = el("div", "bar-fill"), range = el("div", "bar-ci");
          fill.style.width = v * 100 + "%";
          range.style.left = ci[0] * 100 + "%"; range.style.width = (ci[1] - ci[0]) * 100 + "%";
          track.append(fill, range);
          row.append(el("span", "", label), track, el("span", "", F["f1_" + m + "_" + e]));
          block.append(row);
        });
        return block;
      }));
    })();

    // ------------------------------------------------------------------ audit trail and the new-place test
    (function honesty() {
      const link = (text, href) => { const a = el("a", "", text); a.href = href; a.rel = "noopener"; return a; };
      const commit = (h) => link("code version " + h, C.repo + "/commit/" + h);
      const item = (when, parts) => { const li = el("li"); if (when) li.append(el("time", "", when)); parts.forEach((p) => li.append(typeof p === "string" ? document.createTextNode(p) : p)); return li; };
      $("trail").replaceChildren(
        item("", ["I split the land into 1 km squares and locked " + F.n_test_blocks + " of them away before training any model."]),
        item("", ["I wrote the rules for the first test and published them, and only then opened the test land."]),
        item(F.unlock_v1_time, ["First model scored on the test land, once (", commit(F.unlock_v1_commit), ")."]),
        item("", ["A second AI (OpenAI Codex) recomputed every score from the saved answers. All matched."]),
        item("", ["I chose the improved model using training land only, and published its test rules first."]),
        item(F.unlock_v2_time, ["Improved model scored on the test land, a second and last time (", commit(F.unlock_v2_commit), ")."]),
        item("", ["Codex recomputed every score again. All matched."]));
      $("link-activity").href = C.repo + "/activity";
      $("link-plan1").href = C.repo + "/blob/main/docs/test_plan.md";
      $("link-plan2").href = C.repo + "/blob/main/docs/test_plan_v2.md";
      $("link-answers").href = C.repo + "/tree/main/results/test_v2";

      const box = $("region"), T = C.region_test;
      if (!T) {
        box.append(el("p", "", "Running now. The model is being tested on Deir al-Balah, a neighbouring governorate it has never seen. The rules of that test are published before it is scored, and the result will appear here whatever it is."));
        return;
      }
      box.append(el("p", "", T.summary));
      const table = el("table"), thead = el("thead"), hr = el("tr"), tbody = el("tbody");
      T.columns.forEach((h, k) => { const c = el("th", k ? "num" : "", h); c.scope = "col"; hr.append(c); });
      thead.append(hr);
      T.rows.forEach((r) => { const tr = el("tr"); r.forEach((v, k) => tr.append(el("td", k ? "num" : "", v))); tbody.append(tr); });
      table.append(thead, tbody);
      const wrap = el("div", "scroll"); wrap.append(table);
      // the table and its note sit behind a toggle, so the card reads as two sentences first
      const more = el("details"); more.append(el("summary", "", "The full table"), wrap, el("p", "small", T.note));
      box.append(more);
    })();
  });
})();
