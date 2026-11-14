/*
 * app.js - the interactive part of "The Day After" (static page).
 *
 * Plain words: the page loads two small precomputed files (data/plots.json, data/content.json).
 * Nothing is computed from satellites here and no model runs: the page only redraws maps and
 * tables when the visitor changes the month, the damage penalty or the pre-war-use filter.
 * The rules (status, priority, one-line reason) come from rules.js.
 *
 * This project is a prototype, not agricultural or humanitarian advice.
 */
(function () {
  "use strict";
  const R = window.Rules;
  const CELL = 4;                                              // CSS pixels per 100 m plot on the maps
  const STATUS_COLORS = ["#009E73", "#E69F00", "#CC79A7"];     // colour-blind-safe: intact, partly, mostly damaged
  const BLUES = [[0, [247, 251, 255]], [0.25, [198, 219, 239]], [0.5, [107, 174, 214]], [0.75, [33, 113, 181]], [1, [8, 48, 107]]];
  const $ = (id) => document.getElementById(id);

  let P, C;                                                    // plots, content
  let monthIndex = 0, penalty = R.DEFAULT_PENALTY;
  let cellOf;                                                  // grid cell -> plot index (or -1)
  let current = null;                                          // the numbers behind what is on screen

  // ------------------------------------------------------------------ helpers
  function blues(t) {
    t = Math.max(0, Math.min(1, t));
    for (let i = 1; i < BLUES.length; i++) {
      if (t <= BLUES[i][0]) {
        const [t0, c0] = BLUES[i - 1], [t1, c1] = BLUES[i], f = (t - t0) / (t1 - t0);
        return "rgb(" + c0.map((v, k) => Math.round(v + f * (c1[k] - v))).join(",") + ")";
      }
    }
    return "rgb(8,48,107)";
  }
  const statusOf = (i, p) => P.code[i][p] % 3;
  const confidenceOf = (i, p) => Math.floor(P.code[i][p] / 3) % 2;
  const greenOf = (i, p) => Math.floor(P.code[i][p] / 6);
  const hectares = (p) => P.px[p] * 0.01;
  const fmt1 = (x) => Number(R.pyFixed(x, 1)).toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
  const enabledUses = () => new Set([...document.querySelectorAll('input[name="use"]:checked')].map((b) => Number(b.value)));

  // ------------------------------------------------------------------ what is shown for one choice
  function compute() {
    const i = monthIndex, uses = enabledUses();
    const n = P.n, prio = new Float64Array(n), keep = new Uint8Array(n);
    const idx = [];
    for (let p = 0; p < n; p++) {
      if (!uses.has(P.use[p])) continue;
      keep[p] = 1;
      prio[p] = R.priority(P.prod[p], P.share[i][p] / 10000, penalty);
      idx.push(p);
    }
    idx.sort((a, b) => prio[b] - prio[a] || a - b);
    const rank = new Int32Array(n).fill(0);
    idx.forEach((p, k) => { rank[p] = k + 1; });
    current = { i, keep, prio, order: idx, rank };
    return current;
  }

  // ------------------------------------------------------------------ maps
  function setupCanvas(canvas) {
    const w = C.grid.cols * CELL, h = C.grid.rows * CELL, dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);
    return ctx;
  }

  function decorate(ctx) {
    ctx.strokeStyle = "#8a8a84";
    ctx.lineWidth = 1;
    C.boundary.forEach((ring) => {
      ctx.beginPath();
      ring.x.forEach((x, k) => { const px = x * CELL, py = ring.y[k] * CELL; if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
      ctx.stroke();
    });
    const h = C.grid.rows * CELL;
    ctx.strokeStyle = ctx.fillStyle = "#222222";
    ctx.lineWidth = 2;                                           // scale bar: 20 plots x 100 m = 2 km
    ctx.beginPath(); ctx.moveTo(4 * CELL, h - 3 * CELL); ctx.lineTo(24 * CELL, h - 3 * CELL); ctx.stroke();
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("2 km", 14 * CELL, h - 4.5 * CELL);
    ctx.lineWidth = 1.5;                                         // north arrow
    ctx.beginPath(); ctx.moveTo(8 * CELL, 22 * CELL); ctx.lineTo(8 * CELL, 9 * CELL); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(8 * CELL, 8 * CELL); ctx.lineTo(8 * CELL - 4, 8 * CELL + 8); ctx.lineTo(8 * CELL + 4, 8 * CELL + 8); ctx.closePath(); ctx.fill();
    ctx.fillText("N", 8 * CELL, 7 * CELL);
  }

  function drawMap(canvas, colorOf, markers) {
    const ctx = setupCanvas(canvas);
    for (let p = 0; p < P.n; p++) {
      const c = colorOf(p);
      if (!c) continue;
      ctx.fillStyle = c;
      ctx.fillRect(P.col[p] * CELL, P.row[p] * CELL, CELL, CELL);
    }
    decorate(ctx);
    if (markers) {
      ctx.strokeStyle = "#D55E00"; ctx.lineWidth = 2;
      markers.forEach((p) => { ctx.beginPath(); ctx.arc((P.col[p] + 0.5) * CELL, (P.row[p] + 0.5) * CELL, 6, 0, 2 * Math.PI); ctx.stroke(); });
    }
  }

  function renderMaps(cur) {
    const i = cur.i, month = C.months[i];
    drawMap($("map-status"), (p) => (cur.keep[p] ? STATUS_COLORS[statusOf(i, p)] : null));
    drawMap($("map-priority"), (p) => (cur.keep[p] ? blues(cur.prio[p]) : null), cur.order.slice(0, 15));
    $("cap-status").textContent = "Estimated status, " + month.label;
    $("cap-priority").textContent = "Recovery priority, " + month.label + " (damage penalty " + penalty.toFixed(1) + ")";
    $("card-un").hidden = !month.has_un_map;
    if (month.has_un_map) {
      const un = P.un[i];
      drawMap($("map-un"), (p) => (cur.keep[p] && un[p] >= 0 ? STATUS_COLORS[R.statusOf(un[p] / 10000)] : null));
      $("cap-un").textContent = "UN damage map for the same plots, " + month.label;
    }
    $("map-status").setAttribute("aria-label", "Map of estimated status of each 1 hectare plot, " + month.label);
  }

  // ------------------------------------------------------------------ tables
  function td(text, cls) { const c = document.createElement("td"); c.textContent = text; if (cls) c.className = cls; return c; }
  function th(text, cls) { const c = document.createElement("th"); c.scope = "col"; c.textContent = text; if (cls) c.className = cls; return c; }
  function fillTable(table, headers, rows, numCols) {
    table.replaceChildren();
    const head = document.createElement("thead"), hr = document.createElement("tr");
    headers.forEach((h, k) => hr.appendChild(th(h, numCols.has(k) ? "num" : "")));
    head.appendChild(hr);
    const body = document.createElement("tbody");
    rows.forEach((r) => { const tr = document.createElement("tr"); r.forEach((v, k) => tr.appendChild(td(v, numCols.has(k) ? "num" : ""))); body.appendChild(tr); });
    table.append(head, body);
  }

  function describe(i, p) {
    const s = P.share[i][p] / 10000, st = R.STATUS_NAMES[statusOf(i, p)], use = R.USE_NAMES[P.use[p]];
    const conf = R.CONFIDENCE_NAMES[confidenceOf(i, p)], green = R.GREEN_NAMES[greenOf(i, p)];
    return {
      share: s, status: st, use, conf, green,
      reason: R.reason(use, P.prod[p], s, hectares(p), st),
      suggestion: R.suggestion(st, use, P.built[p], green, conf),
      level: R.productivityLabel(P.prod[p]),
    };
  }

  function renderTables(cur) {
    const i = cur.i;
    const names = R.STATUS_NAMES, agg = names.map(() => ({ plots: 0, ha: 0, dmg: 0 }));
    let totalHa = 0, totalDmg = 0, totalPlots = 0;
    cur.order.forEach((p) => {
      const a = agg[statusOf(i, p)], ha = hectares(p), d = (P.share[i][p] / 10000) * ha;
      a.plots++; a.ha += ha; a.dmg += d; totalPlots++; totalHa += ha; totalDmg += d;
    });
    const rows = names.map((nm, k) => [R.capitalize(nm), agg[k].plots.toLocaleString("en-US"), fmt1(agg[k].ha),
      totalHa ? R.pyFixed(100 * agg[k].ha / totalHa, 1) : "0.0", fmt1(agg[k].dmg)]);
    rows.push(["All rated plots", totalPlots.toLocaleString("en-US"), fmt1(totalHa), totalHa ? "100.0" : "0.0", fmt1(totalDmg)]);
    fillTable($("hect"), ["Status", "Plots (1 ha squares)", "Cropland (ha)", "Share of rated cropland (%)", "Estimated damaged since Oct 2023 (ha)"], rows, new Set([1, 2, 3, 4]));

    const top = cur.order.slice(0, 15).map((p, k) => {
      const d = describe(i, p);
      return [String(k + 1), P.lat[p].toFixed(4) + ", " + P.lon[p].toFixed(4), fmt1(hectares(p)), R.capitalize(d.status), R.percent0(d.share),
        R.capitalize(d.use), R.capitalize(d.level), R.capitalize(d.green), d.reason, d.suggestion];
    });
    fillTable($("top"), ["Rank", "Location (lat, lon)", "Size (ha)", "Status", "Est. damaged", "Pre-war use", "Pre-war greenness",
      "Green now vs normal?", "Why it ranks here", "Suggested first step"], top, new Set([0, 2]));
  }

  function syncUrl() {
    const uses = [...enabledUses()].sort().join(",");
    const q = "?month=" + C.months[monthIndex].value + "&penalty=" + penalty.toFixed(1) + "&uses=" + uses;
    history.replaceState(null, "", location.pathname + q + location.hash);
  }

  function render() {
    syncUrl();
    const month = C.months[monthIndex], cur = compute();
    const b = $("banner");
    b.innerHTML = C.banners[month.value];
    b.classList.toggle("warn", !month.has_un_map);
    if (!cur.order.length) b.innerHTML += "<p><strong>Tick at least one pre-war use above to see plots.</strong></p>";
    renderMaps(cur);
    renderTables(cur);
  }

  // ------------------------------------------------------------------ hover details
  function attachHover(canvas) {
    const tip = $("tip");
    function show(ev) {
      if (!current) return;
      const r = canvas.getBoundingClientRect();
      const col = Math.floor(((ev.clientX - r.left) / r.width) * C.grid.cols), row = Math.floor(((ev.clientY - r.top) / r.height) * C.grid.rows);
      const p = col >= 0 && row >= 0 && col < C.grid.cols && row < C.grid.rows ? cellOf[row * C.grid.cols + col] : -1;
      if (p < 0 || !current.keep[p]) { tip.hidden = true; return; }
      const d = describe(current.i, p), un = P.un[current.i][p];
      tip.textContent = "";
      const lines = [
        "Plot at " + P.lat[p].toFixed(4) + ", " + P.lon[p].toFixed(4) + " (" + fmt1(hectares(p)) + " ha)",
        R.capitalize(d.status) + ", about " + R.percent0(d.share) + " damaged since Oct 2023",
        R.capitalize(d.use) + ", " + d.level + " pre-war greenness",
        "Priority rank " + current.rank[p] + " of " + current.order.length,
      ];
      if (un >= 0) lines.push("UN map: " + R.percent0(un / 10000) + " damaged");
      lines.forEach((t, k) => { const s = document.createElement("div"); s.textContent = t; if (k === 0) s.style.fontWeight = "600"; tip.appendChild(s); });
      tip.hidden = false;
      const w = tip.offsetWidth;
      tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - w - 8) + "px";
      tip.style.top = ev.clientY + 14 + "px";
    }
    canvas.addEventListener("mousemove", show);
    canvas.addEventListener("click", show);
    canvas.addEventListener("mouseleave", () => { tip.hidden = true; });
  }

  // ------------------------------------------------------------------ CSV download
  function downloadCsv() {
    const cur = current, i = cur.i, month = C.months[i].value;
    const q = (s) => '"' + String(s).replace(/"/g, '""') + '"';
    const head = ["rank", "lat", "lon", "cropland_ha", "status", "estimated_damaged_share", "priority", "pre_war_use",
      "pre_war_greenness_percentile", "green_now_vs_normal", "confidence", "reason", "suggested_first_step", "note"];
    const lines = [head.join(",")];
    cur.order.forEach((p, k) => {
      const d = describe(i, p);
      lines.push([k + 1, P.lat[p].toFixed(5), P.lon[p].toFixed(5), hectares(p).toFixed(2), d.status, d.share.toFixed(3), cur.prio[p].toFixed(4),
        d.use, P.prod[p].toFixed(3), d.green, d.conf, q(d.reason), q(d.suggestion), q(C.disclaimer)].join(","));
    });
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "day_after_" + month + "_ranked_plots.csv";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  // ------------------------------------------------------------------ tabs and start-up
  function showTab(name) {
    ["plan", "works", "about"].forEach((t) => {
      $("tab-" + t).hidden = t !== name;
      $("t-" + t).setAttribute("aria-selected", String(t === name));
    });
    if (location.hash.slice(1) !== name) history.replaceState(null, "", "#" + name);
  }

  function legend(id) {
    $(id).replaceChildren(...R.STATUS_NAMES.map((n, k) => {
      const li = document.createElement("li"), sw = document.createElement("span");
      sw.className = "sw"; sw.style.background = STATUS_COLORS[k];
      li.append(sw, document.createTextNode(n));
      return li;
    }));
  }

  async function start() {
    try {
      [P, C] = await Promise.all([fetch("data/plots.json").then((r) => r.json()), fetch("data/content.json").then((r) => r.json())]);
    } catch (e) {
      $("loading").hidden = true;
      $("error").hidden = false;
      $("error").textContent = "The data files could not be loaded. If you opened this file directly from a folder, serve it with a web server (for example: python -m http.server) or use the public link.";
      return;
    }
    cellOf = new Int32Array(C.grid.rows * C.grid.cols).fill(-1);
    for (let p = 0; p < P.n; p++) cellOf[P.row[p] * C.grid.cols + P.col[p]] = p;
    const sel = $("month");
    C.months.forEach((m, k) => {
      const o = document.createElement("option");
      o.value = String(k);
      o.textContent = m.label + (m.has_un_map ? "  (UN map available)" : "  (no UN map: model estimate only)");
      sel.appendChild(o);
    });
    monthIndex = C.months.findIndex((m) => m.value === "2025-07");
    // shareable links: ?month=2026-04&penalty=0.4&uses=0,2   (uses: 0 seasonal crops, 1 mixed, 2 orchard)
    const q = new URLSearchParams(location.search);
    const k = C.months.findIndex((m) => m.value === q.get("month"));
    if (k >= 0) monthIndex = k;
    if (q.has("penalty") && isFinite(Number(q.get("penalty")))) penalty = Math.min(1, Math.max(0, Math.round(Number(q.get("penalty")) * 10) / 10));
    if (q.has("uses")) {
      const want = new Set(q.get("uses").split(",").filter((x) => x !== "").map(Number));
      document.querySelectorAll('input[name="use"]').forEach((b) => { b.checked = want.has(Number(b.value)); });
    }
    sel.value = String(monthIndex);
    $("penalty").value = String(penalty);
    $("penalty-out").textContent = penalty.toFixed(1);
    sel.addEventListener("change", () => { monthIndex = Number(sel.value); render(); });
    $("penalty").addEventListener("input", (e) => { penalty = Number(e.target.value); $("penalty-out").textContent = penalty.toFixed(1); render(); });
    document.querySelectorAll('input[name="use"]').forEach((b) => b.addEventListener("change", render));
    $("download").addEventListener("click", downloadCsv);
    legend("legend-status"); legend("legend-un");
    ["map-status", "map-priority", "map-un"].forEach((id) => attachHover($(id)));
    $("tab-works").innerHTML = C.works_html;
    $("tab-about").innerHTML = C.about_html;
    const figs = [["test_scores.png", "Scores of each method on unseen land (macro F1, with 95% ranges)", "Scores on unseen land"],
      ["test_confusion.png", "Where each UN class ends up, for the greenness-drop rule and for the v2 model", "Where each UN class ends up"],
      ["test_failures.png", "The four places with the most mistakes, chosen by a fixed rule", "The four places with the most mistakes, chosen by a fixed rule"]];
    figs.forEach(([f, alt, cap]) => {
      const fig = document.createElement("figure"), img = document.createElement("img"), fc = document.createElement("figcaption");
      img.src = "assets/" + f; img.alt = alt; img.loading = "lazy"; fc.textContent = cap;
      fig.append(img, fc); $("tab-works").appendChild(fig);
    });
    const dis = document.createElement("p"); dis.className = "disclaimer"; dis.textContent = C.disclaimer; $("tab-works").appendChild(dis);
    const dis2 = dis.cloneNode(true); $("tab-about").appendChild(dis2);
    $("t-plan").addEventListener("click", () => showTab("plan"));
    $("t-works").addEventListener("click", () => showTab("works"));
    $("t-about").addEventListener("click", () => showTab("about"));
    $("loading").hidden = true;
    const start_tab = ["plan", "works", "about"].includes(location.hash.slice(1)) ? location.hash.slice(1) : "plan";
    showTab(start_tab);
    render();
  }
  start();
})();
