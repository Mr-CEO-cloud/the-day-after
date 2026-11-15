/*
 * app.js - shared tools and the "Explore the map" part of The Day After.
 *
 * Plain words: the page loads small precomputed files (data/content.json, data/plots.json). Nothing is
 * fetched from a satellite service and no training happens here. This file fills the page's number
 * slots from content.json, draws maps on top of real satellite pictures, and redraws the tables when
 * the visitor changes the month, the damage penalty or the pre-war-use filter. The planner's rules
 * (status, priority, the one-line reason) come from rules.js.
 *
 * This project is a prototype, not agricultural or humanitarian advice.
 */
(function () {
  "use strict";
  const R = window.Rules;
  const STATUS_COLORS = ["#009E73", "#E69F00", "#CC79A7"];     // colour-blind-safe: intact, partly, mostly damaged
  const BLUES = [[0, [247, 251, 255]], [0.25, [198, 219, 239]], [0.5, [107, 174, 214]], [0.75, [33, 113, 181]], [1, [8, 48, 107]]];
  const $ = (id) => document.getElementById(id);
  const DA = (window.DA = { $, STATUS_COLORS, ready: [], listeners: [] });

  window.addEventListener("error", (e) => {                     // show problems instead of failing silently
    const box = $("error");
    if (box) { box.hidden = false; box.textContent = "Something went wrong on this page: " + (e.message || "unknown error"); }
  });

  let P, C;
  const AR = window.RulesAr;                                   // the same sentences in Arabic
  const state = (DA.state = { monthIndex: 0, penalty: R.DEFAULT_PENALTY, view: "priority", selected: -1, lang: "en" });
  let cellOf, current = null;

  // ------------------------------------------------------------------ helpers shared with story.js and proof.js
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
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };

  const images = new Map();                                     // each picture is downloaded once, when first needed
  function loadImage(url) {
    if (!images.has(url)) {
      images.set(url, new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error("could not load " + url));
        img.src = url;
      }));
    }
    return images.get(url);
  }
  const monthPhoto = (m) => "img/month/" + m + ".webp";
  const beforePhoto = (m) => "img/before/" + m.slice(5) + ".webp";      // the same calendar month in 2022

  // Draw one map: an optional satellite picture, then one coloured square per plot, then the outline.
  function drawMap(canvas, o) {
    const cell = o.cell || 4, w = C.grid.cols * cell, h = C.grid.rows * cell, dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);
    if (o.photo) {                                               // the picture covers image_px 10 m pixels; one plot = 10 of them
      ctx.drawImage(o.photo, 0, 0, o.photo.naturalWidth, o.photo.naturalHeight, 0, 0, (C.image_px.width / 10) * cell, (C.image_px.height / 10) * cell);
    }
    if (o.colorOf) {
      ctx.globalAlpha = o.alpha === undefined ? 1 : o.alpha;
      for (let p = 0; p < P.n; p++) {
        const c = o.colorOf(p);
        if (!c) continue;
        ctx.fillStyle = c;
        ctx.fillRect(P.col[p] * cell, P.row[p] * cell, cell, cell);
      }
      ctx.globalAlpha = 1;
    }
    ctx.strokeStyle = o.photo ? "#ffffff" : "#8a7f6f";
    ctx.lineWidth = o.photo ? 1.5 : 1;
    C.boundary.forEach((ring) => {
      ctx.beginPath();
      ring.x.forEach((x, k) => { const px = x * cell, py = ring.y[k] * cell; if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
      ctx.stroke();
    });
    const ink = o.photo ? "#ffffff" : "#2a241c";
    if (o.photo) { ctx.shadowColor = "rgba(0,0,0,0.85)"; ctx.shadowBlur = 4; }   // keep the scale bar readable on a picture
    ctx.strokeStyle = ctx.fillStyle = ink;
    ctx.lineWidth = 2;                                           // scale bar: 20 plots x 100 m = 2 km
    ctx.beginPath(); ctx.moveTo(4 * cell, h - 3 * cell); ctx.lineTo(24 * cell, h - 3 * cell); ctx.stroke();
    ctx.font = (cell >= 6 ? 13 : 11) + "px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("2 km", 14 * cell, h - 4.5 * cell);
    ctx.lineWidth = 1.5;                                         // north arrow
    ctx.beginPath(); ctx.moveTo(8 * cell, 22 * cell); ctx.lineTo(8 * cell, 9 * cell); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(8 * cell, 8 * cell); ctx.lineTo(8 * cell - 4, 8 * cell + 8); ctx.lineTo(8 * cell + 4, 8 * cell + 8); ctx.closePath(); ctx.fill();
    ctx.fillText("N", 8 * cell, 7 * cell);
    ctx.shadowBlur = 0; ctx.shadowColor = "transparent";
    if (o.markers) {
      ctx.strokeStyle = "#D55E00"; ctx.lineWidth = 2;
      o.markers.forEach((p) => { ctx.beginPath(); ctx.arc((P.col[p] + 0.5) * cell, (P.row[p] + 0.5) * cell, cell * 1.5, 0, 2 * Math.PI); ctx.stroke(); });
    }
    if (o.selected >= 0) {
      const p = o.selected;
      ctx.strokeStyle = "#111111"; ctx.lineWidth = 3;
      ctx.strokeRect(P.col[p] * cell - 2, P.row[p] * cell - 2, cell + 4, cell + 4);
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.5;
      ctx.strokeRect(P.col[p] * cell - 2, P.row[p] * cell - 2, cell + 4, cell + 4);
    }
  }

  function statusLegend(list) {
    list.replaceChildren(...R.STATUS_NAMES.map((n, k) => {
      const li = el("li"), sw = el("span", "sw");
      sw.style.background = STATUS_COLORS[k];
      li.append(sw, document.createTextNode(n));
      return li;
    }));
  }

  // A small line chart of one patch's greenness: every month against its own normal for that month.
  function rhythmChart(box, S, k, caption) {
    const H = S.history, names = H.months, pix = H.pixels[k];
    const upTo = names.indexOf(S.months[S.month[pix]]);
    const series = H.ndvi_x1000[k], normal = H.normal_x1000[k];
    const W = 600, Ht = 250, L = 34, Rr = 12, T = 14, B = 26;
    const x = (i) => L + (i / upTo) * (W - L - Rr), y = (v) => T + (1 - Math.max(0, Math.min(0.9, v)) / 0.9) * (Ht - T - B);
    const NS = "http://www.w3.org/2000/svg";
    const s = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const a in attrs) e.setAttribute(a, attrs[a]); if (text) e.textContent = text; return e; };
    const svg = s("svg", { viewBox: "0 0 " + W + " " + Ht, role: "img", "aria-label": caption });
    const war = names.indexOf("2023-10");
    svg.append(s("rect", { class: "since", x: x(war), y: T, width: x(upTo) - x(war), height: Ht - T - B }));
    [0, 0.3, 0.6, 0.9].forEach((v) => {
      svg.append(s("line", { class: "axis", x1: L, x2: W - Rr, y1: y(v), y2: y(v) }), s("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, v.toFixed(1)));
    });
    names.forEach((n, i) => { if (i <= upTo && n.slice(5) === "01") svg.append(s("text", { x: x(i), y: Ht - 8, "text-anchor": "middle" }, n.slice(0, 4))); });
    const path = (get) => {
      let d = "", pen = false;
      for (let i = 0; i <= upTo; i++) {
        const v = get(i);
        if (v === null || v === undefined) { pen = false; continue; }
        d += (pen ? "L" : "M") + x(i).toFixed(1) + " " + y(v / 1000).toFixed(1);
        pen = true;
      }
      return d;
    };
    svg.append(s("path", { class: "normal", d: path((i) => normal[Number(names[i].slice(5)) - 1]) }));
    svg.append(s("path", { class: "actual", d: path((i) => series[i]) }));
    svg.append(s("text", { x: x(war) + 6, y: T + 13 }, "since Oct 2023"));
    box.replaceChildren(svg);
    const key = el("ul", "legend");
    [["its normal for that month (2017 to 2022)", "normal"], ["what the satellite saw", "actual"]].forEach(([t, cls]) => {
      const li = el("li"), sw = document.createElementNS(NS, "svg");
      sw.setAttribute("width", "26"); sw.setAttribute("height", "10");
      const ln = document.createElementNS(NS, "line");
      ln.setAttribute("x1", "0"); ln.setAttribute("x2", "26"); ln.setAttribute("y1", "5"); ln.setAttribute("y2", "5"); ln.setAttribute("class", cls);
      sw.append(ln); sw.setAttribute("class", "chart-key"); li.append(sw, document.createTextNode(t)); key.append(li);
    });
    box.append(key);
  }

  // ------------------------------------------------------------------ what is shown for one choice
  function compute() {
    const i = state.monthIndex, uses = enabledUses();
    const n = P.n, prio = new Float64Array(n), keep = new Uint8Array(n), idx = [];
    for (let p = 0; p < n; p++) {
      if (!uses.has(P.use[p])) continue;
      keep[p] = 1;
      prio[p] = R.priority(P.prod[p], P.share[i][p] / 10000, state.penalty);
      idx.push(p);
    }
    idx.sort((a, b) => prio[b] - prio[a] || a - b);
    const rank = new Int32Array(n).fill(0);
    idx.forEach((p, k) => { rank[p] = k + 1; });
    current = DA.current = { i, keep, prio, order: idx, rank };
    return current;
  }

  function describe(i, p) {
    const s = P.share[i][p] / 10000, st = R.STATUS_NAMES[statusOf(i, p)], use = R.USE_NAMES[P.use[p]];
    const conf = R.CONFIDENCE_NAMES[confidenceOf(i, p)], green = R.GREEN_NAMES[greenOf(i, p)];
    return { share: s, status: st, use, conf, green, level: R.productivityLabel(P.prod[p]),
      reason: R.reason(use, P.prod[p], s, hectares(p), st), suggestion: R.suggestion(st, use, P.built[p], green, conf),
      reasonAr: AR.reason(use, P.prod[p], s, hectares(p), st), suggestionAr: AR.suggestion(st, use, P.built[p], green, conf) };
  }

  // ------------------------------------------------------------------ explore: map
  let photoToken = 0;
  function renderMainMap(cur) {
    const i = cur.i, month = C.months[i], canvas = $("map-main"), view = state.view === "un" && !month.has_un_map ? "priority" : state.view;
    document.querySelectorAll("[data-view]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.view === view)));
    $("view-un").disabled = !month.has_un_map;
    const un = P.un[i];
    const colorOf = view === "photo" ? null
      : view === "status" ? (p) => (cur.keep[p] ? STATUS_COLORS[statusOf(i, p)] : null)
      : view === "un" ? (p) => (cur.keep[p] && un[p] >= 0 ? STATUS_COLORS[R.statusOf(un[p] / 10000)] : null)
      : (p) => (cur.keep[p] ? blues(cur.prio[p]) : null);
    const paint = (photo) => drawMap(canvas, { cell: 6, photo, colorOf, alpha: photo ? 0.82 : 1,
      markers: view === "priority" || view === "photo" ? cur.order.slice(0, 15) : null, selected: state.selected });
    paint(null);
    const token = ++photoToken;
    loadImage(monthPhoto(month.value)).then((img) => { if (token === photoToken) paint(img); }).catch(() => {});
    const lg = $("map-legend");
    lg.replaceChildren();
    if (view === "priority") {
      const ramp = el("div", "ramp");
      ramp.append(el("span", "", "low priority"), el("div", "bar"), el("span", "", "high"));
      lg.append(ramp);
    } else if (view !== "photo") {
      const ul = el("ul", "legend"); statusLegend(ul); lg.append(ul);
    }
    canvas.setAttribute("aria-label", "Map of Khan Younis farmland plots, " + month.label + ", showing " + view);
  }

  // ------------------------------------------------------------------ explore: tables
  function fillTable(table, headers, rows, numCols, onRow) {
    table.replaceChildren();
    const head = el("thead"), hr = el("tr");
    headers.forEach((h, k) => { const c = el("th", numCols.has(k) ? "num" : "", h); c.scope = "col"; hr.appendChild(c); });
    head.appendChild(hr);
    const body = el("tbody");
    rows.forEach((r, n) => {
      const tr = el("tr");
      r.forEach((v, k) => tr.appendChild(el("td", numCols.has(k) ? "num" : "", v)));
      if (onRow) onRow(tr, n);
      body.appendChild(tr);
    });
    table.append(head, body);
  }

  function renderTables(cur) {
    const i = cur.i, names = R.STATUS_NAMES, agg = names.map(() => ({ plots: 0, ha: 0, dmg: 0 }));
    let totalHa = 0, totalDmg = 0, totalPlots = 0;
    cur.order.forEach((p) => {
      const a = agg[statusOf(i, p)], ha = hectares(p), d = (P.share[i][p] / 10000) * ha;
      a.plots++; a.ha += ha; a.dmg += d; totalPlots++; totalHa += ha; totalDmg += d;
    });
    const rows = names.map((nm, k) => [R.capitalize(nm), agg[k].plots.toLocaleString("en-US"), fmt1(agg[k].ha),
      totalHa ? R.pyFixed(100 * agg[k].ha / totalHa, 1) : "0.0", fmt1(agg[k].dmg)]);
    rows.push(["All rated plots", totalPlots.toLocaleString("en-US"), fmt1(totalHa), totalHa ? "100.0" : "0.0", fmt1(totalDmg)]);
    fillTable($("hect"), ["Status", "Plots (1 ha squares)", "Cropland (ha)", "Share of rated cropland (%)", "Estimated damaged since Oct 2023 (ha)"], rows, new Set([1, 2, 3, 4]));

    const topPlots = cur.order.slice(0, 15), arabic = state.lang === "ar";
    const top = topPlots.map((p, k) => {
      const d = describe(i, p), where = P.lat[p].toFixed(4) + ", " + P.lon[p].toFixed(4);
      return arabic
        ? [String(k + 1), "\u2066" + where + "\u2069", fmt1(hectares(p)), AR.STATUS[d.status], R.percent0(d.share), AR.USE[d.use], AR.LEVEL[d.level], AR.GREEN[d.green], d.reasonAr, d.suggestionAr]
        : [String(k + 1), where, fmt1(hectares(p)), R.capitalize(d.status), R.percent0(d.share),
          R.capitalize(d.use), R.capitalize(d.level), R.capitalize(d.green), d.reason, d.suggestion];
    });
    const table = $("top"), note = $("lang-note");
    table.dir = arabic ? "rtl" : "ltr";
    table.lang = arabic ? "ar" : "en";
    note.hidden = !arabic;
    note.replaceChildren();
    if (arabic) {
      const a = el("span", "", AR.DISCLAIMER + " " + AR.NOTE);
      a.lang = "ar"; a.dir = "rtl"; a.style.display = "block";
      note.append(a, document.createTextNode("Arabic translation written by one AI and reviewed by a second AI. It has not been checked by a professional translator."));
    }
    document.querySelectorAll("[data-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === state.lang)));
    const usesShown = [...enabledUses()].sort().map((u) => R.USE_NAMES[u]).join(", ");
    $("print-title").textContent = "The Day After: shortlist for " + C.months[i].label + ", Khan Younis governorate. Damage penalty " + state.penalty.toFixed(1) + ". Pre-war uses shown: " + usesShown + ". " + C.disclaimer;
    fillTable(table, arabic ? AR.HEADERS : ["Rank", "Location (lat, lon)", "Size (ha)", "Status", "Est. damaged", "Pre-war use", "Pre-war greenness",
      "Green now vs normal?", "Why it ranks here", "Suggested first step"], top, new Set([0, 2]), (tr, n) => {
      if (topPlots[n] === state.selected) tr.className = "sel";
      tr.addEventListener("click", () => DA.set({ selected: topPlots[n] }));
    });
  }

  // ------------------------------------------------------------------ explore: one plot in detail
  function crop(canvas, img, p) {
    const size = 60, half = size / 2, ctx = canvas.getContext("2d");
    canvas.width = canvas.height = 240;
    const sx = Math.max(0, Math.min(C.image_px.width - size, P.col[p] * 10 + 5 - half)), sy = Math.max(0, Math.min(C.image_px.height - size, P.row[p] * 10 + 5 - half));
    const k = img.naturalWidth / C.image_px.width;               // 1 for full-size pictures
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img, sx * k, sy * k, size * k, size * k, 0, 0, 240, 240);
    const s = 240 / size;
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2;
    ctx.strokeRect((P.col[p] * 10 - sx) * s, (P.row[p] * 10 - sy) * s, 10 * s, 10 * s);
  }

  function renderDetail(cur) {
    const box = $("detail"), p = state.selected, i = cur.i, month = C.months[i];
    box.replaceChildren();
    if (p < 0 || !cur.keep[p]) {
      box.append(el("h3", "", "Pick a plot"), el("p", "small", "Click any square on the map, or a row in the table below, to see that plot up close: the satellite picture before and now, and how its priority is worked out."));
      return;
    }
    const d = describe(i, p);
    box.append(el("h3", "", "Plot at " + P.lat[p].toFixed(4) + ", " + P.lon[p].toFixed(4)));
    const crops = el("div", "crops");
    [[beforePhoto(month.value), month.label.slice(0, 3) + " 2022"], [monthPhoto(month.value), month.label]].forEach(([url, cap]) => {
      const fig = el("figure"), cv = el("canvas");
      cv.width = cv.height = 240;
      cv.setAttribute("role", "img"); cv.setAttribute("aria-label", "Satellite close-up of the plot, " + cap);
      fig.append(cv, el("figcaption", "", cap));
      crops.append(fig);
      loadImage(url).then((img) => crop(cv, img, p)).catch(() => {});
    });
    box.append(crops, el("p", "small", "600 m across. The white square is the plot."));
    const dl = el("dl");
    const add = (k, v) => { dl.append(el("dt", "", k), el("dd", "", v)); };
    add("AI estimate", R.capitalize(d.status) + ", about " + R.percent0(d.share) + " damaged" + (d.conf === "lower" ? " (uncertain)" : ""));
    if (P.un[i][p] >= 0) add("UN map", R.percent0(P.un[i][p] / 10000) + " damaged");
    add("Before the war", R.capitalize(d.use) + ", " + d.level + " greenness");
    add("Green now?", R.capitalize(d.green) + " for this month");
    add("Size", fmt1(hectares(p)) + " ha of farmland");
    add("Rank", cur.rank[p].toLocaleString("en-US") + " of " + cur.order.length.toLocaleString("en-US"));
    box.append(dl);
    const f = el("p", "formula");
    f.append(document.createTextNode(P.prod[p].toFixed(2) + " × (1 − " + state.penalty.toFixed(1) + " × " + d.share.toFixed(2) + ") = "), el("span", "res", cur.prio[p].toFixed(2)));
    const step = el("p", "", state.lang === "ar" ? d.suggestionAr : d.suggestion);
    if (state.lang === "ar") { step.lang = "ar"; step.dir = "rtl"; }
    box.append(el("p", "small", "Priority = productivity × (1 − penalty × damaged share):"), f, step);
  }

  // ------------------------------------------------------------------ explore: banner, link, download
  function renderBanner(cur) {
    const month = C.months[cur.i], b = $("banner"), F = C.facts;
    b.replaceChildren();
    const p = el("p");
    if (month.has_un_map) {
      p.append(el("strong", "", month.label + ": a UN map exists for this month. "), document.createTextNode("Every plot's estimate comes from a model that had not seen that land, so you can compare it with the UN map."));
    } else {
      p.append(el("strong", "", month.label + ": no UN map exists for this month, so nothing here can be checked. "),
        document.createTextNode("This is the model's reading of the satellite pictures only. In the one test like this (learn up to Dec 2024, judge Jul 2025 on unseen land) it scored " + F.f1_v2_fwd + " out of 1 and was not clearly better than a simple rule. A falling damage estimate means land looks greener, not that it was repaired."));
    }
    b.append(p);
    b.classList.toggle("warn", !month.has_un_map);
    if (!cur.order.length) b.append(el("p", "", "Tick at least one pre-war use above to see plots."));
  }

  function syncUrl() {
    const uses = [...enabledUses()].sort().join(",");
    let q = "?month=" + C.months[state.monthIndex].value + "&penalty=" + state.penalty.toFixed(1) + "&uses=" + uses;
    if (state.selected >= 0) q += "&plot=" + P.plot_id[state.selected];
    if (state.lang === "ar") q += "&lang=ar";
    if (DA.query.has("autorun")) q += "&autorun=1";
    history.replaceState(null, "", location.pathname + q + location.hash);
  }

  function downloadCsv() {
    const cur = current, i = cur.i, month = C.months[i].value;
    const q = (s) => '"' + String(s).replace(/"/g, '""') + '"';
    const head = ["rank", "lat", "lon", "cropland_ha", "status", "estimated_damaged_share", "priority", "pre_war_use",
      "pre_war_greenness_percentile", "green_now_vs_normal", "confidence", "reason", "suggested_first_step", "note"];
    const lines = [head.join(",")];
    const arabic = state.lang === "ar";
    if (arabic) { head.push("reason_ar", "suggested_first_step_ar", "note_ar"); lines[0] = head.join(","); }
    cur.order.forEach((p, k) => {
      const d = describe(i, p);
      const row = [k + 1, P.lat[p].toFixed(5), P.lon[p].toFixed(5), hectares(p).toFixed(2), d.status, d.share.toFixed(3), cur.prio[p].toFixed(4),
        d.use, P.prod[p].toFixed(3), d.green, d.conf, q(d.reason), q(d.suggestion), q(C.disclaimer)];
      if (arabic) row.push(q(d.reasonAr), q(d.suggestionAr), q(AR.DISCLAIMER + " " + AR.NOTE));
      lines.push(row.join(","));
    });
    const a = el("a");
    // the first character (a byte-order mark) tells spreadsheet programs the file is UTF-8, so Arabic shows correctly
    a.href = URL.createObjectURL(new Blob(["\ufeff" + lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" }));
    a.download = "day_after_" + month + "_ranked_plots" + (arabic ? "_ar" : "") + ".csv";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  // ------------------------------------------------------------------ drawing everything again after a change
  function render() {
    const cur = compute();
    $("month").value = String(state.monthIndex);
    $("penalty").value = String(state.penalty);
    $("penalty-out").textContent = state.penalty.toFixed(1);
    document.querySelectorAll("[data-preset]").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.preset) === state.penalty)));
    syncUrl();
    renderBanner(cur);
    renderMainMap(cur);
    renderTables(cur);
    renderDetail(cur);
    DA.listeners.forEach((fn) => fn(cur));
  }
  DA.set = (changes) => { Object.assign(state, changes); render(); };

  function plotAt(canvas, ev) {
    const r = canvas.getBoundingClientRect();
    const col = Math.floor(((ev.clientX - r.left) / r.width) * C.grid.cols), row = Math.floor(((ev.clientY - r.top) / r.height) * C.grid.rows);
    return col >= 0 && row >= 0 && col < C.grid.cols && row < C.grid.rows ? cellOf[row * C.grid.cols + col] : -1;
  }

  function attachMap(canvas) {
    const tip = $("tip");
    canvas.addEventListener("mousemove", (ev) => {
      const p = current ? plotAt(canvas, ev) : -1;
      if (p < 0 || !current.keep[p]) { tip.hidden = true; return; }
      const d = describe(current.i, p), un = P.un[current.i][p];
      tip.replaceChildren();
      const lines = [fmt1(hectares(p)) + " ha, " + d.use, R.capitalize(d.status) + ", about " + R.percent0(d.share) + " damaged", "Rank " + current.rank[p] + " of " + current.order.length];
      if (un >= 0) lines.push("UN map: " + R.percent0(un / 10000) + " damaged");
      lines.forEach((t, k) => { const s = el("div", "", t); if (k === 0) s.style.fontWeight = "700"; tip.appendChild(s); });
      tip.hidden = false;
      tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8) + "px";
      tip.style.top = ev.clientY + 14 + "px";
    });
    canvas.addEventListener("mouseleave", () => { tip.hidden = true; });
    canvas.addEventListener("click", (ev) => { const p = plotAt(canvas, ev); if (p >= 0 && current.keep[p]) DA.set({ selected: p }); });
  }

  // ------------------------------------------------------------------ start-up
  async function start() {
    DA.query = new URLSearchParams(location.search);
    try {
      [P, C] = await Promise.all([fetch("data/plots.json").then((r) => r.json()), fetch("data/content.json").then((r) => r.json())]);
    } catch (e) {
      $("error").hidden = false;
      $("error").textContent = "The data files could not be loaded. If you opened this file from a folder, use the public link, or serve the folder with a web server (python -m http.server).";
      return;
    }
    Object.assign(DA, { P, C, blues, statusOf, hectares, fmt1, el, loadImage, monthPhoto, beforePhoto, drawMap, statusLegend, rhythmChart, describe });
    DA.sample = () => (DA._sample = DA._sample || fetch("data/sample.json").then((r) => r.json()));
    document.querySelectorAll("[data-fact]").forEach((e) => { e.textContent = C.facts[e.dataset.fact] ?? ""; });     // numbers come from the results files
    document.querySelectorAll('[data-legend="status"]').forEach(statusLegend);
    ["credit-1", "credit-2", "credit-3"].forEach((id) => { if ($(id)) $(id).textContent = C.image_credit; });
    cellOf = new Int32Array(C.grid.rows * C.grid.cols).fill(-1);
    for (let p = 0; p < P.n; p++) cellOf[P.row[p] * C.grid.cols + P.col[p]] = p;

    const sel = $("month"), q = DA.query;
    C.months.forEach((m, k) => {
      const o = el("option", "", m.label + (m.has_un_map ? "  (UN map available)" : "  (no UN map: model estimate only)"));
      o.value = String(k);
      sel.appendChild(o);
    });
    state.monthIndex = C.months.findIndex((m) => m.value === "2025-07");
    // shareable links: ?month=2026-04&penalty=0.4&uses=0,2&plot=115101   (uses: 0 seasonal crops, 1 mixed, 2 orchard)
    const k = C.months.findIndex((m) => m.value === q.get("month"));
    if (k >= 0) state.monthIndex = k;
    if (q.has("penalty") && isFinite(Number(q.get("penalty")))) state.penalty = Math.min(1, Math.max(0, Math.round(Number(q.get("penalty")) * 10) / 10));
    if (q.has("uses")) {
      const want = new Set(q.get("uses").split(",").filter((x) => x !== "").map(Number));
      document.querySelectorAll('input[name="use"]').forEach((b) => { b.checked = want.has(Number(b.value)); });
    }
    if (q.has("plot")) state.selected = P.plot_id.indexOf(Number(q.get("plot")));
    if (q.get("lang") === "ar") state.lang = "ar";

    sel.addEventListener("change", () => DA.set({ monthIndex: Number(sel.value) }));
    $("penalty").addEventListener("input", (e) => DA.set({ penalty: Number(e.target.value) }));
    document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => DA.set({ penalty: Number(b.dataset.preset) })));
    document.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => DA.set({ view: b.dataset.view })));
    document.querySelectorAll('input[name="use"]').forEach((b) => b.addEventListener("change", render));
    $("download").addEventListener("click", downloadCsv);
    $("print").addEventListener("click", () => window.print());
    document.querySelectorAll("[data-lang]").forEach((b) => b.addEventListener("click", () => DA.set({ lang: b.dataset.lang })));
    const themeButton = $("theme-toggle"), root = document.documentElement;
    const nameTheme = () => { themeButton.textContent = root.dataset.theme === "dark" ? "Switch to light colours" : "Switch to dark colours"; };
    themeButton.addEventListener("click", () => {
      if (root.dataset.theme === "dark") delete root.dataset.theme; else root.dataset.theme = "dark";
      try { localStorage.setItem("theme", root.dataset.theme || "light"); } catch (e) { /* private window: the choice just is not remembered */ }
      nameTheme();
    });
    nameTheme();
    attachMap($("map-main"));
    DA.ready.forEach((fn) => fn());                              // story.js and proof.js set themselves up
    render();
  }
  start();
})();
