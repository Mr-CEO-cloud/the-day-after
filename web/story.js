/*
 * story.js - the guided part of the page: the satellite time-lapse at the top, the before/after
 * slider, one field's greenness chart, the UN maps month by month, and the worked priority example.
 *
 * Plain words: everything drawn here is real data. The pictures are Sentinel-2 satellite images, the
 * UN maps are the UNOSAT-FAO assessments, and the numbers come from data/content.json.
 *
 * This project is a prototype, not agricultural or humanitarian advice.
 */
(function () {
  "use strict";
  const DA = window.DA, $ = DA.$;
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  DA.ready.push(function () {
    const { C, P } = DA, R = window.Rules;

    // ------------------------------------------------------------------ time-lapse at the top
    (function lapse() {
      const canvas = $("lapse"), range = $("lapse-range"), label = $("lapse-label"), play = $("lapse-play");
      const frames = C.lapse, imgs = new Array(frames.length).fill(null);
      const W = 530, H = Math.round((W * C.image_px.height) / C.image_px.width), dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      range.max = String(frames.length - 1);
      let at = frames.length - 1, timer = null, loaded = 0;

      function draw() {
        let k = at;
        while (k >= 0 && !imgs[k]) k--;                           // show the nearest picture already downloaded
        if (k < 0) { for (k = at; k < frames.length && !imgs[k]; k++); }
        if (k >= 0 && k < frames.length && imgs[k]) ctx.drawImage(imgs[k], 0, 0, W, H);
        const sx = W / (C.image_px.width / 10), sy = H / (C.image_px.height / 10);
        ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.6;
        C.boundary.forEach((ring) => {
          ctx.beginPath();
          ring.x.forEach((x, n) => { if (n === 0) ctx.moveTo(x * sx, ring.y[n] * sy); else ctx.lineTo(x * sx, ring.y[n] * sy); });
          ctx.stroke();
        });
        label.textContent = frames[at].label;
        range.value = String(at);
        range.setAttribute("aria-valuetext", frames[at].label);
      }
      function stop() { clearInterval(timer); timer = null; play.textContent = "Play"; play.setAttribute("aria-pressed", "false"); }
      function start() {
        if (at >= frames.length - 1) at = 0;
        play.textContent = "Pause"; play.setAttribute("aria-pressed", "true");
        timer = setInterval(() => { if (at >= frames.length - 1) { stop(); return; } at++; draw(); }, 230);
      }
      play.addEventListener("click", () => (timer ? stop() : start()));
      range.addEventListener("input", () => { stop(); at = Number(range.value); draw(); });

      // the last picture first (so something real is on screen at once), then the rest in order
      const order = [frames.length - 1].concat(frames.map((_, k) => k).slice(0, -1));
      let next = 0;
      (function loadNext() {
        if (next >= order.length) return;
        const k = order[next++];
        DA.loadImage("img/lapse/" + frames[k].month + ".webp").then((img) => {
          imgs[k] = img; loaded++;
          if (k === at || loaded === 1) draw();
          if (loaded === frames.length && !reduceMotion && !DA.query.has("autorun") && !location.hash) { at = 0; draw(); start(); }   // play once
          loadNext();
        }).catch(loadNext);
      })();
    })();

    // ------------------------------------------------------------------ step 2: before / after
    (function beforeAfter() {
      const before = $("ba-before"), after = $("ba-after"), range = $("ba-range");
      const clip = () => { before.style.clipPath = "inset(0 " + (100 - Number(range.value)) + "% 0 0)"; };
      function show(m) {
        const lab = C.months.find((x) => x.value === m).label;
        before.src = DA.beforePhoto(m); after.src = DA.monthPhoto(m);
        before.alt = "Satellite picture of Khan Younis, " + lab.slice(0, 3) + " 2022";
        after.alt = "Satellite picture of Khan Younis, " + lab;
        $("ba-tag-before").textContent = lab.slice(0, 3) + " 2022";
        $("ba-tag-after").textContent = lab;
        document.querySelectorAll("[data-ba]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.ba === m)));
        clip();
      }
      range.addEventListener("input", clip);
      document.querySelectorAll("[data-ba]").forEach((b) => b.addEventListener("click", () => show(b.dataset.ba)));
      // only fetch the two large pictures when this step is about to be seen
      const io = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) { io.disconnect(); show("2025-07"); } }, { rootMargin: "600px" });
      io.observe($("ba"));
    })();

    // ------------------------------------------------------------------ step 2: one field's rhythm
    (function rhythm() {
      const box = $("rhythm");
      let list = null, at = 0;
      function show(S) {
        if (!list) {
          // fields with a clear yearly rhythm that the UN marked damaged: the pattern is easiest to see on these
          list = S.history.pixels.map((pix, k) => ({ k, pix })).filter(({ k, pix }) => {
            const n = S.history.normal_x1000[k].filter((v) => v !== null);
            return S.label[pix] === 1 && n.length === 12 && Math.max(...n) - Math.min(...n) > 250;
          }).map((o) => o.k);
          if (!list.length) list = S.history.pixels.map((_, k) => k);
        }
        const k = list[at % list.length], pix = S.history.pixels[k];
        DA.rhythmChart(box, S, k, "Greenness of one patch of farmland each month, against its own normal");
        box.append(DA.el("p", "small", "A patch at " + S.lat[pix].toFixed(4) + ", " + S.lon[pix].toFixed(4) + ". The UN marked it damaged by " + monthLabel(S.months[S.month[pix]]) + "."));
      }
      const monthLabel = (m) => C.months.find((x) => x.value === m).label;
      const io = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) { io.disconnect(); DA.sample().then(show); } }, { rootMargin: "600px" });
      io.observe(box);
      $("rhythm-next").addEventListener("click", () => { at++; DA.sample().then(show); });
    })();

    // ------------------------------------------------------------------ step 3: the UN maps, month by month
    (function unStory() {
      const canvas = $("map-unstory"), range = $("un-range"), label = $("un-label"), series = C.un_series;
      const unMonths = C.months.map((m, i) => ({ m, i })).filter((o) => o.m.has_un_map);
      range.max = String(unMonths.length - 1);
      range.value = String(unMonths.length - 1);
      let token = 0, seen = false;

      function chart(at) {
        const W = 560, H = 190, L = 40, Rr = 14, T = 12, B = 28, NS = "http://www.w3.org/2000/svg";
        const x = (k) => L + (k / (series.length - 1)) * (W - L - Rr), y = (v) => T + (1 - v) * (H - T - B);
        const s = (tag, attrs, text) => { const e = document.createElementNS(NS, tag); for (const a in attrs) e.setAttribute(a, attrs[a]); if (text) e.textContent = text; return e; };
        const svg = s("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Share of Khan Younis farmland marked damaged in each UN map, with the model's estimate" });
        [0, 0.5, 1].forEach((v) => svg.append(s("line", { class: "axis", x1: L, x2: W - Rr, y1: y(v), y2: y(v) }), s("text", { x: L - 6, y: y(v) + 4, "text-anchor": "end" }, Math.round(v * 100) + "%")));
        series.forEach((d, k) => { if (k === 0 || k === series.length - 1 || d.month.slice(5) === "01") svg.append(s("text", { x: x(k), y: H - 8, "text-anchor": k === 0 ? "start" : k === series.length - 1 ? "end" : "middle" }, d.label)); });
        const line = (key) => series.map((d, k) => (k ? "L" : "M") + x(k).toFixed(1) + " " + y(d[key]).toFixed(1)).join("");
        svg.append(s("path", { class: "ai", d: line("model_share") }), s("path", { class: "un", d: line("un_share") }));
        svg.append(s("circle", { class: "dot", cx: x(at), cy: y(series[at].un_share), r: 5 }));
        $("un-chart").replaceChildren(svg);
        const key = DA.el("ul", "legend");
        [["UN experts", "un"], ["our model's estimate", "ai"]].forEach(([t, cls]) => {
          const li = DA.el("li"), sw = document.createElementNS(NS, "svg"), ln = document.createElementNS(NS, "line");
          sw.setAttribute("width", "26"); sw.setAttribute("height", "10"); sw.setAttribute("class", "chart-key");
          ln.setAttribute("x1", "0"); ln.setAttribute("x2", "26"); ln.setAttribute("y1", "5"); ln.setAttribute("y2", "5"); ln.setAttribute("class", cls);
          sw.append(ln); li.append(sw, document.createTextNode(t)); key.append(li);
        });
        $("un-chart").append(key);
      }

      function show() {
        const at = Number(range.value), { m, i } = unMonths[at], un = P.un[i];
        const colorOf = (p) => (un[p] >= 0 ? DA.STATUS_COLORS[R.statusOf(un[p] / 10000)] : null);
        label.textContent = m.label + ": " + Math.round(series[at].un_share * 100) + "% of farmland marked damaged";
        range.setAttribute("aria-valuetext", m.label);
        DA.drawMap(canvas, { cell: 4, colorOf });
        chart(at);
        const mine = ++token;
        DA.loadImage(DA.monthPhoto(m.value)).then((img) => { if (mine === token) DA.drawMap(canvas, { cell: 4, photo: img, colorOf, alpha: 0.72 }); }).catch(() => {});
      }
      range.addEventListener("input", show);
      const io = new IntersectionObserver((entries) => { if (!seen && entries.some((e) => e.isIntersecting)) { seen = true; io.disconnect(); show(); } }, { rootMargin: "600px" });
      io.observe(canvas);
    })();

    // ------------------------------------------------------------------ step 5: the formula with a real plot's numbers
    (function worked() {
      const slider = $("penalty-story");
      slider.addEventListener("input", () => DA.set({ penalty: Number(slider.value) }));
      DA.listeners.push((cur) => {
        slider.value = String(DA.state.penalty);
        $("penalty-story-out").textContent = DA.state.penalty.toFixed(1);
        $("worked-month").textContent = C.months[cur.i].label;
        const box = $("worked"), text = $("worked-text");
        box.replaceChildren();
        if (!cur.order.length) { text.textContent = "Tick at least one pre-war use in the map section to see a plot."; return; }
        const p = cur.order[0], d = DA.describe(cur.i, p);
        box.append(document.createTextNode(P.prod[p].toFixed(2) + " × (1 − " + DA.state.penalty.toFixed(1) + " × " + d.share.toFixed(2) + ") = "), DA.el("span", "res", cur.prio[p].toFixed(2)));
        text.textContent = d.reason + " Move the slider: with a higher penalty, less damaged plots move up the list.";
      });
    })();
  });
})();
