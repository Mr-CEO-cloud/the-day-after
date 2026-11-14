/*
 * model.js - runs the trained model (v2) in the browser.
 *
 * Plain words: the model is 400 decision trees. Each tree asks a chain of yes/no questions about a
 * pixel ("is its worst greenness drop since Oct 2023 below -0.21?") and ends in a number. The 400
 * numbers are added up and squeezed into a probability between 0 and 1 that the pixel was damaged.
 * This file does exactly that, using the trees exported from the trained model (data/model.json).
 * tests/test_web.py checks that it gives the same probability as Python for 6,000 test pixels.
 *
 * This project is a prototype, not agricultural or humanitarian advice.
 */
(function (root) {
  "use strict";

  function create(json) {
    const trees = json.trees.map((t) => ({
      f: Int32Array.from(t.f), t: Float64Array.from(t.t), l: Int32Array.from(t.l), r: Int32Array.from(t.r),
      d: Uint8Array.from(t.d), v: Float64Array.from(t.v),
    }));
    const calX = Float64Array.from(json.calibration.x), calY = Float64Array.from(json.calibration.y);

    // Sum of the 400 leaf values for one pixel. x holds all pixels' features in one long array;
    // the pixel starts at `offset`.
    function rawScore(x, offset) {
      let sum = 0;
      for (let k = 0; k < trees.length; k++) {
        const T = trees[k];
        if (T.f.length === 0) { sum += T.v[0]; continue; }
        let node = 0;
        while (node >= 0) {
          let value = x[offset + T.f[node]];
          const flag = T.d[node], missing = value !== value;         // NaN is the only number not equal to itself
          if (missing && (flag & 2)) {                                 // this split treats "no value" as its own case
            node = (flag & 1) ? T.l[node] : T.r[node];
            continue;
          }
          if (missing) value = 0;                                      // otherwise LightGBM reads a missing value as 0
          node = value <= T.t[node] ? T.l[node] : T.r[node];
        }
        sum += T.v[-node - 1];
      }
      return sum;
    }

    const probability = (x, offset) => 1 / (1 + Math.exp(-rawScore(x, offset)));

    // Calibration: a lookup curve learned on training land that turns the model's raw probability into
    // "what share of pixels with this score did the UN really mark damaged". Straight lines between points.
    function calibrate(p) {
      const n = calX.length;
      if (p <= calX[0]) return calY[0];
      if (p >= calX[n - 1]) return calY[n - 1];
      let lo = 0, hi = n - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (calX[mid] <= p) lo = mid; else hi = mid; }
      return calY[lo] + (calY[hi] - calY[lo]) * (p - calX[lo]) / (calX[hi] - calX[lo]);
    }

    // The simple rival: "damaged if the worst monthly greenness drop since Oct 2023 is below a threshold".
    const ruleIndex = json.features.indexOf(json.rule.feature);
    function rule(x, offset, month) {
      let v = x[offset + ruleIndex];
      if (v !== v) v = 0;                                              // no clear view counts as "normal"
      return v < json.rule.thresholds[month] ? 1 : 0;
    }

    return { features: json.features, threshold: json.threshold, nTrees: trees.length,
      nSplits: trees.reduce((a, t) => a + t.f.length, 0), rawScore, probability, calibrate, rule };
  }

  // Macro F1 from the four confusion-matrix counts (tn, fp, fn, tp).
  function macroF1(tn, fp, fn, tp) {
    const f = (a, b, c) => (2 * a + b + c > 0 ? (2 * a) / (2 * a + b + c) : 0);
    return (f(tp, fp, fn) + f(tn, fn, fp)) / 2;
  }

  const api = { create, macroF1 };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Model = api;
})(typeof self !== "undefined" ? self : this);
