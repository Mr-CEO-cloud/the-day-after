// web_model_check.js - run by tests/test_web.py. Runs the page's model.js on the page's sample of
// test pixels and prints what it found, so Python can compare it with its own numbers.
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "..");
const M = require(path.join(root, "web", "model.js"));
const json = JSON.parse(fs.readFileSync(path.join(root, "web", "data", "model.json"), "utf8"));
const S = JSON.parse(fs.readFileSync(path.join(root, "web", "data", "sample.json"), "utf8"));
const buf = fs.readFileSync(path.join(root, "web", "data", "sample.bin"));
const X = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
const model = M.create(json);
let maxProb = 0, maxCal = 0, decisionDiff = 0, ruleDiff = 0;
const cm = S.months.map(() => ({ model: [0, 0, 0, 0], rule: [0, 0, 0, 0] }));     // tn, fp, fn, tp
for (let i = 0; i < S.n; i++) {
  const off = i * S.n_features, p = model.probability(X, off), c = model.calibrate(p);
  maxProb = Math.max(maxProb, Math.abs(p - S.py_prob[i]));
  maxCal = Math.max(maxCal, Math.abs(c - S.py_cal[i]));
  const pred = p >= model.threshold ? 1 : 0, py = S.py_prob[i] >= model.threshold ? 1 : 0;
  if (pred !== py) decisionDiff++;
  const r = model.rule(X, off, S.months[S.month[i]]);
  if (r !== S.py_rule[i]) ruleDiff++;
  cm[S.month[i]].model[S.label[i] * 2 + pred]++;
  cm[S.month[i]].rule[S.label[i] * 2 + r]++;
}
console.log(JSON.stringify({ n: S.n, trees: model.nTrees, splits: model.nSplits, max_prob_diff: maxProb, max_cal_diff: maxCal,
  decision_diff: decisionDiff, rule_diff: ruleDiff, months: S.months,
  confusion: cm.map((c) => ({ model: c.model, rule: c.rule, model_f1: M.macroF1(...c.model), rule_f1: M.macroF1(...c.rule) })) }));
