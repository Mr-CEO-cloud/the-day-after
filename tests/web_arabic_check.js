// web_arabic_check.js - run by tests/test_web.py. Checks that the Arabic shortlist (web/rules_ar.js)
// has a sentence for every case the planner can produce, and that the numbers in each Arabic sentence
// are the same as in the English one, for every plot in every month.
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "..");
const R = require(path.join(root, "web", "rules.js"));
const A = require(path.join(root, "web", "rules_ar.js"));
const P = JSON.parse(fs.readFileSync(path.join(root, "web", "data", "plots.json"), "utf8"));
const arabic = /[؀-ۿ]/, latinWord = /[A-Za-z]{2,}/;
const numbers = (s) => (s.match(/\d+(\.\d+)?/g) || []).join(" ");
let texts = 0, problems = 0, firstProblem = null;
const note = (why, s) => { problems++; if (!firstProblem) firstProblem = why + ": " + s; };
P.months.forEach((m, i) => {
  for (let p = 0; p < P.n; p++) {
    const code = P.code[i][p], status = R.STATUS_NAMES[code % 3], conf = R.CONFIDENCE_NAMES[Math.floor(code / 3) % 2], green = R.GREEN_NAMES[Math.floor(code / 6)];
    const use = R.USE_NAMES[P.use[p]], share = P.share[i][p] / 10000, ha = P.px[p] * 0.01;
    const en = R.reason(use, P.prod[p], share, ha, status), ar = A.reason(use, P.prod[p], share, ha, status);
    const sen = R.suggestion(status, use, P.built[p], green, conf), sar = A.suggestion(status, use, P.built[p], green, conf);
    texts += 2;
    for (const s of [ar, sar]) { if (!s || /undefined|NaN/.test(s) || !arabic.test(s) || latinWord.test(s)) note("not a complete Arabic sentence", s); }
    if (numbers(en) !== numbers(ar)) note("numbers differ", en + " | " + ar);
    // the same optional warnings must be present in both languages
    const parts = (s) => s.split(/[.؟]\s*/).filter(Boolean).length;
    if (parts(sen) !== parts(sar)) note("different number of sentences", sen + " | " + sar);
  }
});
const complete = Object.keys(R.USE_RULES).every((k) => A.USE_RULES[k]) && R.STATUS_NAMES.every((s) => A.STATUS[s]) && R.USE_NAMES.every((u) => A.USE[u]) &&
  R.GREEN_NAMES.every((g) => A.GREEN[g]) && ["low", "medium", "high"].every((l) => A.LEVEL[l]) && A.HEADERS.length === 10;
console.log(JSON.stringify({ texts, problems, firstProblem, complete, disclaimer: A.DISCLAIMER }));
