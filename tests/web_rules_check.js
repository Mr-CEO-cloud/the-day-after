// web_rules_check.js - run by tests/test_web.py. Uses the page's rules.js on the page's data and prints a
// fingerprint of every one-line reason and suggested step (all plots, all 21 months).
const fs = require("fs");
const crypto = require("crypto");
const path = require("path");
const root = path.resolve(__dirname, "..");
const R = require(path.join(root, "web", "rules.js"));
const P = JSON.parse(fs.readFileSync(path.join(root, "web", "data", "plots.json"), "utf8"));
const hash = crypto.createHash("sha256");
let count = 0;
P.months.forEach((m, i) => {
  for (let p = 0; p < P.n; p++) {
    const code = P.code[i][p], status = R.STATUS_NAMES[code % 3], conf = R.CONFIDENCE_NAMES[Math.floor(code / 3) % 2], green = R.GREEN_NAMES[Math.floor(code / 6)];
    const use = R.USE_NAMES[P.use[p]];
    hash.update(R.reason(use, P.prod[p], P.share[i][p] / 10000, P.px[p] * 0.01, status) + "\n" + R.suggestion(status, use, P.built[p], green, conf) + "\n");
    count += 2;
  }
});
console.log(JSON.stringify({ sha256: hash.digest("hex"), texts: count }));
