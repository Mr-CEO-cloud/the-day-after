/*
 * rules.js - the Day After planner's rules, in JavaScript.
 *
 * Plain words: these are the same rules as app/planner_rules.py (status cut-offs, the one-line
 * reason, the suggested first step, the priority formula). The web page uses them to redraw the
 * maps and tables when the visitor moves the slider. tests/test_web.py runs this file in Node and
 * checks that it prints exactly the same sentences as the Python version for every plot in every month.
 *
 * This project is a prototype, not agricultural or humanitarian advice.
 */
(function (root) {
  "use strict";

  const STATUS_NAMES = ["intact", "partly damaged", "mostly damaged"];
  const STATUS_CUTS = [0.25, 0.75];       // damaged share: < 25% intact, 25-75% partly, >= 75% mostly
  const PRODUCTIVITY_LEVELS = [1 / 3, 2 / 3];
  const BUILT_SHARE_FLAG = 0.5;
  const DEFAULT_PENALTY = 0.6;            // priority = productivity x (1 - penalty x damaged share)
  const USE_NAMES = ["seasonal crops", "mixed trees and crops", "orchard"];
  const CONFIDENCE_NAMES = ["higher", "lower"];
  const GREEN_NAMES = ["below normal", "at or above normal"];

  const USE_RULES = {
    "intact|orchard": "Keep the trees producing: restore water and tree care first.",
    "intact|mixed trees and crops": "Restore water; keep the trees and plant the open parts next season.",
    "intact|seasonal crops": "Ready for the next planting season: seeds, fertiliser and water.",
    "partly damaged|orchard": "Survey which trees survived; repair irrigation and replant gaps.",
    "partly damaged|mixed trees and crops": "Plant the undamaged parts first; survey trees and clear the rest.",
    "partly damaged|seasonal crops": "Plant the undamaged parts first; clear and check the rest.",
    "mostly damaged|orchard": "Longer term: clear and check the land, then replant trees.",
    "mostly damaged|mixed trees and crops": "Longer term: clear and check soil and safety before planting.",
    "mostly damaged|seasonal crops": "Clear and check soil and safety first, then plant seasonal crops.",
  };

  /* Python-style rounding. Python formats the EXACT binary value of a number and breaks exact ties
     towards the even digit; JavaScript's toFixed breaks them upwards. toFixed(100) gives the exact
     digits, so I can round them the way Python does. */
  function pyFixed(x, d) {
    const neg = x < 0;
    const s = Math.abs(x).toFixed(100);
    const dot = s.indexOf(".");
    const whole = s.slice(0, dot);
    const frac = s.slice(dot + 1);
    let kept = whole + frac.slice(0, d);                       // digits I keep, as one integer string
    const next = frac.charCodeAt(d) - 48;                      // first dropped digit
    const rest = frac.slice(d + 1);
    const restZero = /^0*$/.test(rest);
    const lastKept = kept.charCodeAt(kept.length - 1) - 48;
    let up = next > 5 || (next === 5 && (!restZero || lastKept % 2 === 1));
    if (up) {                                                  // add one to the kept integer string
      const digits = kept.split("").map(Number);
      let i = digits.length - 1;
      while (i >= 0) {
        if (digits[i] === 9) { digits[i] = 0; i--; } else { digits[i]++; break; }
      }
      if (i < 0) digits.unshift(1);
      kept = digits.join("");
    }
    const intLen = kept.length - d;
    const out = d > 0 ? kept.slice(0, intLen) + "." + kept.slice(intLen) : kept;
    return (neg && /[1-9]/.test(out) ? "-" : "") + out;
  }
  const pyRound = (x) => parseInt(pyFixed(x, 0), 10);           // Python's round(): halves go to the even number
  const fixed1 = (x) => pyFixed(x, 1);                          // Python's f"{x:.1f}"
  const percent0 = (x) => pyFixed(x * 100, 0) + "%";            // Python's f"{x:.0%}"

  const statusOf = (share) => (share < STATUS_CUTS[0] ? 0 : share < STATUS_CUTS[1] ? 1 : 2);   // like numpy.digitize
  const productivityLabel = (pct) => ["low", "medium", "high"][pct < PRODUCTIVITY_LEVELS[0] ? 0 : pct < PRODUCTIVITY_LEVELS[1] ? 1 : 2];
  const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  function reason(use, productivityPct, damagedShare, croplandHa, status) {
    const level = productivityLabel(productivityPct);
    const top = Math.max(1, pyRound(100 * (1 - productivityPct)));
    const prod = level === "high" ? "top " + top + "% for pre-war greenness" : level + " pre-war greenness";
    return capitalize(use) + ", " + prod + "; the model estimates " + percent0(damagedShare) + " of its " +
      fixed1(croplandHa) + " ha damaged since Oct 2023 (" + status + ").";
  }

  function suggestion(status, use, builtShare, greenNow, confidence) {
    let text = USE_RULES[status + "|" + use];
    if (builtShare >= BUILT_SHARE_FLAG) text += " Pre-war map shows built-up cover here (possibly greenhouses): check structures.";
    if (status !== "intact" && greenNow === "at or above normal")
      text += " Damaged, but as green as normal this month: could be replanted crops, weeds or regrowth.";
    text += confidence === "lower" ? " Uncertain estimate: verify on the ground first." : " Verify on the ground.";
    return text;
  }

  const priority = (productivityPct, damagedShare, penalty) => productivityPct * (1 - penalty * damagedShare);

  const api = { STATUS_NAMES, STATUS_CUTS, DEFAULT_PENALTY, USE_NAMES, CONFIDENCE_NAMES, GREEN_NAMES, USE_RULES,
    pyFixed, pyRound, fixed1, percent0, statusOf, productivityLabel, capitalize, reason, suggestion, priority };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.Rules = api;
})(typeof self !== "undefined" ? self : this);
