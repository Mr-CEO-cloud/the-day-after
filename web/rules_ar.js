/*
 * rules_ar.js - the shortlist in Arabic.
 *
 * Plain words: the people who would use this tool read Arabic. This file holds an Arabic version of
 * every sentence the planner can produce (status, pre-war use, the one-line reason, the suggested
 * first step and its warnings). The numbers and the logic are exactly those of rules.js; only the
 * words change. tests/test_web.py checks that every combination has an Arabic sentence and that
 * the numbers in the Arabic and English sentences are the same.
 *
 * The translation was written by an AI (Claude) and reviewed by a second AI (OpenAI Codex). It has
 * not been checked by a professional translator, and the page says so.
 *
 * This project is a prototype, not agricultural or humanitarian advice.
 */
(function (root) {
  "use strict";
  const R = typeof module !== "undefined" && module.exports ? require("./rules.js") : root.Rules;

  const STATUS = { "intact": "سليمة", "partly damaged": "متضررة جزئياً", "mostly damaged": "متضررة في معظمها" };
  const USE = { "seasonal crops": "محاصيل موسمية", "mixed trees and crops": "أشجار ومحاصيل مختلطة", "orchard": "بستان" };
  const LEVEL = { "low": "منخفض", "medium": "متوسط", "high": "مرتفع" };
  const GREEN = { "below normal": "أقل من المعتاد", "at or above normal": "بالمستوى المعتاد أو أعلى" };
  const HEADERS = ["الترتيب", "الموقع (خط العرض، خط الطول)", "المساحة (هكتار)", "الحالة", "نسبة الضرر المقدّرة", "الاستخدام قبل الحرب",
    "الاخضرار قبل الحرب", "الاخضرار الآن مقارنة بالمعتاد", "سبب هذا الترتيب", "الخطوة الأولى المقترحة"];
  const DISCLAIMER = "هذا المشروع نموذج أولي، وليس نصيحة زراعية أو إنسانية.";
  const NOTE = "ترجمة آلية أعدّها نظام ذكاء اصطناعي وراجعها نظام ذكاء اصطناعي آخر، ولم يراجعها مترجم محترف.";

  const USE_RULES = {
    "intact|orchard": "حافظوا على إنتاج الأشجار: أعيدوا المياه والعناية بالأشجار أولاً.",
    "intact|mixed trees and crops": "أعيدوا المياه؛ حافظوا على الأشجار وازرعوا المساحات المكشوفة في الموسم القادم.",
    "intact|seasonal crops": "جاهزة لموسم الزراعة القادم: بذور وأسمدة ومياه.",
    "partly damaged|orchard": "تفقّدوا الأشجار لتحديد ما بقي منها حياً؛ أصلحوا شبكة الري وأعيدوا الزراعة في الأماكن الخالية.",
    "partly damaged|mixed trees and crops": "ازرعوا الأجزاء غير المتضررة أولاً؛ تفقّدوا الأشجار ونظّفوا الباقي.",
    "partly damaged|seasonal crops": "ازرعوا الأجزاء غير المتضررة أولاً؛ نظّفوا الباقي وافحصوه.",
    "mostly damaged|orchard": "على المدى الأبعد: نظّفوا الأرض وافحصوها، ثم أعيدوا زراعة الأشجار.",
    "mostly damaged|mixed trees and crops": "على المدى الأبعد: نظّفوا الأرض وتحققوا من التربة والسلامة قبل الزراعة.",
    "mostly damaged|seasonal crops": "نظّفوا الأرض وتحققوا من التربة والسلامة أولاً، ثم ازرعوا محاصيل موسمية.",
  };
  const BUILT = " خريطة ما قبل الحرب تُظهر منشآت هنا (ربما دفيئات زراعية): تحققوا من المنشآت.";
  const GREEN_AGAIN = " متضررة، لكنها خضراء كالمعتاد هذا الشهر: قد يعود هذا الاخضرار إلى محاصيل أُعيدت زراعتها أو أعشاب ضارة أو نموّ متجدد للنباتات.";
  const UNCERTAIN = " تقدير غير مؤكد: تحققوا ميدانياً أولاً.";
  const VERIFY = " تحققوا ميدانياً.";

  // The same sentence as Rules.reason, in Arabic. "top N%" and the two numbers are computed exactly as in English.
  function reason(use, productivityPct, damagedShare, croplandHa, status) {
    const level = R.productivityLabel(productivityPct);
    const top = Math.max(1, R.pyRound(100 * (1 - productivityPct)));
    const prod = level === "high" ? "كانت ضمن أعلى " + top + "% من القطع من حيث الاخضرار قبل الحرب" : "كان اخضرارها " + LEVEL[level] + " قبل الحرب";
    return "استخدام القطعة قبل الحرب: " + USE[use] + "؛ " + prod + "؛ يقدّر النموذج أن " + R.percent0(damagedShare) + " من مساحة القطعة البالغة " +
      R.fixed1(croplandHa) + " هكتار تضررت منذ أكتوبر 2023 (" + STATUS[status] + ").";
  }

  function suggestion(status, use, builtShare, greenNow, confidence) {
    let text = USE_RULES[status + "|" + use];
    if (builtShare >= 0.5) text += BUILT;
    if (status !== "intact" && greenNow === "at or above normal") text += GREEN_AGAIN;
    text += confidence === "lower" ? UNCERTAIN : VERIFY;
    return text;
  }

  const api = { STATUS, USE, LEVEL, GREEN, HEADERS, DISCLAIMER, NOTE, USE_RULES, reason, suggestion };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.RulesAr = api;
})(typeof self !== "undefined" ? self : this);
