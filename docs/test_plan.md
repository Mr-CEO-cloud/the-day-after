# Test plan: written before the test set was opened

This file fixes the rules of the final exam. It was committed and pushed **before** `src/05b_test.py unlock` was run. The commit that opened the test set is recorded in `results/test/unlock.json`, so the order can be checked in the git history.

Why bother: if you look at the test results first and choose the rules afterwards, you can always find a way to look good. Fixing the rules first removes that option.

## What is being tested
- **Question:** for each 10 m cropland pixel in Khan Younis, does the method's answer (intact / damaged) agree with the UNOSAT-FAO map?
- **Test land:** 40 blocks of 1 km × 1 km, about 125,000 cropland pixels per date. No model, threshold or decision has used them. Training pixels within 100 m of them were dropped.
- **Main model:** `rf_all` (RandomForest, all 27 features, decision at score 0.5). Chosen from cross-validation on training land.
- **Main baseline:** `ndvi_drop` (greenness-drop rule, one threshold per date, tuned on training land).
- **Also reported:** `majority`, `rf_ndvi`, `rf_spectral`, `lgbm_all`.
- All models are already fitted and saved. Nothing is refitted or retuned after unlocking.

## The three exams
1. **Same-date, Dec 2024:** models fitted on training land (both dates), scored on test land, Dec 2024.
2. **Same-date, Jul 2025:** the same, Jul 2025.
3. **Time test:** models fitted on Dec 2024 training land only, scored on Jul 2025 test land (new place and new season).

## What counts as the result
- **Headline number:** macro F1 of `rf_all` minus macro F1 of `ndvi_drop`, in each exam, with a 95% interval from a block bootstrap (2,000 redraws of the 40 test blocks, seed 42).
- **Claim we will make only if it holds:** "the model beats the baseline on unseen land" requires the whole 95% interval of the gap to be above zero. If it is not, we say so.
- Always reported with it: per-class precision and recall, the confusion matrix, and plain accuracy, for every method.

## Plot level (the planner's unit)
- A plot is a 100 m × 100 m cell with at least 20 cropland pixels on test land.
- "Truth" is the share of the plot's cropland pixels the UN marked damaged. Status: intact < 25%, partly damaged 25–75%, mostly damaged ≥ 75%.
- We report three estimators of that share: the baseline's pixel share, the model's pixel share, and the model's average **calibrated** probability (calibration curve learned on training land only).
- Reported for each: status agreement, average error of the share, bias, rank correlation, and how often intact and mostly damaged are swapped.
- **Decision already made from training land:** the planner will use the calibrated share. On training land it had the smallest error and almost no bias, but it rarely calls a plot "intact". We report that weakness on test land as well.

## Hectares
Damaged hectares on test land: UN map vs calibrated model vs pixel counts.

## Where it fails
- **Subgroups** (fixed list): pre-war land cover (ESA WorldCover), pre-war productivity (cut points from training land), edge vs interior of UN label patches, greenness in the recent image, and how long ago the UN first marked the pixel damaged.
- **Failure examples** are picked by a rule, not by hand: for each date, the 400 m window with the most "missed damage" pixels and the window with the most "false alarm" pixels. That gives four examples. We explain each using the measured numbers, and we say "likely" where we cannot be sure.

## What we will not do
- Change the model, features, thresholds or these rules after seeing test results.
- Report only the exams or subgroups where the model looks good.
- Phase 8 (one improvement) will be chosen by cross-validation on training land, then scored on the same test blocks once, and reported whether or not it helps. We will state clearly that the test set had already been seen by then.

## Known limits, stated in advance
- The UN map is the reference, not ground truth. It was made from the same satellite's greenness (NDVI), so scores measure agreement with that map.
- The UN "damaged" label is cumulative: land that has regrown stays "damaged".
- Test blocks lie between training blocks in the same governorate. This tests unseen fields, not a new region.
