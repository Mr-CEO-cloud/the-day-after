# Test plan for model v2: written before the second look at the test set

This file fixes the rules for scoring model v2 on the 40 test blocks. It was committed and pushed **before** `src/05b_test.py unlock-v2` was run. The commit that took the second look is recorded in `results/test_v2/unlock.json`.

The first look was model v1 (`docs/test_plan.md`, `results/test/`). That plan promised: one improvement, chosen on training land, scored on the same test blocks once, reported whether or not it helps. This is that improvement.

## What v2 changes
v1 compared one month with the same month of 2022. v2 gives the model each pixel's **season history**:
- One cloud-free image per calendar month from Jan 2017 to Oct 2026 (118 months).
- For each pixel and calendar month, its **normal** value from 2017-2022.
- 61 features describing how far the pixel is below its own normal: now, 1-6 months back, over the last 3 / 6 / 12 months, and since October 2023.
- It learns from all **18 monthly UN maps** (Oct 2023 to Jul 2025), on training blocks only. v1 learned from 2.

Nothing else changes: same 1 km blocks, same test blocks, same 100 m buffer, same labels (the UN class at the pixel centre), same metric.

## How v2 was chosen (training land only)
Cross-validation on the 93 training blocks, with the same five buffered folds as v1 (`results/v2/cv_candidates.csv`, macro F1):

| Candidate | Dec 2024 | Jul 2025 | Mean of the other 16 months |
|---|---|---|---|
| v1 model (`rf_all`) | 0.750 | 0.746 | not trained for them |
| v1 rule (greenness drop vs 2022) | 0.637 | 0.591 | |
| New rule (`rule_war`): worst monthly greenness drop since Oct 2023 | 0.772 | 0.775 | 0.802 |
| History features, learned from 2 months (`hist_2m`) | 0.860 | 0.870 | 0.795 |
| **History features, learned from 18 months (`hist_18m`) = v2** | **0.853** | **0.848** | **0.849** |
| + clean-pixel weights | 0.852 | 0.845 | 0.847 |
| + neighbourhood averages | 0.853 | 0.847 | 0.848 |
| RandomForest in place of LightGBM | 0.829 | 0.844 | 0.843 |
| LightGBM, smaller / bigger / slower settings | 0.846 / 0.855 / 0.852 | 0.840 / 0.849 / 0.845 | 0.845 / 0.849 / 0.848 |

### We departed from our own selection rule, and here is why
The rule we set before running anything was: *highest mean score on Dec 2024 + Jul 2025; within 0.005 the simpler candidate wins.* Applied mechanically, it picks `hist_2m` (0.865 against 0.850 for `hist_18m`).

We chose `hist_18m` instead. `hist_2m` is a specialist: it learned from exactly the two months the rule looks at.
- On the other 16 months it scores 0.795, below the simple new rule (0.802). `hist_18m` scores 0.849.
- Learning from Dec 2024 only and checking Jul 2025, it drops to 0.673, below both the new rule (0.768) and v1 (0.694). `hist_18m` gets 0.799 in the same exam.
- The app shows November, April and October maps. A model that only works in the two months it studied is the wrong model for that.

The rule was flawed: it rewarded specialising on two months. We found this on training land, before any test look. Russell made the decision on Sat 14 Nov at about 23:20, after seeing the numbers above. Both files are kept: `results/v2/chosen.json` records the rule's pick, the final choice, and that we departed.

To hide nothing, **`hist_2m` is scored on the test set too**, as a declared side result (`v2_2m`). It cannot become the headline, whatever it scores.

The extras that did not help (clean-pixel weights, neighbourhood, other settings) are left out: all are within 0.005 of plain `hist_18m`, and the simpler model wins a tie. The decision threshold stays 0.5 (tuning it gained 0.002).

## What is scored
Six methods on the same test pixels, labels and blocks as v1:

| Name | What it is | Role |
|---|---|---|
| `majority` | always the most common answer | floor |
| `ndvi_drop` | v1's rule | v1 baseline |
| `rule_war` | new rule: worst monthly greenness drop since Oct 2023, one threshold per month, tuned on training land | **main baseline for v2** |
| `rf_all` | the v1 model, exactly as tested in the first look | reference |
| `v2_2m` | `hist_2m`, the specialist | side result only |
| **`v2`** | **`hist_18m`** | **main model** |

All models are already fitted and saved. Their file fingerprints are in `results/v2/final_models.json`, and the unlock step refuses to run if a file has changed.

## The three exams
1. **Same-date, Dec 2024.**
2. **Same-date, Jul 2025.**
3. **Forward in time:** learn from UN maps up to Dec 2024 only, score Jul 2025 on test blocks. For `rule_war`, the Dec 2024 threshold is applied to Jul 2025.

## What counts as the result
- **Headline:** macro F1 of `v2` minus `rule_war`, and of `v2` minus the v1 model `rf_all`, in each exam, with 95% intervals from the paired block bootstrap (2,000 redraws of the 40 test blocks, seed 42).
- **We claim "v2 beats X on unseen land" only if the whole interval is above zero.** Otherwise we say it did not.
- Always shown with it: per-class precision and recall, confusion matrices, plain accuracy, for every method.
- **Second number, never the headline:** the score on "clean" pixels (fully cropland and fully one class in the UN's 5 m map; about 40% of pixels). It shows how much of the remaining error comes from our 10 m pixels being coarser than the answer key.

**What cross-validation leads us to expect:** v2 about 0.85 / 0.85 / 0.80; ahead of `rule_war` by about 0.08 / 0.07 / 0.03; ahead of v1 by about 0.10. If the test says something else, the test numbers are what we report.

## Also reported, same as v1
- **Plot level** (1 ha plots with at least 20 cropland pixels): status agreement, error and bias of the damaged share, rank correlation, for `rule_war`, v2 and v1.
- **Hectares** of damaged cropland on test land: UN, v2 calibrated, v1 calibrated, pixel counts.
- **Subgroups:** land cover, pre-war productivity, edge vs interior, greenness now, time since first marked damaged.
- **Failure examples** picked by the same fixed rule (the 400 m window with the most mistakes of each kind at each date), now for v2.
- **v1's four failure windows revisited:** how many of those mistakes v2 still makes.

## Stated openly
- **This is a second look.** The test blocks were scored once before, for v1.
- **The idea for v2 came partly from v1's mistakes on test land** (failure cases A and C: fields bare in the compared month). The same pattern had shown up earlier on training land, in the rehearsal before the first look, and the pilot that confirmed the fix ran on training land only. Still, the test set is not fully blind for v2. No v2 model, feature, threshold or choice has used a test pixel or a test label.
- **The forward exam is the honest guide for the app's later maps.** Nov 2025, Apr 2026 and Oct 2026 lie after the last UN map. Cross-validation says to expect about 0.80 there, with only a small lead over the simple rule, and no UN map exists to check them.
- **The UN map is the reference, not ground truth.** It was made from the same satellite's greenness history, which v2 now also sees. A high score means "agrees with the UN analysts", and `rule_war` shows how far a simple rule gets with the same information.

## What we will not do
- Take a third look, or change any model, feature, threshold or rule after this one.
- Move the headline to whichever method or exam looks best.

---
## Notes added after the look (Sun 15 Nov, 00:15)
Everything above this line is as it was when the test set was scored (commit f611891). These notes were added afterwards.

1. **A promised number was missing from the first report.** The plan lists "v1 calibrated" hectares. The report code did not print it. It has been added to `results/test_v2/area.json` (`damaged_ha_v1_calibrated`). It is computed from v1 answers that were saved at the first look, so nothing was re-scored.
2. **One sentence above is too broad.** "No v2 model, feature, threshold or choice has used a test pixel or a test label" is true for the data the models were fitted and chosen on. But the *idea* for v2 came partly from v1's mistakes on test land, as the same paragraph says. The accurate statement is: no test pixel or label was used to fit, tune or choose anything, and the test set was not blind to the designers. A fully independent check would need fresh blocks. Codex's review raised both points (`docs/ai_log.md`).
