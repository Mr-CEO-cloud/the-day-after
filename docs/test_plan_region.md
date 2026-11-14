# Test plan for a new place: Deir al-Balah

This file fixes the rules for testing the models on a second governorate they have never seen. It was committed and pushed **before** `src/09_region_test.py` was run. The commit that scored it is recorded in `results/area_deir_al_balah/unlock.json`.

## Why this test
Everything so far was learned and tested inside Khan Younis governorate. The test blocks there sit between training blocks, and they have now been scored twice (`docs/test_plan.md`, `docs/test_plan_v2.md`). A fair question is: does the model only know Khan Younis?

Deir al-Balah is the governorate just north of Khan Younis. No model, threshold, feature choice or decision in this project has used a single pixel or label from it. Until this plan was pushed, only its outline, its UN label rasters and its satellite images had been prepared; nothing had been predicted or scored.

## What is tested
- **Land:** all UN-mapped cropland in Deir al-Balah governorate (OCHA code PS0265), about 2,590 hectares, every pixel. There is no training or tuning there, so there is no split.
- **Answer key:** the same UNOSAT-FAO maps, with the same label rule (the UN class at the centre of each 10 m pixel).
- **Models, frozen as they are:** the files in `models/` that were fitted on Khan Younis training land, checked against their recorded fingerprints (`results/v2/final_models.json`). Nothing is refitted, recalibrated or re-thresholded.

| Name | What it is |
|---|---|
| `majority` | always the most common answer in Khan Younis training land ("damaged") |
| `ndvi_drop` | the first rule, with its Khan Younis thresholds |
| `rule_war` | the season-history rule, with its Khan Younis thresholds for each month |
| `rf_all` | the first model (v1) |
| **`v2`** | **the main model (season history, 18 months)** |

- **Features:** built by the same scripts (`03_features.py`, `03b_features_v2.py`) from Deir al-Balah's own satellite history, including its own "normal" for each calendar month.

## The three exams (same as before)
1. Same-date, Dec 2024.
2. Same-date, Jul 2025.
3. Forward in time: models fitted on maps up to Dec 2024, scored on Jul 2025.

Pixels with a missing value in the first model's features are left out for every method, as in the Khan Younis tests.

## What counts as the result
- **Headline:** macro F1 of `v2` in Deir al-Balah, next to its Khan Younis test score, with 95% intervals from a block bootstrap over Deir al-Balah's 1 km blocks (2,000 redraws, seed 42).
- **Also reported:** `v2` minus `rule_war` and `v2` minus `rf_all`, with paired intervals. We say "v2 beats X in the new place" only if the whole interval is above zero.
- Per-class precision and recall, confusion matrices and plain accuracy for every method.
- Second numbers, never the headline: the score on clean pixels; `v2` and `rule_war` scores for each of the 18 UN months; plot-level agreement; damaged hectares against the UN total; subgroups; failure windows picked by the same fixed rule.

**What we expect, written before looking:** a lower score than in Khan Younis. The model has never seen this land, and the thresholds and calibration were set for Khan Younis. If `v2` drops to the level of the simple rule or below, that is a real finding and we will report it.

## What this test can and cannot show
- It can show whether the method carries over to nearby land with the same crops, the same satellite and the same UN labelling.
- It cannot show that the method works in a different country, climate or conflict.
- Deir al-Balah borders Khan Younis. Fields near the shared border resemble Khan Younis fields.
- The answer key has the same limits as before: it was made from the same satellite's greenness, and "damaged" stays once marked.

## What we will not do
- Run it twice, or change anything after seeing the result.
- Use Deir al-Balah to improve the model. If it suggests an improvement, that goes under "next steps".
- Hide the result if it is bad.
