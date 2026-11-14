# Explainers: The Day After in plain words

One section per phase, written so Russell can explain each part to judges.

## Phase 0: Setup
- **What we did:** We made a clean project folder with its own Python "virtual environment" (`.venv`). This is a private box of libraries, so this project's versions never clash with anything else on the computer.
- **Why one settings file (`src/config.py`)?** Every date, area, weight and threshold lives in one place. If a judge asks "why 1 km blocks?" or "why weight 0.6 for damaged?", the answer is in one file.
- **Why numbered scripts (01 to 08)?** Each step reads the previous step's output and saves its own. Anyone can rerun the whole project in order and get the same numbers.
- **Why `CLAUDE.md` and `AGENTS.md`?** These are instruction files for the AI tools I used (Claude Code and Codex). They hold the rules: no leakage, cite everything, no made-up facts.

### Likely judge questions
- *"Did you use AI tools?"* Yes. Claude Code wrote most of the code with me, and Codex reviewed it independently. Everything is logged in `docs/ai_log.md`, and I can explain each part.

## Phase 1: Data loads
**The three ingredients**
1. **The answer key (labels):** UNOSAT (the UN satellite centre) and FAO made maps of Gaza cropland marked "damaged" or "non-affected". We use their 28 Jul 2025 file, which also contains 17 earlier monthly maps going back to Oct 2023. Each map has only **two classes**, so our model predicts intact vs damaged. They made it by comparing Sentinel-2 greenness (NDVI) with earlier seasons, which is why our "greenness drop" baseline is close to their method. We say this openly.
2. **Satellite photos:** Sentinel-2 (European Space Agency), via Microsoft Planetary Computer. 10 m pixels, with a new photo every few days.
3. **Pre-war land cover:** ESA WorldCover 2021, which tells us which land was trees (orchards) vs crops.

**What a "composite" is:** for a time window (e.g. July 2025) we take every photo, hide the cloudy and shadowy pixels using the satellite's own cloud map (SCL), and keep the **median** value per pixel. This gives one clean, cloud-free picture.

**Why "same season in 2022":** fields are naturally greener in winter than in summer. Comparing July 2025 with July 2022 (not with February) means the change we see is damage, not the season. 2022 is the last full year before October 2023.

**One grid for everything:** every picture is put on the same 10 m grid in metres (UTM zone 36N), so pixel (row, col) is the same spot of ground in every image and in the labels.

**Two data traps we caught:**
- Since January 2022, ESA adds +1000 to stored values ("processing baseline 04.00"). We subtract it, otherwise recent images would look brighter than pre-war ones.
- Planetary Computer sometimes has **two copies of the same photo** (original + a 2024 reprocessing). For 2017-2021 one copy has the +1000 and the other doesn't. We keep only the newest copy per day, so the correction is always right.

**Sanity check (results/figures/quicklook_*.png):** land the UN marked damaged lost much more greenness than land marked intact. Dec 2024: median NDVI change -0.21 (damaged) vs -0.06 (intact). Jul 2025: -0.12 vs -0.04. The labels and the imagery agree. The gap is smaller in summer, so July is the harder test.

**Numbers (results/label_summary.json):** Khan Younis has about 4,207 ha of cropland in the UN map. Damaged share: 68% (Dec 2024) and 82% (Jul 2025).

### Likely judge questions
- *"Why not just use the UN map?"* The UN file we use ends in July 2025. The model can make a map for any new date in seconds (we show post-ceasefire 2025-2026 maps). It also feeds the planner, which the UN map doesn't do.
- *"Isn't the model just copying the UN method?"* Partly, which is why the greenness-drop rule is our main baseline. The model has to beat it on land it never saw, and at a date it wasn't trained on.

## Phase 2: The table and the split
**From pictures to a table (`src/03_features.py`).** A model reads a table, not a picture. Each row is one 10 m cropland pixel (422,273 rows per date). Each column is one measurement, called a "feature". There are 27:
- **Now** (`post_*`): the 6 Sentinel-2 bands in the recent image.
- **Then** (`pre_*`): the same 6 bands in the same month of 2022.
- **Three indices**, each for now, then and the change (`d_*` = now minus then):
  - **NDVI** = (NIR - red) / (NIR + red). Healthy leaves reflect a lot of near-infrared and little red, so high NDVI means green, living plants.
  - **NDMI** = (NIR - SWIR) / (NIR + SWIR). Short-wave infrared is absorbed by water, so NDMI tracks how much water the plants hold.
  - **BSI** (bare-soil index): high when the ground is bare earth.
- **Pre-war context** (same for every date): the average yearly green peak in 2017-2022 (productivity), how much that peak varied, how green the pixel stays in its brownest month, the yearly swing, and whether ESA WorldCover called it trees or built-up.

The pixel's position is saved for bookkeeping, but it is **never** a feature. A model that knows positions could memorise the map instead of learning what damage looks like.

**The split (`src/04_split.py`): why not random?** Neighbouring pixels look alike and usually share a label. With a random split, the model could score well by copying the pixel next door. So:
1. We cut the map into **1 km blocks** (133 blocks contain cropland).
2. A whole block goes to **train** (93 blocks) or **test** (40 blocks). The test blocks stay locked until Phase 4.
3. The **same** assignment is used at every date, so test land is never seen in training at any time.
4. The mix is fair: blocks are sorted into 3 groups by how damaged they are, and 30% of each group goes to test. This is called a stratified split.
5. **100 m buffer:** training pixels within 100 m of a test block are thrown away (9% of cropland). The closest remaining training pixel is 110 m from test land.
6. Training blocks are dealt into **5 folds**. In Phase 3 we practise on 4 folds and check on the 5th, five times ("cross-validation"), so we can tune things without touching the test set.

**Numbers (`results/split_summary.json`):** train 258,219 pixels (2,582 ha, 61%), test 125,282 pixels (1,253 ha, 30%), buffer 38,772 pixels (9%). Damaged share in train vs test: 66% vs 72% (Dec 2024) and 80% vs 84% (Jul 2025), so both sides are similar.

   The folds get the same 100 m buffer from each other (`cv_splits()` in `src/common.py`), which keeps 93-96% of the practice-training pixels.

**The lock.** `load_table(date, "test")` raises an error unless the Phase 4 code passes `unlock_test=True`. The table used for drawing maps (`"all"`) has its label column removed. `tests/test_no_leakage.py` runs 9 automatic checks: no shared blocks, buffer distance, no position columns in the feature list, locked test set, train rows only on train land, buffered folds made of whole blocks, plots inside blocks, all 34 rasters on one grid, and table values matching the images at all 5 dates.

**Independent review.** Codex (a different AI, read-only) reviewed this phase for leakage. It found no indexing bugs and agreed the stratified split is sound. It caught one real gap, that the practice folds had no buffer between them, which we fixed. It also asked for stricter checks (now checks 4, 6, 8 and 9), and reminded us to describe scores as "agreement with the UN reference map", not as independent proof of damage.

**Three things we learned in this phase:**
- **The UN "damaged" label only ever grows.** From Dec 2024 to Jul 2025, 55,750 pixels went from non-affected to damaged and 0 went back. So "damaged" seems to mean "damaged at some point so far", even if weeds have since turned the field green. Expect the model to struggle on regrown land.
- **Orchards were hit more.** On training land, 26% of damaged pixels were trees before the war, against 13% of non-affected pixels (Dec 2024).
- **April 2026 is greener than April 2022** (median NDVI 0.38 vs 0.33 over all cropland). After a ceasefire, green can mean weeds or regrowth, not a working farm. The app must say this clearly.

### Likely judge questions
- *"Why 1 km blocks?"* Big enough that a block's neighbours don't give away its answer pixel by pixel, small enough that we still get 133 blocks to mix fairly.
- *"Did you look at the test set?"* Only its positions and class counts, to build a fair split. One sanity check in Phase 1 (median greenness change by class) used all cropland, before the split existed. No model or threshold has seen test land.
- *"Why throw away 9% of the data?"* The 20 m infrared bands are blurred onto the 10 m grid, so next-door pixels share information. The buffer makes sure no training pixel shares anything with test land.
- *"What is the weak point of this split?"* Test blocks are scattered among training blocks, so the test is "unseen fields in the same governorate", not "a new region". We add a time test (train on Dec 2024, test on Jul 2025 test blocks) as a second, harder check.

## Phase 3: Opponents and the AI (`src/05_train_eval.py`)
**The question:** for each 10 m cropland pixel, did the UN mark it "damaged" or "non-affected"?

**Two simple opponents (baselines):**
1. **Majority:** always answer "damaged", the most common answer. It never looks at the image.
2. **NDVI-drop:** "damaged if greenness fell by more than X since the same month in 2022". X is chosen on training pixels only, separately for each date, by trying every possible cut and keeping the best: -0.116 for Dec 2024 and -0.060 for Jul 2025 (summer fields are less green, so a smaller drop already matters). This is close to how the UN made its map, so it is a strong rival.

**The AI: a RandomForest.** It is 200 decision trees. Each tree learns a chain of yes/no questions ("did moisture fall?", "was this land green before the war?") from a random 25% of the training pixels, and the trees vote. Rarer "intact" pixels count more, so the forest can't win by always saying "damaged". We trained three versions to see which information helps:
- `rf_ndvi`: greenness only, the same information the NDVI-drop rule sees
- `rf_spectral`: plus all 6 bands, moisture (NDMI) and bare soil (BSI)
- `rf_all`: plus pre-war context (productivity, seasonality, trees/built-up)

We also trained `lgbm_all` (LightGBM, trees built one after another, each fixing the previous one's mistakes) as a second opinion.

**The score: macro F1.** For each class we combine precision ("when I say intact, how often am I right?") and recall ("of all the intact land, how much did I find?") into one F1 number, then average the two classes. Plain accuracy would reward always saying "damaged": that gets 66-80% accuracy but finds zero intact land, and the planner needs exactly the intact land.

**Cross-validation (training land only).** The 93 training blocks are dealt into 5 folds. Five times, we fit on 4 folds (minus a 100 m buffer) and check on the 5th, so every pixel is checked once by a model that never saw its block. The test blocks are not touched.

**Results (`results/cv_scores.csv`, mean of 5 folds ± spread):**

| Method | Dec 2024 | Jul 2025 | Time test (fit Dec, check Jul) |
|---|---|---|---|
| Majority | 0.40 | 0.45 | 0.45 |
| NDVI-drop | 0.64 ± 0.02 | 0.59 ± 0.03 | 0.56 ± 0.05 |
| RF, greenness only | 0.66 ± 0.03 | 0.63 ± 0.03 | 0.63 ± 0.04 |
| RF, + bands/moisture/soil | 0.71 ± 0.02 | 0.68 ± 0.03 | 0.61 ± 0.02 |
| **RF, + pre-war context (`rf_all`)** | **0.75 ± 0.02** | **0.74 ± 0.03** | **0.69 ± 0.02** |
| LightGBM, all features | 0.74 ± 0.02 | 0.74 ± 0.02 | 0.66 ± 0.01 |

What this says:
- `rf_all` beats the NDVI-drop rule by 0.11-0.15 on average in every exam, and it wins in **all 5 folds of all 3 exams**. The smallest single-fold win is 0.08 (`results/cv_paired_gaps.csv`).
- **Each kind of information adds something.** Greenness alone gives +0.02-0.07 over the rule. Bands and moisture add more at the same date. Pre-war context adds the most.
- **The time test is the hard one.** Raw band brightness changes with the season, so the bands-only model trained on December does worse in July (0.61) than the greenness-only one (0.63). The full model holds up best (0.69).
- **For finding intact land in Dec 2024,** the NDVI-drop rule finds 64% of it and is right 49% of the time when it says "intact". `rf_all` finds 83% and is right 60% of the time.

**Chosen before opening the test:** `rf_all` is the main model and NDVI-drop is the main baseline. We decide this now, from training land only, so we can't pick whichever looks best on the test.

**What the forest leans on** (`results/feature_importance_rf_all.csv`): pre-war productivity (12%), greenness now (11%), greenness change (10%), blue band now (8%). The top feature being "how green was this land before the war" makes sense: land that was never green cannot show a drop. It also means the model partly learns which kinds of land tended to be damaged. Phase 4 checks whether it works equally well on orchards and on seasonal crops.

### Likely judge questions
- *"Why is your score only 0.75?"* It's measured on land the model never saw, against a UN map that itself has errors and counts regrown fields as damaged. A suspiciously high score would more likely mean leakage. What matters is the gap over the honest baseline, and that gap holds in every fold.
- *"Isn't the baseline handicapped?"* No. It gets its own best threshold for each date, chosen on the same training rows, and is scored on the same folds with the same metric.
- *"Why RandomForest and not a big neural network?"* We have one value per pixel per band, not images of objects. Tree models are the standard strong choice for that kind of table, train in seconds, and are explainable. A pretrained satellite model (Prithvi) is our Phase 8 experiment.

## Phase 4: The real exam (`src/05b_test.py`, `src/evaluation.py`)
**How we kept it honest**
1. We wrote the exam rules in `docs/test_plan.md` and pushed them to GitHub **before** opening the test set: which model is the main one, which numbers count, how failure examples get picked.
2. We rehearsed all the analysis code on training land first (`results/cv_report/`).
3. `05b_test.py unlock` opened the 40 test blocks **once** (Sat 14 Nov, 21:39). It refuses to run with uncommitted code and refuses to run twice. It recorded the code version in `results/test/unlock.json`.
4. Codex then recomputed the scores on its own from the saved answers (`results/test/predictions_main.csv.gz`).

**Result 1: the model beats the baseline on land it never saw** (`results/test/scores.csv`, macro F1, with 95% intervals)

| Method | Dec 2024 | Jul 2025 | Time test (learn Dec, check Jul) |
|---|---|---|---|
| Always "damaged" | 0.42 | 0.46 | 0.46 |
| Greenness-drop rule | 0.63 (0.60-0.67) | 0.56 (0.51-0.60) | 0.51 (0.46-0.56) |
| RF, greenness only | 0.65 | 0.60 | 0.60 |
| RF, + bands/moisture/soil | 0.70 | 0.67 | 0.62 |
| **RF, all features (main model)** | **0.74 (0.71-0.77)** | **0.73 (0.69-0.76)** | **0.69 (0.65-0.72)** |
| LightGBM, all features | 0.73 | 0.72 | 0.66 |

- **The gap** (main model minus the rule) is +0.10 (0.08 to 0.13), +0.17 (0.14 to 0.20) and +0.17 (0.13 to 0.21). All three intervals are above zero, so the claim in our test plan holds.
- **The practice exam was honest.** Cross-validation predicted 0.75 / 0.74 / 0.69. The real exam gave 0.74 / 0.73 / 0.69.
- **What a "95% interval" is here:** we redraw the list of 40 test blocks 2,000 times (with repeats) and recompute the score each time. 95% of those scores fall inside the interval. We redraw whole blocks, not pixels, because pixels in one block are alike.

**Result 2: the confusion matrix** (`results/test/confusion.png`), main model:

| | Model says intact | Model says damaged |
|---|---|---|
| **Dec 2024, UN: intact** (352 ha) | 289 ha (82%) | 63 ha (18%) |
| **Dec 2024, UN: damaged** (901 ha) | 236 ha (26%) | 665 ha (74%) |
| **Jul 2025, UN: intact** (202 ha) | 142 ha (71%) | 59 ha (29%) |
| **Jul 2025, UN: damaged** (1,051 ha) | 160 ha (15%) | 891 ha (85%) |

In words, for Jul 2025: when the model says "damaged" it agrees with the UN 94% of the time. When it says "intact" it agrees only 47% of the time. Intact land is rare (16% of test pixels), and rare things are hard to find without false finds.

**An uncomfortable number we report anyway:** by plain accuracy in Jul 2025, "always say damaged" gets 84% and our model gets 82.5%. That is because 84% of test pixels are damaged. But "always damaged" finds none of the intact land, which is the land the planner is looking for. This is why we use macro F1.

**Result 3: at the planner's unit, 1-hectare plots** (`results/test/plot_level.json`)
We estimate each plot's damaged share and compare it with the UN's share.

| | Greenness-drop rule | Main model (calibrated) |
|---|---|---|
| Status agrees with UN (Dec / Jul) | 56% / 55% | 72% / 80% |
| Average error of the damaged share | 0.25 / 0.28 | 0.14 / 0.11 |
| Bias of the damaged share | -0.13 / -0.22 | -0.01 / -0.02 |
| Plots ranked in the right order (rank correlation) | 0.51 / 0.40 | 0.78 / 0.70 |
| "Intact" and "mostly damaged" swapped | 8.6% / 14.1% | 0.2% / 0.2% |

- **Calibration** is the step that makes shares and hectares add up. The forest's raw score leans towards "intact" (we made rare intact pixels count more). A calibration curve, learned on training land, converts each raw score into "what share of such pixels did the UN really mark damaged". On test land the calibrated total is 875 ha damaged vs the UN's 901 ha (Dec) and 1,014 vs 1,051 ha (Jul). Simply counting pixels the model calls damaged gives 728 and 950 ha, which is too low.
- **The weakness:** the calibrated model almost never dares to call a whole plot "intact". It found 32 of 266 intact plots (Dec) and 6 of 79 (Jul). Most of the others it called "partly damaged".
- **What still works:** ranking. The 10% of plots the model calls least damaged are 28% damaged on average according to the UN, against 72% for all plots (Dec). In Jul it is 46% against 84%. So the top of the list is clearly better land, but in Jul nearly half of it is still damaged. The planner must say "check on the ground".

**Result 4: where it fails** (`results/test/subgroups.csv`)
- **Field edges.** Error rate 39% on pixels next to the other UN class, 18% on interior pixels (Dec). In Jul: 36% vs 12%. A 10 m pixel on a border mixes two fields. About a quarter of all pixels are edge pixels.
- **Bare-but-intact land.** Where the recent image is bare (NDVI below 0.10), the model calls about two thirds of the UN-intact pixels "damaged". From one image it cannot tell a resting or harvested field from a damaged one.
- **Green-but-damaged land.** Where the recent image is green (NDVI 0.20 or more), it misses 40% of the damage in Dec and 28% in Jul.
- **Productive intact land is the hardest find.** On the most productive third of land in Jul 2025, only 4.8% is still intact per the UN. The model finds 41% of it, and is right 39% of the time when it says "intact". The rule is right 9% of the time.
- **Orchards are the easiest** (6-11% error). Open cropland is the hardest land-cover class (23-30% error).
- **A guess of ours that was wrong:** we expected old damage to be missed because fields regrow. The opposite is true. Damage first mapped 10 or more months earlier is found 87% of the time. In Dec 2024, damage mapped in the last 9 months is found only about 60% of the time.

**Four failure examples** (`results/test/failures.png`). They were picked by a fixed rule, the 400 m window with the most mistakes of each kind at each date, not by hand.

| | Where, when | What happened | Likely reason, with the measurements |
|---|---|---|---|
| A | Block 13008, Dec 2024 | Missed damage: 764 of 1,096 pixels the UN marked damaged were called intact | The field was bare in December 2022 too (NDVI 0.12 then, 0.14 now). A December-to-December comparison shows no change. The UN first marked 99% of these pixels 4-9 months earlier, in a season when crops should have been growing. The greenness rule misses it as well (it flags 2% of the window). |
| B | Block 10009, Dec 2024 | False alarm: 168 of 904 pixels the UN marked non-affected were called damaged | Greenness did not drop (0.13 to 0.14). The mistakes form one strip that looks darker in the recent image. Likely the model reacted to a change in soil colour, such as ploughing or moisture. We cannot confirm this from 10 m images. |
| C | Block 14008, Jul 2025 | Missed damage: 712 of 973 pixels | Same cause as A, in summer: bare in July 2022 (0.13) and slightly greener now (0.16). 74% of these pixels were first marked damaged 10 or more months earlier. The rule flags 0% of this window. |
| D | Block 7011, Jul 2025 | False alarm: 310 of 1,186 pixels | Here greenness did fall (0.19 to 0.10) and bare soil rose, yet the UN map says non-affected. 75% of these pixels sit on a border between UN classes, in a window that is 68% damaged. Likely mixed border pixels, or fields left unplanted. |

**The lesson from A and C** is the most useful one for the next version: comparing one month with the same month of 2022 is blind to fields that are normally bare in that month. The UN looks at the whole season. Using several months of images per date is the natural next step.

**Cross-check with a published number** (`results/label_history.json`): FAO's table for March 2025 gives Khan Younis 3,230 of 4,278 ha damaged (75.5%). Our processing of the UN's March 2025 layer gives 3,194 of 4,207 ha (75.9%). The small difference comes from using a different boundary file.

### Likely judge questions
- *"How do I know you didn't tune on the test set?"* The exam rules were pushed to GitHub before the test was opened, the unlock step records the code version and refuses to run twice, and the git history shows the order. The practice scores from training land (0.75 / 0.74 / 0.69) match the test scores (0.74 / 0.73 / 0.69).
- *"Is 0.74 good?"* It is 0.10-0.17 better than the method closest to how the UN made the map, on land the model never saw. It is not good enough to act on without a field visit, and the app says so.
- *"Your accuracy is lower than always guessing damaged?"* In Jul 2025, yes: 82.5% vs 84%. Always guessing "damaged" finds zero intact land. Ours finds 71% of it.
- *"What is the biggest weakness?"* Three things: border pixels, fields that are bare in the month we look at, and the fact that intact land is rare, so many "intact" calls are wrong (47% right in Jul 2025).
- *"Why calibrate?"* So that "70% damaged" means 70%. Without it, the hectare totals are 10-19% too low.
