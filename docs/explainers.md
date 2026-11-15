# Explainers: The Day After in plain words

One section per phase, written so Russell can explain each part to judges.

"I" in this file is Russell, who is working alone. Much of the work described here was done with Claude Code and checked by OpenAI Codex. `docs/ai_log.md` says who did what.

## Phase 0: Setup
- **What I did:** I made a clean project folder with its own Python "virtual environment" (`.venv`). This is a private box of libraries, so this project's versions never clash with anything else on the computer.
- **Why one settings file (`src/config.py`)?** Every date, area, weight and threshold lives in one place. If a judge asks "why 1 km blocks?" or "why weight 0.6 for damaged?", the answer is in one file.
- **Why numbered scripts (01 to 08)?** Each step reads the previous step's output and saves its own. Anyone can rerun the whole project in order and get the same numbers.
- **Why `CLAUDE.md` and `AGENTS.md`?** These are instruction files for the AI tools I used (Claude Code and Codex). They hold the rules: no leakage, cite everything, no made-up facts.

### Likely judge questions
- *"Did you use AI tools?"* Yes. Claude Code wrote most of the code with me, and Codex reviewed it independently. Everything is logged in `docs/ai_log.md`, and I can explain each part.

## Phase 1: Data loads
**The three ingredients**
1. **The answer key (labels):** UNOSAT (the UN satellite centre) and FAO made maps of Gaza cropland marked "damaged" or "non-affected". I use their 28 Jul 2025 file, which also contains 17 earlier monthly maps going back to Oct 2023. Each map has only **two classes**, so my model predicts intact vs damaged. They made it by comparing Sentinel-2 greenness (NDVI) with earlier seasons, which is why my "greenness drop" baseline is close to their method. I say this openly.
2. **Satellite photos:** Sentinel-2 (European Space Agency), via Microsoft Planetary Computer. 10 m pixels, with a new photo every few days.
3. **Pre-war land cover:** ESA WorldCover 2021, which tells me which land was trees (orchards) vs crops.

**What a "composite" is:** for a time window (e.g. July 2025) I take every photo, hide the cloudy and shadowy pixels using the satellite's own cloud map (SCL), and keep the **median** value per pixel. This gives one clean, cloud-free picture.

**Why "same season in 2022":** fields are naturally greener in winter than in summer. Comparing July 2025 with July 2022 (not with February) means the change I see is damage, not the season. 2022 is the last full year before October 2023.

**One grid for everything:** every picture is put on the same 10 m grid in metres (UTM zone 36N), so pixel (row, col) is the same spot of ground in every image and in the labels.

**Two data traps I caught:**
- Since January 2022, ESA adds +1000 to stored values ("processing baseline 04.00"). I subtract it, otherwise recent images would look brighter than pre-war ones.
- Planetary Computer sometimes has **two copies of the same photo** (original + a 2024 reprocessing). For 2017-2021 one copy has the +1000 and the other doesn't. I keep only the newest copy per day, so the correction is always right.

**Sanity check (results/figures/quicklook_*.png):** land the UN marked damaged lost much more greenness than land marked intact. Dec 2024: median NDVI change -0.21 (damaged) vs -0.06 (intact). Jul 2025: -0.12 vs -0.04. The labels and the imagery agree. The gap is smaller in summer, so July is the harder test.

**Numbers (results/label_summary.json):** Khan Younis has about 4,207 ha of cropland in the UN map. Damaged share: 68% (Dec 2024) and 82% (Jul 2025).

### Likely judge questions
- *"Why not just use the UN map?"* The UN file I use ends in July 2025. The model can make a map for any new date in seconds (I show post-ceasefire 2025-2026 maps). It also feeds the planner, which the UN map doesn't do.
- *"Isn't the model just copying the UN method?"* Partly, which is why the greenness-drop rule is my main baseline. The model has to beat it on land it never saw, and at a date it wasn't trained on.

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
1. I cut the map into **1 km blocks** (133 blocks contain cropland).
2. A whole block goes to **train** (93 blocks) or **test** (40 blocks). The test blocks stay locked until Phase 4.
3. The **same** assignment is used at every date, so test land is never seen in training at any time.
4. The mix is fair: blocks are sorted into 3 groups by how damaged they are, and 30% of each group goes to test. This is called a stratified split.
5. **100 m buffer:** training pixels within 100 m of a test block are thrown away (9% of cropland). The closest remaining training pixel is 110 m from test land.
6. Training blocks are dealt into **5 folds**. In Phase 3 I practise on 4 folds and check on the 5th, five times ("cross-validation"), so I can tune things without touching the test set.

**Numbers (`results/split_summary.json`):** train 258,219 pixels (2,582 ha, 61%), test 125,282 pixels (1,253 ha, 30%), buffer 38,772 pixels (9%). Damaged share in train vs test: 66% vs 72% (Dec 2024) and 80% vs 84% (Jul 2025), so both sides are similar.

   The folds get the same 100 m buffer from each other (`cv_splits()` in `src/common.py`), which keeps 93-96% of the practice-training pixels.

**The lock.** `load_table(date, "test")` raises an error unless the Phase 4 code passes `unlock_test=True`. The table used for drawing maps (`"all"`) has its label column removed. `tests/test_no_leakage.py` runs 9 automatic checks: no shared blocks, buffer distance, no position columns in the feature list, locked test set, train rows only on train land, buffered folds made of whole blocks, plots inside blocks, all 34 rasters on one grid, and table values matching the images at all 5 dates.

**Independent review.** Codex (a different AI, read-only) reviewed this phase for leakage. It found no indexing bugs and agreed the stratified split is sound. It caught one real gap, that the practice folds had no buffer between them, which I fixed. It also asked for stricter checks (now checks 4, 6, 8 and 9), and reminded me to describe scores as "agreement with the UN reference map", not as independent proof of damage.

**Three things I learned in this phase:**
- **The UN "damaged" label only ever grows.** From Dec 2024 to Jul 2025, 55,750 pixels went from non-affected to damaged and 0 went back. So "damaged" seems to mean "damaged at some point so far", even if weeds have since turned the field green. Expect the model to struggle on regrown land.
- **Orchards were hit more.** On training land, 26% of damaged pixels were trees before the war, against 13% of non-affected pixels (Dec 2024).
- **April 2026 is greener than April 2022** (median NDVI 0.38 vs 0.33 over all cropland). After a ceasefire, green can mean weeds or regrowth, not a working farm. The app must say this clearly.

### Likely judge questions
- *"Why 1 km blocks?"* Big enough that a block's neighbours don't give away its answer pixel by pixel, small enough that I still get 133 blocks to mix fairly.
- *"Did you look at the test set?"* Only its positions and class counts, to build a fair split. One sanity check in Phase 1 (median greenness change by class) used all cropland, before the split existed. No model or threshold has seen test land.
- *"Why throw away 9% of the data?"* The 20 m infrared bands are blurred onto the 10 m grid, so next-door pixels share information. The buffer makes sure no training pixel shares anything with test land.
- *"What is the weak point of this split?"* Test blocks are scattered among training blocks, so the test is "unseen fields in the same governorate", not "a new region". I add a time test (train on Dec 2024, test on Jul 2025 test blocks) as a second, harder check.

## Phase 3: Opponents and the AI (`src/05_train_eval.py`)
**The question:** for each 10 m cropland pixel, did the UN mark it "damaged" or "non-affected"?

**Two simple opponents (baselines):**
1. **Majority:** always answer "damaged", the most common answer. It never looks at the image.
2. **NDVI-drop:** "damaged if greenness fell by more than X since the same month in 2022". X is chosen on training pixels only, separately for each date, by trying every possible cut and keeping the best: -0.116 for Dec 2024 and -0.060 for Jul 2025 (summer fields are less green, so a smaller drop already matters). This is close to how the UN made its map, so it is a strong rival.

**The AI: a RandomForest.** It is 200 decision trees. Each tree learns a chain of yes/no questions ("did moisture fall?", "was this land green before the war?") from a random 25% of the training pixels, and the trees vote. Rarer "intact" pixels count more, so the forest can't win by always saying "damaged". I trained three versions to see which information helps:
- `rf_ndvi`: greenness only, the same information the NDVI-drop rule sees
- `rf_spectral`: plus all 6 bands, moisture (NDMI) and bare soil (BSI)
- `rf_all`: plus pre-war context (productivity, seasonality, trees/built-up)

I also trained `lgbm_all` (LightGBM, trees built one after another, each fixing the previous one's mistakes) as a second opinion.

**The score: macro F1.** For each class I combine precision ("when I say intact, how often am I right?") and recall ("of all the intact land, how much did I find?") into one F1 number, then average the two classes. Plain accuracy would reward always saying "damaged": that gets 66-80% accuracy but finds zero intact land, and the planner needs exactly the intact land.

**Cross-validation (training land only).** The 93 training blocks are dealt into 5 folds. Five times, I fit on 4 folds (minus a 100 m buffer) and check on the 5th, so every pixel is checked once by a model that never saw its block. The test blocks are not touched.

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

**Chosen before opening the test:** `rf_all` is the main model and NDVI-drop is the main baseline. I decide this now, from training land only, so I can't pick whichever looks best on the test.

**What the forest leans on** (`results/feature_importance_rf_all.csv`): pre-war productivity (12%), greenness now (11%), greenness change (10%), blue band now (8%). The top feature being "how green was this land before the war" makes sense: land that was never green cannot show a drop. It also means the model partly learns which kinds of land tended to be damaged. Phase 4 checks whether it works equally well on orchards and on seasonal crops.

### Likely judge questions
- *"Why is your score only 0.75?"* It's measured on land the model never saw, against a UN map that itself has errors and counts regrown fields as damaged. A suspiciously high score would more likely mean leakage. What matters is the gap over the honest baseline, and that gap holds in every fold.
- *"Isn't the baseline handicapped?"* No. It gets its own best threshold for each date, chosen on the same training rows, and is scored on the same folds with the same metric.
- *"Why RandomForest and not a big neural network?"* I have one value per pixel per band, not images of objects. Tree models are the standard strong choice for that kind of table, train in seconds, and are explainable. A pretrained satellite model (Prithvi) is my Phase 8 experiment.

## Phase 4: The real exam (`src/05b_test.py`, `src/evaluation.py`)
**How I kept it honest**
1. I wrote the exam rules in `docs/test_plan.md` and pushed them to GitHub **before** opening the test set: which model is the main one, which numbers count, how failure examples get picked.
2. I rehearsed all the analysis code on training land first (`results/cv_report/`).
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

- **The gap** (main model minus the rule) is +0.10 (0.08 to 0.13), +0.17 (0.14 to 0.20) and +0.17 (0.13 to 0.21). All three intervals are above zero, so the claim in my test plan holds.
- **The practice exam was honest.** Cross-validation predicted 0.75 / 0.74 / 0.69. The real exam gave 0.74 / 0.73 / 0.69.
- **What a "95% interval" is here:** I redraw the list of 40 test blocks 2,000 times (with repeats) and recompute the score each time. 95% of those scores fall inside the interval. I redraw whole blocks, not pixels, because pixels in one block are alike.

**Result 2: the confusion matrix** (`results/test/confusion.png`), main model:

| | Model says intact | Model says damaged |
|---|---|---|
| **Dec 2024, UN: intact** (352 ha) | 289 ha (82%) | 63 ha (18%) |
| **Dec 2024, UN: damaged** (901 ha) | 236 ha (26%) | 665 ha (74%) |
| **Jul 2025, UN: intact** (202 ha) | 142 ha (71%) | 59 ha (29%) |
| **Jul 2025, UN: damaged** (1,051 ha) | 160 ha (15%) | 891 ha (85%) |

In words, for Jul 2025: when the model says "damaged" it agrees with the UN 94% of the time. When it says "intact" it agrees only 47% of the time. Intact land is rare (16% of test pixels), and rare things are hard to find without false finds.

**An uncomfortable number I report anyway:** by plain accuracy in Jul 2025, "always say damaged" gets 84% and my model gets 82.5%. That is because 84% of test pixels are damaged. But "always damaged" finds none of the intact land, which is the land the planner is looking for. This is why I use macro F1.

**Result 3: at the planner's unit, 1-hectare plots** (`results/test/plot_level.json`)
I estimate each plot's damaged share and compare it with the UN's share.

| | Greenness-drop rule | Main model (calibrated) |
|---|---|---|
| Status agrees with UN (Dec / Jul) | 56% / 55% | 72% / 80% |
| Average error of the damaged share | 0.25 / 0.28 | 0.14 / 0.11 |
| Bias of the damaged share | -0.13 / -0.22 | -0.01 / -0.02 |
| Plots ranked in the right order (rank correlation) | 0.51 / 0.40 | 0.78 / 0.70 |
| "Intact" and "mostly damaged" swapped | 8.6% / 14.1% | 0.2% / 0.2% |

- **Calibration** is the step that makes shares and hectares add up. The forest's raw score leans towards "intact" (I made rare intact pixels count more). A calibration curve, learned on training land, converts each raw score into "what share of such pixels did the UN really mark damaged". On test land the calibrated total is 875 ha damaged vs the UN's 901 ha (Dec) and 1,014 vs 1,051 ha (Jul). Simply counting pixels the model calls damaged gives 728 and 950 ha, which is too low.
- **The weakness:** the calibrated model almost never dares to call a whole plot "intact". It found 32 of 266 intact plots (Dec) and 6 of 79 (Jul). Most of the others it called "partly damaged".
- **What still works:** ranking. The 10% of plots the model calls least damaged are 28% damaged on average according to the UN, against 72% for all plots (Dec). In Jul it is 46% against 84%. So the top of the list is clearly better land, but in Jul nearly half of it is still damaged. The planner must say "check on the ground".

**Result 4: where it fails** (`results/test/subgroups.csv`)
- **Field edges.** Error rate 39% on pixels next to the other UN class, 18% on interior pixels (Dec). In Jul: 36% vs 12%. A 10 m pixel on a border mixes two fields. About a quarter of all pixels are edge pixels.
- **Bare-but-intact land.** Where the recent image is bare (NDVI below 0.10), the model calls about two thirds of the UN-intact pixels "damaged". From one image it cannot tell a resting or harvested field from a damaged one.
- **Green-but-damaged land.** Where the recent image is green (NDVI 0.20 or more), it misses 40% of the damage in Dec and 28% in Jul.
- **Productive intact land is the hardest find.** On the most productive third of land in Jul 2025, only 4.8% is still intact per the UN. The model finds 41% of it, and is right 39% of the time when it says "intact". The rule is right 9% of the time.
- **Orchards are the easiest** (6-11% error). Open cropland is the hardest land-cover class (23-30% error).
- **A guess that was wrong:** I expected old damage to be missed because fields regrow. The opposite is true. Damage first mapped 10 or more months earlier is found 87% of the time. In Dec 2024, damage mapped in the last 9 months is found only about 60% of the time.

**Four failure examples** (`results/test/failures.png`). They were picked by a fixed rule, the 400 m window with the most mistakes of each kind at each date, not by hand.

| | Where, when | What happened | Likely reason, with the measurements |
|---|---|---|---|
| A | Block 13008, Dec 2024 | Missed damage: 764 of 1,096 pixels the UN marked damaged were called intact | The field was bare in December 2022 too (NDVI 0.12 then, 0.14 now). A December-to-December comparison shows no change. The UN first marked 99% of these pixels 4-9 months earlier, in a season when crops should have been growing. The greenness rule misses it as well (it flags 2% of the window). |
| B | Block 10009, Dec 2024 | False alarm: 168 of 904 pixels the UN marked non-affected were called damaged | Greenness did not drop (0.13 to 0.14). The mistakes form one strip that looks darker in the recent image. Likely the model reacted to a change in soil colour, such as ploughing or moisture. I cannot confirm this from 10 m images. |
| C | Block 14008, Jul 2025 | Missed damage: 712 of 973 pixels | Same cause as A, in summer: bare in July 2022 (0.13) and slightly greener now (0.16). 74% of these pixels were first marked damaged 10 or more months earlier. The rule flags 0% of this window. |
| D | Block 7011, Jul 2025 | False alarm: 310 of 1,186 pixels | Here greenness did fall (0.19 to 0.10) and bare soil rose, yet the UN map says non-affected. 75% of these pixels sit on a border between UN classes, in a window that is 68% damaged. Likely mixed border pixels, or fields left unplanted. |

**The lesson from A and C** is the most useful one for the next version: comparing one month with the same month of 2022 is blind to fields that are normally bare in that month. The UN looks at the whole season. Using several months of images per date is the natural next step.

**Cross-check with a published number** (`results/label_history.json`): FAO's table for March 2025 gives Khan Younis 3,230 of 4,278 ha damaged (75.5%). My processing of the UN's March 2025 layer gives 3,194 of 4,207 ha (75.9%). The small difference comes from using a different boundary file.

### Likely judge questions
- *"How do I know you didn't tune on the test set?"* The exam rules were pushed to GitHub before the test was opened, the unlock step records the code version and refuses to run twice, and the git history shows the order. The practice scores from training land (0.75 / 0.74 / 0.69) match the test scores (0.74 / 0.73 / 0.69).
- *"Is 0.74 good?"* It is 0.10-0.17 better than the method closest to how the UN made the map, on land the model never saw. It is not good enough to act on without a field visit, and the app says so.
- *"Your accuracy is lower than always guessing damaged?"* In Jul 2025, yes: 82.5% vs 84%. Always guessing "damaged" finds zero intact land. This model finds 71% of it.
- *"What is the biggest weakness?"* Three things: border pixels, fields that are bare in the month I look at, and the fact that intact land is rare, so many "intact" calls are wrong (47% right in Jul 2025).
- *"Why calibrate?"* So that "70% damaged" means 70%. Without it, the hectare totals are 10-19% too low.

**Independent check.** Codex recomputed every test number from the saved answers in a different programming language (Node.js), without reading my code first. The largest difference was 0.00005. It also pointed out that the model files are not stored in git, so I recorded their fingerprints (SHA-256 hashes) and save times in `results/test/model_hashes.json`. All were saved before the test was opened at 21:39.

## Phase 5: The Day After planner (`src/06_planner.py`)
**What it does.** It turns the AI's pixel answers into a ranked list of 1-hectare plots, each with a reason and a suggested first step. Every rule is a short line of code with its number in `src/config.py`, so anyone can check or change it.

**The unit: 1-hectare plots.** The UN map has no field outlines, so I lay a grid of 100 m × 100 m squares over Khan Younis. A plot is rated if it has at least 0.2 ha of UN-mapped cropland, which gives 7,613 plots.

**For each plot:**

| Column | How it is made |
|---|---|
| Damaged share | The average **calibrated** damage probability of its pixels. On test land, this was 0.11-0.14 away from the UN share on average, with almost no bias. |
| Status | Intact (under 25% damaged), partly damaged (25-75%), mostly damaged (75% or more). |
| Confidence | "Lower" when the share is within 0.10 of a status boundary. On training land, those plots matched the UN status only 50-67% of the time, against 73-90% for the others. |
| Productivity | How green the plot was at its yearly peak in 2017-2022, as a percentile among all plots (1.0 = greenest). The same for every date. |
| Pre-war use | ESA WorldCover 2021: orchard (50% or more trees), mixed (20-50%), or seasonal crops. |
| **Priority** | **productivity × (1 − 0.6 × damaged share)** |
| Reason | One sentence built from the numbers above. |
| Suggested use | One line from a 9-row rule table (3 statuses × 3 uses), plus warnings. |

**Why 0.6?** An intact plot keeps its full productivity score, a half-damaged plot keeps 70% of it, and a fully damaged plot keeps 40%. There is no "destroyed" class, so even damaged land keeps some value: a very productive damaged plot can outrank a poor intact one. This is a values choice, not a measured fact, so the app lets the user change it. The trade-off on training land, for the top 10% of plots in Dec 2024 (`results/planner/validation.json`):

| Damage penalty | Share of the top 10% that the UN map calls intact | Average productivity percentile of the top 10% |
|---|---|---|
| 0 (ignore damage) | 18% | 0.95 |
| 0.4 | 37% | 0.90 |
| **0.6 (the default)** | **46%** | **0.85** |
| 1.0 | 58% | 0.75 |

**Warnings the planner adds by itself:**
- "Uncertain estimate: verify on the ground first" for lower-confidence plots. Every other plot also says "Verify on the ground".
- "Pre-war map shows built-up cover here (possibly greenhouses): check structures" when 50% or more of the plot was "built-up" in WorldCover.
- "Greener than in 2022: this may be weeds or regrowth, not active farming" on dates with no UN map.

**Where the pixel answers come from (honesty).** On Dec 2024 and Jul 2025, training pixels use their cross-validation answer, so **every pixel on every map comes from a model that never saw that block**. On the three later dates, the final model answers everywhere.

**Does the ranking point to intact land?** (test land). I took the top 10% of plots under four ways of ranking and asked how much of their cropland the UN map calls intact:

| Ranking | Dec 2024 | Jul 2025 |
|---|---|---|
| Productivity only (no damage map) | 9% | 3% |
| Productivity × greenness-drop rule | 25% | 10% |
| **The planner** | **29%** | **13%** |
| Productivity × the UN map itself (best possible) | 52% | 25% |
| *All test plots, for comparison* | *28%* | *16%* |

What this says, honestly:
- **Ignoring damage would send aid almost entirely to damaged land** (9% and 3% intact), because the most productive land was hit hardest.
- **The planner beats the greenness rule, but it is far from the best possible.** Its top 10% is about as intact as an average plot, while being much more productive (0.86-0.88 percentile, against 0.5 for an average plot). Its ranking correlates 0.95-0.96 with the best-possible ranking.
- **So the planner is a shortlist for field teams, not a decision.** That is why every line says "verify on the ground".

**Hectares, all of Khan Younis** (`results/planner/summary.json`, estimated damaged, out of 4,223 ha of UN-mapped cropland):

| Date | Model estimate | UN map |
|---|---|---|
| Dec 2024 | 2,873 ha | 2,888 ha |
| Jul 2025 | 3,387 ha | 3,446 ha |
| Nov 2025 | 3,117 ha | no UN map |
| Apr 2026 | 2,415 ha | no UN map |
| Oct 2026 | 2,626 ha | no UN map |

**Be careful with the later dates.** The model was trained only on December and July images. April is the green season, and it never saw one: 2,139 plots look greener than in April 2022. The UN label means "damaged at some point", so a lower estimate does **not** mean land was repaired. It means the land looks greener, which could be crops, weeds or regrowth. I show these maps as the model's reading of the images, with no UN map to check them against.

**Not in the model, and said in the app:** access restrictions. FAO's April 2025 note lists 2,610 of 4,278 ha of Khan Younis cropland as not accessible. The planner does not know which land can be reached.

### Likely judge questions
- *"Why multiply and not add?"* Multiplying means a plot needs both things: being productive and being undamaged. Adding would let very high productivity hide heavy damage.
- *"Who chose the rules?"* I did, and they are written in `config.py`. The penalty is the user's choice in the app. The status cut-offs (25%, 75%) were set before the test and never tuned.
- *"Why is your top 10% only 29% intact?"* The most productive land was hit hardest, and the model can't find intact land perfectly. It is 3 times better than ignoring damage, better than the greenness rule, and about half of what a perfect damage map would give.

## Model v2: season history (the improvement)
*Everything above describes model v1 and the planner built on it. This section describes v2, which replaces v1 as the main model. v1 stays in the write-up as the "before".*

**The one idea.** v1 compared one month with the same month of 2022. A field that is bare every December shows no change in December, so v1 could not see damage there. v2 compares each field with **its own normal for that time of year, month after month**, the way the UN analysts did.

**How it is built**
1. **A picture for every month** (`src/02b_monthly.py`): 118 cloud-free monthly images, January 2017 to October 2026, made exactly like the Phase 1 composites.
2. **Normal** (`src/03b_features_v2.py`): for each pixel and each calendar month, the average of six pre-war years (2017-2022). That is what "normal for March" means for that pixel.
3. **Anomaly:** a month's value minus its normal. -0.20 means "much less green than usual for this time of year". I do this for greenness (NDVI), plant water (NDMI) and bare soil (BSI). A "z-score" divides the anomaly by how much that month normally varies.
4. **History features (61 in total):** the anomaly now, 1-6 months back, the lowest and average over the last 3 / 6 / 12 months, and since October 2023 (lowest point, average, how often far below normal, how long ago the lowest point), plus the pre-war facts from v1.
5. **18 lessons, not 2** (`src/01c_labels_monthly.py`): the UN file holds 18 monthly maps (Oct 2023 to Jul 2025). v2 learns from all of them, on training blocks only, so it has seen every season.
6. **The model** is LightGBM: decision trees built one after another, each correcting the previous ones' mistakes.

**A stronger rival, to stay fair.** If the model gets the season history, the simple rule must get it too. The new rule (`rule_war`) says: "damaged if the worst monthly greenness drop since October 2023 is below X", with X tuned on training land. This is close to how the UN made its map.

**How v2 was chosen (training land only, `results/v2/cv_candidates.csv`)**

| Candidate | Dec 2024 | Jul 2025 | Other 16 months |
|---|---|---|---|
| v1 model | 0.75 | 0.75 | |
| New rule | 0.77 | 0.78 | 0.80 |
| History features, learned from 2 months | 0.86 | 0.87 | 0.80 |
| **History features, learned from 18 months = v2** | **0.85** | **0.85** | **0.85** |

- **I departed from my own selection rule, and wrote it down before the test.** The rule said "best on Dec 2024 + Jul 2025", which picks the 2-month model. But that model only studied those two months. On the other 16 months it is no better than the simple rule, and in the forward exam it collapses (0.67). The app shows other months, so I chose the 18-month model. Russell made this call at 23:20 on Saturday.
- **Things that did not help, so I left them out:** neighbourhood averages, giving clean pixels more weight, bigger or smaller models, RandomForest. All were within 0.005 of the plain model.

**The second and last test look** (`results/test_v2/scores.csv`, Sun 15 Nov 00:05, macro F1 with 95% intervals)

| Method | Dec 2024 | Jul 2025 | Forward in time (learn up to Dec 2024, check Jul 2025) |
|---|---|---|---|
| Always "damaged" | 0.42 | 0.46 | 0.46 |
| v1 rule (greenness drop vs 2022) | 0.63 | 0.56 | 0.51 |
| v1 model | 0.74 | 0.73 | 0.69 |
| New rule (worst month since Oct 2023) | 0.78 | 0.78 | 0.77 |
| **v2 model** | **0.86 (0.84-0.87)** | **0.84 (0.82-0.85)** | **0.79 (0.77-0.81)** |
| *Side result: the 2-month specialist* | *0.87* | *0.87* | *0.69* |

- **v2 against v1:** +0.12, +0.11, +0.10. All three intervals are above zero.
- **v2 against the new rule:** +0.07 (0.06 to 0.09) and +0.06 (0.04 to 0.08) on the same-date exams. In the forward exam it is +0.02 (-0.002 to +0.04). That interval includes zero, so **I do not claim v2 beats the rule when predicting later dates.** My test plan set this condition in advance.
- **Training land predicted 0.85 / 0.85 / 0.80. The test gave 0.86 / 0.84 / 0.79.**
- **The specialist confirmed the decision:** 0.87 on its own two months, 0.69 forward.
- **Codex recomputed all of it** in another language; the largest difference was 0.00005.

**Three honest readings of this table**
1. **The right information matters as much as the model.** A one-line rule with season history (0.78) already beats the whole v1 model (0.74). Learning adds 0.06-0.07 on top.
2. **The honest guide for the app's later maps is the forward exam: about 0.79**, and there v2 is only about as good as the simple rule.
3. **It is a second look at the same test blocks**, and the idea for v2 came partly from v1's mistakes there. No test pixel or label was used to fit or choose anything, but a fully independent check would need fresh land.

**Did it fix what it was meant to fix?** (`results/test_v2/v1_failures_revisited.json`) In the four places where v1 failed worst:

| v1 failure | v1 mistakes | v2 mistakes |
|---|---|---|
| A: missed damage, field bare every December | 764 | 131 |
| B: false alarm, darker strip | 168 | 44 |
| C: missed damage, field bare every July | 712 | 184 |
| D: false alarm, border pixels | 310 | 119 |

**Where v2 still fails** (`results/test_v2/subgroups.csv`)
- **Borders are now most of the problem.** Error is 27% on pixels next to the other UN class and 4-5% inside fields. About two thirds of all remaining mistakes are on border pixels.
- **The answer key is finer than my pixels.** The UN map is drawn on a 5 m grid, shifted 2.5 m from my 10 m pixels. Only about 40% of my pixels are fully one UN class. On those "clean" pixels v2 scores 0.89 and 0.87 (a second number; never the headline).
- **Bare-but-intact land:** v2 still calls 42-58% of it damaged (v1: 63-68%).
- **Damage from the middle period:** in Dec 2024, damage first mapped 4-9 months earlier is found 83% of the time, against 99% for older damage.

**v2's own four failure examples** (`results/test_v2/failures.png`, picked by the same fixed rule)

| | Where, when | What happened | Likely reason, with the measurements |
|---|---|---|---|
| A | Block 11009, Dec 2024 | Missed damage: 410 of 1,273 pixels | Small patches: 55% of the missed pixels are border pixels. The UN first marked 96% of them 4-9 months earlier. The simple rule misses most of this window too (it flags 18%, the UN 43%). I do not know why the season history shows so little here. |
| B | Block 10004, Dec 2024 | False alarm: 169 of 617 pixels | An area with bright roofs in both images (likely greenhouses or sheds). It is as green now as in Dec 2022 (0.24 to 0.26), but at some point since Oct 2023 it fell well below normal, so both v2 and the rule call it damaged. The UN map does not. |
| C | Block 5003, Jul 2025 | Missed damage: 300 of 740 pixels | Land that was hardly green before the war (pre-war yearly peak 0.17, the lowest of any example) and is greener now than in Jul 2022 (0.12 to 0.22). With no greenness to lose, there is no drop to detect. |
| D | Block 10009, Jul 2025 | False alarm: 124 of 904 pixels | Thin strip fields. Greenness did fall (0.18 to 0.12) but the UN marks these strips non-affected. This is the same block as v1's failure B. |

**The planner with v2** (`results/planner_v2/`)
- **Hectares now follow the UN map month by month.** For example Dec 2024: 2,916 ha estimated, 2,888 ha UN. Jul 2025: 3,344 and 3,446.
- **Plot status agrees with the UN map 84% and 87% of the time** on test land (v1: 72% and 80%). v2 finds 137 of 266 intact plots in Dec 2024 (v1: 32).
- **The shortlist got better.** Share of the top 10% of plots that the UN map calls intact, on test land:

| Ranking | Dec 2024 | Jul 2025 |
|---|---|---|
| Productivity only | 9% | 3% |
| v1 planner | 29% | 13% |
| Productivity × the new rule | 44% | 19% |
| **v2 planner** | **43%** | **21%** |
| Productivity × the UN map itself (best possible) | 52% | 25% |

  The v2 planner is now close to the best possible. It is about as good as a planner built on the simple rule, which is the same honest lesson as above.
- **A new column, "green now?"** v2's "damaged" means damaged at some point since October 2023, like the UN's. It does not go away when a field turns green. So each plot also says whether it is as green as normal this month. A damaged plot that is green again could be replanted, or weeds. In April 2026 that is true of 4,809 plots.
- **After July 2025 there is no UN map to check against.** The estimate falls (3,434 ha in Nov 2025, 3,199 in Apr 2026, 2,816 in Oct 2026). That means land has looked greener for months. It does not prove anything was repaired.

### Likely judge questions about v2
- *"Your score jumped from 0.74 to 0.86. Why should I believe it?"* Three reasons. The practice exam on training land predicted it (0.85). A simple rule with the same information also jumps, to 0.78, so the gain is explainable. And the mistakes fell exactly where v1 was failing.
- *"Isn't this just copying the UN's method?"* Largely, yes, and I say so: the rule that mimics their method gets 0.78. The model adds 0.06-0.07 on the months the UN mapped. The value of a model is speed and reach: a new map in seconds for any month, including months with no UN map.
- *"Why didn't you use the model with the highest score?"* It only studied two months. It scored 0.87 there and 0.69 when asked about a later date. I chose the one that works in every season, and I reported both.
- *"You looked at the test set twice."* Yes, and it is written in both test plans. v2 was chosen on training land, the rules were pushed first, and I took exactly one more look.
- *"What would you do next?"* Sharper images (the answer key is on a 5 m grid), radar data that sees through clouds, and a test on a different governorate.

## The new-place test: Deir al-Balah (`src/09_region_test.py`)
**The question a judge will ask:** "Does your model only know Khan Younis?"

**What I did.** I took the models exactly as they were saved after learning from Khan Younis: no refitting, no new thresholds, no new calibration. I ran them once on **all 2,592 hectares of farmland in Deir al-Balah**, the governorate to the north. None of its data had been used to choose, fit, calibrate or set thresholds for any model. (Its own satellite history is used to describe its fields, exactly as in Khan Younis.)

**How it was kept honest**
1. The rules were written in `docs/test_plan_region.md` and pushed first (commit 3cd2a54).
2. The script refuses to run with uncommitted code or a second time, and checks that the model files are the ones that were fitted.
3. It ran once, on Sun 15 Nov at 01:47 (`results/area_deir_al_balah/unlock.json`).
4. In the plan I wrote that I expected a **lower** score than in Khan Younis. I was wrong, in the good direction.

**The result** (`results/area_deir_al_balah/region_test/scores.csv`, macro F1 with 95% ranges)

| Method | Dec 2024 | Jul 2025 | Forward in time |
|---|---|---|---|
| Always "damaged" | 0.43 | 0.46 | 0.46 |
| First rule (one month) | 0.64 | 0.63 | 0.58 |
| First model (v1) | 0.75 | 0.78 | 0.75 |
| Season rule, with Khan Younis thresholds | 0.80 | 0.82 | 0.82 |
| **Main model (v2)** | **0.85 (0.84-0.86)** | **0.87 (0.86-0.87)** | **0.84 (0.83-0.84)** |
| *v2 on Khan Younis test land, for comparison* | *0.86* | *0.84* | *0.79* |

- **The model works about as well on land it has never seen as on Khan Younis.**
- **v2 beats the season rule in all three exams here:** +0.05 (0.04 to 0.06), +0.05 (0.04 to 0.06) and, looking forward, +0.02 (0.01 to 0.03). All three ranges are above zero. In Khan Younis the forward lead was not clear; here it is small but clear. Deir al-Balah has twice as many blocks to judge from (81 against 40).
- **v2 beats the first model by 0.08 to 0.10.**
- **At 1-hectare plots,** the status agrees with the UN map for 83% (Dec) and 92% (Jul) of plots.
- **Month by month,** v2 scores between 0.81 and 0.88 on all 18 UN maps there (`by_month.csv`).
- **My label processing checks out here too:** FAO's table gives 81.5% of Deir al-Balah cropland damaged in March 2025; I get 81.6%.

**Where it is weaker there**
- In Dec 2024 it **overestimates damage**: 2,082 ha against the UN's 1,941 ha (7% too high). The calibration curve was learned in Khan Younis.
- It finds only 66% of intact land in Dec 2024 (75% in Jul 2025).
- In the first three months (Oct to Dec 2023) the simple rule is as good as the model.

**What this test does and does not show**
- It shows the method carries over to neighbouring land with the same crops, the same satellite and the same UN labelling.
- It does not show it works in another country, climate or conflict.
- Deir al-Balah borders Khan Younis, so fields near the shared border are similar.

### Likely judge questions
- *"Did you tune anything on Deir al-Balah?"* No. Nothing was changed for it, the plan was published first, and the script can only run once.
- *"Why is it as good as at home?"* The features describe each field against its own normal, so they mean the same thing in any place. And this test uses all the land, not just blocks between training blocks.
- *"Why test only one more governorate?"* Time. Each one needs 118 monthly images. The scripts work for any governorate by changing one setting.

## Phase 6: The web app
**Live link:** https://huggingface.co/spaces/russellbuilds/the-day-after (the page itself: https://russellbuilds-the-day-after.static.hf.space/). Code: `web/`.

**Why it was rebuilt.** The first version was a dashboard, and even after watching it being built Russell could not read it. The new page tells the story in five steps, then lets the visitor play. It also answers "did they fake it?" by running the real model in the visitor's own browser.

**The page from top to bottom**

| Part | What the visitor sees | What it is for |
|---|---|---|
| Top | A time-lapse of real satellite pictures of Khan Younis, 2017 to 2026 | The one "wow" moment, and every frame is real data |
| 1 The question | Three numbers: 7,613 plots, 18 UN maps, 118 monthly pictures | Who needs this and why |
| 2 What the satellite sees | A before/after slider (same month, 2022 vs now) and one field's greenness chart against its own normal | Shows what "damage" looks like, and the idea of a field's rhythm |
| 3 What UN experts say | The 18 UN maps, month by month, over the satellite picture, with a line that rises from 6% to 82% | Shows the answer key |
| 4 What the AI says | The AI's map next to the UN's, then **"Check it yourself"** | Trust (see below) |
| 5 From damage to a shortlist | The priority formula with a real plot's numbers and a slider | Shows that ranking is a plain rule and a values choice |
| Explore the map | Month, penalty slider with three presets, pre-war-use filter, one big map on the satellite picture, click a plot for a close-up | Free play |
| How I kept it honest | The order things happened in, with links; the new-place test; limits; sources | Lets a judge verify |

**"Check it yourself", in plain words.** This is the part to show a judge who doubts the numbers.
1. The page downloads the **real trained model**: 400 decision trees, 24,800 yes-or-no questions (`web/data/model.json`, made by `src/07c_export_model.py`).
2. It downloads **6,000 patches of test land the model never saw**: 3,000 from Dec 2024 and 3,000 from Jul 2025, picked at random with my fixed seed.
3. The button runs the model on each patch **in the visitor's browser** (`web/model.js`), compares its answer with the UN's, and fills in the confusion matrix while they watch. The simple rule is scored on the same patches.
4. It then states: "Your browser's answers match the answers Python saved at the test on 6,000 of 6,000 patches."
5. **"Show a patch"** draws one patch's greenness history with the AI's answer and the UN's. **"Show one the AI got wrong"** does the same for a mistake.

**What it cannot do:** it runs the *trained* model, not the *training*. Training needed 118 months of pictures and a few minutes of computing. The code and the notebook cover that.

**The sample score is close to, not equal to, the full score.** On the 6,000 patches the model scores 0.85 and 0.84. On all 250,536 test patches it scores 0.86 and 0.84. The page says this.

**Design choices**
- **Colours come from the pictures.** The page has three backgrounds: a dark "sea" (the Mediterranean in the satellite pictures), white, and "sand" (bare soil as measured, #e9cfb0). The page itself has no other colour. Colour is kept for the data: green, amber and dark red for intact, partly and mostly damaged, and blue only for priority.
- **Real pictures only.** No AI-generated images. Every picture is credited: "Contains modified Copernicus Sentinel data, 2017-2026".
- **Everything is square.** Buttons, markers and numbers have square corners, because the unit of the whole project is a square: a 10 m patch and a 1 ha plot.
- **One typeface (Archivo).** Narrow and heavy for titles, normal width for reading. Arabic text uses IBM Plex Sans Arabic.
- **Every number is filled in from the results files.** The page's HTML has empty slots, and `src/07b_static_site.py` fills them from `results/`. No result is typed by hand.

**Why a plain web page and not Gradio.** Hugging Face now charges for hosting Gradio apps. A static page is free and runs in the visitor's browser, so there is no server that can crash during judging. A Gradio version of the first design is still in `app/` and runs on your own computer.

**How I know the page is right (`tests/test_web.py`)**
- The page's data equals the planner's results.
- The page's rules print the same sentence as the Python rules for all 319,746 plot-month sentences.
- The exported model gives Python's probability on all 6,000 patches (largest difference 0.00000008, no different decisions).
- A real browser (headless Edge) opened the page for five choices of month, penalty and filter; its tables equal an independent Python calculation.
- The same browser ran the live model; its counts equal Python's.
- Every number slot is filled, and the disclaimer and both caveats are on the page.

### Likely judge questions about the app
- *"Is the AI running live?"* Yes, in step 4: your browser runs the trained model on 6,000 test patches. The maps for the whole governorate are precomputed, because running 422,000 patches for 21 months in a browser would be too slow.
- *"How do I know these 6,000 patches are not hand-picked?"* They were drawn at random with the project's fixed seed (42) from the saved test answers, and the test checks that every one of them is in that file.
- *"Why a slider?"* How much damage should count against a plot is a values choice, not a fact. The agronomist decides and sees the shortlist change.
- *"Can I use it for a real decision?"* No, and it says so at the top and the bottom. It is a shortlist for field teams, who would still need to check access, safety and the ground itself.

## Three additions for impact
**1. "Why this could help" (on the page, in the README and in the video).** One measured benefit, in hectares. Suppose a field team can only visit the top-ranked 10% of plots (228 of the 2,284 test plots):

| How the plots are chosen | Intact farmland reached, Dec 2024 | Jul 2025 |
|---|---|---|
| Most productive land first, no damage map | 13 ha | 4 ha |
| **This shortlist** | **59 ha** | **29 ha** |
| Best possible, using the UN map itself | 71 ha | 34 ha |

With the same number of visits, the shortlist reaches 4.5 times (Dec) and 7.7 times (Jul) more intact farmland than choosing by productivity alone, and about 84-85% of the best possible. The reason: the most productive land was hit hardest, so without a damage map the first visits go almost entirely to damaged fields. This is measured on test land with the UN map as the reference, and it says nothing about access or safety.

**2. The shortlist in Arabic.** The people who would use this tool read Arabic. A button above the top-15 table switches the reasons, the suggested first steps and the downloaded file to Arabic, right to left.
- The numbers and the logic are identical to English; only the words change (`web/rules_ar.js`).
- **How the Arabic was made, honestly:** Claude wrote it, and a second AI (Codex) reviewed it against the English. Codex found two real errors and three awkward phrasings, and all five were fixed. No professional translator has checked it. The page says this next to the Arabic.
- The test checks that all 319,746 plot-month sentences exist in Arabic and carry the same numbers as the English ones.

**3. A printable field sheet.** "Print this shortlist" gives a compact landscape sheet with the top 15 plots for teams with no internet. In English the map sits beside the list and it fits one page in the settings I tried. The Arabic sentences are longer, so the Arabic sheet leaves the map out and can run onto a second page.

**Why not link it to the 2014 war?** Russell asked. The satellite I use (Sentinel-2) only started in 2015, and there is no field-by-field UN answer key for 2014 in my sources, so nothing I showed could be tested. The README lists it honestly under next steps: the same method on older Landsat images could measure how long farmland took to recover after earlier conflicts.

### Likely judge questions
- *"Who checked the Arabic?"* A second AI, not a person, and the page says so. A real deployment would need a professional translator and local agronomists.
- *"Where does 59 hectares come from?"* From the test land: the UN-intact cropland inside the top 228 plots of my ranking (`results/planner_v2/validation.json`).

## The second redesign (Sunday 13:00)
**Why.** Russell looked at the page after sleeping and said it still looked machine-made. He was right: a sand background with a serif headline, every block in a rounded white card, and the same two-column layout in every section is the most common look of AI-built pages.

**What changed. Only the look. No number, no sentence of the planner, and no part of the model changed, and all 8 automatic checks pass again.**
- **Opening:** a dark sea-coloured screen. The time-lapse is large, and its sea corner fades into the page, so the coast seems to rise out of the background. The month is printed on the picture in large type. The title is also shown in Arabic (اليوم التالي, "the day after").
- **No cards.** Sections are separated by space, a thick line, or a change of background (dark for the part that runs live, sand for "Why this could help").
- **The before/after picture** is now dragged directly, with a white handle.
- **Maps:** every map is drawn on the real satellite picture. On the priority map the picture is darkened and the best plots are the brightest blue, so they light up. The top 15 are ringed in white.
- **Status colours** changed from green / orange / pink to green / amber / dark red. They differ in lightness as well as hue, and a palette checker confirmed they stay apart for colour-blind readers.
- **"Why this could help"** shows bars instead of a table: 13, 59 and 71 hectares in December 2024.
- **Removed:** the dark-mode switch. The page now has one look, with dark and light parts.

### Likely judge questions
- *"Did you design the page yourself?"* I decided what it should say and what was wrong with the first two versions; Claude Code wrote the HTML and CSS. It is in the AI log.

## "Run it on a random 1,000 of them" (Sunday 13:30)
**The question Russell asked.** When you press "Run the model" twice, the same counts come up. Is that a problem?

**No. It is what should happen.** The trained model is 400 fixed decision trees with no randomness, and the first button always uses the same 6,000 test patches. Same model plus same patches gives the same answers every time.

**But a judge could wonder whether the numbers are just stored.** So there is a second button. It picks 1,000 of the 6,000 patches at random on every click and runs the model on those. The counts and the score move a little each time (for example 0.85 and 0.88 on one draw, 0.85 and 0.85 on another), which a stored answer could not do. The page still checks every answer against the one Python saved, and they match on 1,000 of 1,000.

**Why a score from 1,000 patches moves more than one from 6,000:** fewer patches means more luck in which ones were drawn. This is the same reason the write-up reports a 95% range next to every score.

### Likely judge questions
- *"Why do I get the same numbers every time?"* The model has no randomness. Press the second button for a fresh random draw.

## The words on the page, rewritten (Sunday 16:30)
**What Russell asked.** The page showed too many statistics and the text sounded AI-written. He asked what winning hackathon pages do.

**What the advice from hackathon judges says** (blog posts by judges and organisers, linked in `docs/ai_log.md`): show the product first, open with one plain sentence a judge can repeat ("this is a tool that does X for Y"), use one number where others use ten, give one clear "you can try this" moment, and admit a limit.

**What changed on the page. No number changed, only where it sits and how the sentences read.**
- The page now speaks as "I", because one student made it. "We" and "our AI" are gone.
- The opening sentence says what the tool is and who it is for.
- The row of three big numbers in step 1 (118 pictures, 18 maps, 7,613 plots) is gone. Those numbers are still in the README.
- After the live run, the four-box counts stay. The table that repeated them is behind a toggle called "The same counts as a table".
- The score gaps (+0.12, +0.07 and so on) are behind a toggle called "The exact gaps". The score chart and the sentence about the forward test stay in view.
- "Why this could help" now gives the plain hectares (59 against 13, and 29 against 4). The "times more" and "percent of the best list" figures were dropped from the page because the bars already show them.
- The Deir al-Balah table is behind a toggle called "The full table". Its two-sentence summary stays in view.
- "Hectares by status" moved under the shortlist, so the shortlist comes right after the map.
- Lists that started every item with a bold label were rewritten as ordinary sentences.

**What did not change:** every number, the model, the planner sentences, the Arabic, the maps, the disclaimer (still at the top and the bottom) and the sentence "Looking forward in time it is not clearly better than the simple rule".

### Likely judge questions
- *"Did you write this page yourself?"* The wording was drafted with Claude Code and I edited it. The AI tools are listed under Sources on the page and every step is in the AI log.
- *"Where are the detailed numbers?"* Behind the toggles on the page, and in full in the README and the results folder.
