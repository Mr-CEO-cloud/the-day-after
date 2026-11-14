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
- *"Why not just use the UN map?"* It stops at July 2025, and it is updated only every few months. The model can make a map for any new date (we show post-ceasefire 2025-2026 maps). It also feeds the planner, which the UN map doesn't do.
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

**The lock.** `load_table(date, "test")` raises an error unless the Phase 4 code passes `unlock_test=True`. `tests/test_no_leakage.py` runs 8 automatic checks (no shared blocks, buffer distance, no position columns in the feature list, locked test set, folds made of whole blocks, same land at every date, plots inside blocks, table matches the images).

**Three things we learned in this phase:**
- **The UN "damaged" label only ever grows.** From Dec 2024 to Jul 2025, 55,750 pixels went from non-affected to damaged and 0 went back. So "damaged" seems to mean "damaged at some point so far", even if weeds have since turned the field green. Expect the model to struggle on regrown land.
- **Orchards were hit more.** On training land, 26% of damaged pixels were trees before the war, against 13% of non-affected pixels (Dec 2024).
- **April 2026 is greener than April 2022** (median NDVI 0.38 vs 0.33 over all cropland). After a ceasefire, green can mean weeds or regrowth, not a working farm. The app must say this clearly.

### Likely judge questions
- *"Why 1 km blocks?"* Big enough that a block's neighbours don't give away its answer pixel by pixel, small enough that we still get 133 blocks to mix fairly.
- *"Did you look at the test set?"* Only its positions and class counts, to build a fair split. One sanity check in Phase 1 (median greenness change by class) used all cropland, before the split existed. No model or threshold has seen test land.
- *"Why throw away 9% of the data?"* The 20 m infrared bands are blurred onto the 10 m grid, so next-door pixels share information. The buffer makes sure no training pixel shares anything with test land.
- *"What is the weak point of this split?"* Test blocks are scattered among training blocks, so the test is "unseen fields in the same governorate", not "a new region". We add a time test (train on Dec 2024, test on Jul 2025 test blocks) as a second, harder check.
