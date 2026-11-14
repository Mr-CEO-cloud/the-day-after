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
