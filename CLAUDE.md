# CLAUDE.md: The Day After (Veritas AI Hackathon 2026)

Claude: read this whole file at the start of every session. The original brief from Russell is kept verbatim below.
The approved roadmap is at `C:\Users\russe\.claude\plans\c-users-russe-downloads-veritas-ai-hack-magical-parnas.md`.

## Working rules for Claude (summary)
- Russell is building this solo, mainly with Claude Code's help. Explain every code block in plain words, and add a section to `docs/explainers.md` at the end of each phase.
- Every number in the write-up must come from code output saved in `results/`. Never invent results, sources or facts.
- SPATIAL train/test split only (1 km blocks). The test blocks are opened once, in Phase 4. Baselines are tuned on train only.
- Keep the framing humanitarian and neutral. No unsourced facts. The disclaimer must appear in the README and the app:
  "This project is a prototype, not agricultural or humanitarian advice."
- Cite every dataset, model, library and AI tool. Log AI help in `docs/ai_log.md`, including what Russell decided.
- If stuck for 30 minutes on one thing, use that phase's fallback.
- Remind Russell to switch `/model` and `/effort` at each phase boundary (see the table below).
- Ignore `C:\Users\russe\Downloads\CLAUDE.md`. It belongs to an unrelated project.

## Decisions log
| When (Israel) | Decision | By |
|---|---|---|
| Sat 14 Nov | Project folder `C:\Users\russe\the-day-after`; Python 3.12 venv in `.venv` | Russell (accepted recommendation) |
| Sat 14 Nov | Labels = UNOSAT-FAO cropland damage geodatabases from HDX (Jul 2025, Dec 2024; CC BY-SA). No hand-labelling needed. | Claude found, Russell approved |
| Sat 14 Nov | Labels are partly circular (UNOSAT used Sentinel-2 NDVI vs 2017-2024). State this openly. The model must beat the NDVI-drop baseline on unseen blocks, pass a temporal test, and map post-label dates. | Claude proposed, Russell approved |
| Sat 14 Nov | Core model = RandomForest/LightGBM. Phase 8 stretch = Prithvi-EO-2.0 embeddings. Nemotron on Prime Intellect Lab rejected as core (LLM is a poor fit for tabular spectral data; cost and deadline risk). | Russell chose after Claude's advice |
| Sat 14 Nov | Codex CLI (two business plans: `~/.codex` = A, `~/.codex-b` = B) used as an independent read-only reviewer, plus an optional parallel app builder | Russell + Claude |
| Sat 14 Nov 16:20 | Phase 1 finding: UNOSAT labels are BINARY (Damaged / Non-affected cropland) and dissolved (no per-field polygons). The July 2025 GDB holds 18 monthly snapshots (Oct 2023 to Jul 2025). Only the July 2025 GDB is used, since it is a superset of the Dec 2024 one. | Claude (data finding) |
| Sat 14 Nov 16:20 | Unit design: the model predicts per 10 m pixel; the planner uses 1 ha plots (100 m cells). Plot status comes from the share of predicted-damaged pixels: intact <25%, partly damaged 25-75%, mostly damaged >75%. We do not claim "destroyed", because the labels don't measure it. | Claude proposed, Russell approved (Sat 17:40) |
| Sat 14 Nov 16:25 | Bug caught: Planetary Computer holds original + reprocessed (baseline 05.10) copies of the same photo. Pre-2022 originals have no +1000 offset; reprocessed copies do. Fix: keep the newest version per (day, tile) before compositing. | Claude |
| Sat 14 Nov 17:50 | Phase 2 split: 1 km blocks, stratified by damage level (3 groups), 30% of blocks to test (93 train / 40 test), seed 42, same blocks at every date. Train pixels within 100 m of a test block are dropped. 5 CV folds of whole train blocks. Main test = both label dates on test blocks. Time test = train on Dec 2024 train blocks, test on Jul 2025 test blocks. | Claude (Russell delegated) |
| Sat 14 Nov 17:50 | Test lock: `common.load_table(date, "test")` needs `unlock_test=True`. Only Phase 4 evaluation code may pass it. `tests/test_no_leakage.py` must pass before any training run. | Claude |
| Sat 14 Nov 17:55 | After the Codex review: all Phase 3 tuning must use `common.cv_splits(train)` (folds buffered 100 m from each other). Models get `X = table[FEATURES_...]` from config, never the whole table. Write-up wording: scores are "agreement with the UN reference map". | Codex found, Claude fixed |
| Sat 14 Nov 18:05 | Commit email switched to GitHub noreply (321324413+Mr-CEO-cloud@users.noreply.github.com); unpushed history rewritten. Public repo: https://github.com/Mr-CEO-cloud/the-day-after. HF username: russellbuilds (write token stored locally). | Russell |
| Sat 14 Nov 18:20 | Phase 3 CV (train blocks only, 5 buffered folds, macro F1): rf_all 0.747 / 0.741 / 0.690 (Dec / Jul / time test) vs NDVI-drop 0.636 / 0.589 / 0.562, majority 0.399 / 0.445 / 0.445. MAIN MODEL = rf_all, MAIN BASELINE = ndvi_drop, fixed BEFORE opening the test set. lgbm_all and the rf ablations are also reported. Decision threshold stays 0.5 (balanced class weights), not tuned. | Claude (Russell delegated) |
| Sat 14 Nov 17:50 | Finding: UN labels are cumulative (0 pixels go from damaged back to non-affected between Dec 2024 and Jul 2025). Regrown fields stay "damaged". Treat as an expected failure case. | Claude (data finding) |
| Sat 14 Nov 17:50 | Finding: Apr 2026 cropland is greener than Apr 2022 (median NDVI 0.38 vs 0.33). Post-ceasefire maps must warn that green can mean weeds or regrowth. Check in Phases 4/5 whether the model is reliable on these dates. | Claude (data finding) |
| Sat 14 Nov 17:50 | Feature tables are saved as NumPy `.npz`, not parquet: Windows Application Control blocks the pyarrow parquet DLL on this PC. Do not try to bypass the policy. | Claude |
| Sat 14 Nov 21:30 | Phase 4 prep (before unlock): methods moved to `src/methods.py`; all exam analysis in `src/evaluation.py`, rehearsed on training land (`results/cv_report/`); isotonic calibration curve fitted on out-of-fold scores (`models/rf_all_calibrator.joblib`); exam rules pre-registered in `docs/test_plan.md` and pushed (commit 2253c93). | Claude (Russell delegated) |
| Sat 14 Nov 21:39 | TEST SET OPENED ONCE (`results/test/unlock.json`). Test macro F1: rf_all 0.738 / 0.728 / 0.686 vs ndvi_drop 0.635 / 0.558 / 0.514 (Dec / Jul / time test). Gap +0.10 / +0.17 / +0.17, all 95% block-bootstrap intervals above zero. Do NOT change model, features, thresholds or `docs/test_plan.md` because of test results. `05b_test.py report` may be re-run (reads saved answers only). | Claude |
| Sat 14 Nov 21:42 | Planner design, decided from training land and confirmed on test: plot damaged share = mean CALIBRATED probability (unbiased hectares, MAE 0.11-0.14), plots need >= 20 cropland px. It rarely calls a plot "intact" (finds 8-12% of intact plots), so Phase 5 should rank with a CONTINUOUS status weight from the damaged share and always tell the user to verify on the ground. STATUS_WEIGHT in config still needs updating. | Claude (to confirm with Russell in Phase 5) |
| Sat 14 Nov 21:42 | Findings for the write-up: edge pixels fail most (36-39% error vs 12-18% interior); bare-but-intact land is called damaged about 2/3 of the time; fields bare in the compared month hide damage (failures A, C); the "regrowth hides old damage" guess was WRONG (oldest damage is found best, 87%); Jul 2025 plain accuracy 82.5% is below the 84% of always-damaged. Report all of these. | Claude (data findings) |
| Sat 14 Nov 21:35 | FAO cross-check passed: our Khan Younis March 2025 damaged share 75.9% (3,194 / 4,207 ha) vs FAO 75.5% (3,230 / 4,278 ha). UN labels revert in only 307 px over 18 snapshots (at the June 2024 mask revision). | Claude |
| Sat 14 Nov 21:50 | Codex recomputed all Phase 4 test metrics independently (Node): max difference 0.00005, no bug. Its provenance point (model files not in git) answered with `results/test/model_hashes.json` (all model files written 21:29-21:38, before the 21:39 unlock). Codex background runs must use `< /dev/null` or they hang waiting for input. | Codex found, Claude fixed |
| Sat 14 Nov 21:55 | Phase 5 planner: 7,613 rated plots (>= 20 cropland px); priority = productivity percentile x (1 - DAMAGE_PENALTY x calibrated damaged share), DAMAGE_PENALTY = 0.6 (values choice; app slider). On labelled dates, training pixels use cross-validation answers so every map pixel is out-of-sample. Test-land check: top 10% by our planner 29% / 13% UN-intact (Dec / Jul) vs NDVI-drop 25% / 10%, productivity-only 9% / 3%, UN oracle 52% / 25%. | Claude (Russell delegated) |
| Sat 14 Nov 21:55 | Caveat for the app: model trained on Dec + Jul only; Apr 2026 is an unseen season (2,139 plots greener than Apr 2022). Lower damaged estimates on later dates mean "looks greener", not "repaired". Access restrictions are not modelled. | Claude |
| Sat 14 Nov 22:15 | MODEL v2 approved by Russell (plan mode): season history from 118 monthly images (2017-01..2026-10), each pixel compared with its own 2017-2022 normal per calendar month; learn from all 18 UN monthly maps; stronger rule baseline `rule_war`; chosen on training land, one more test look. Prithvi DROPPED (Russell agreed via the plan). New scripts: `02b_monthly.py`, `01c_labels_monthly.py`, `03b_features_v2.py`, `05c_train_v2.py`; `05b_test.py unlock-v2 / report-v2`; `load_table(version="v2")`. | Russell + Claude |
| Sat 14 Nov 22:10 | Planning finding: the UN map is on a 5 m grid shifted 2.5 m from our 10 m pixels. Only ~40% of our cropland pixels are fully one UN class ("clean"). Clean-pixel score is reported as a second number, never the headline. | Claude (data finding) |
| Sat 14 Nov 23:20 | v2 choice: the pre-set rule (best on Dec 2024 + Jul 2025) picked `hist_2m` (0.865 CV). Russell chose `hist_18m` (0.850 CV) because hist_2m is a two-month specialist (other 16 months 0.795 vs 0.849; forward exam 0.67 vs 0.80). Departure documented in `docs/test_plan_v2.md` before the look; hist_2m reported as side result `v2_2m`. | Russell (asked), Claude recommended |
| Sun 15 Nov 00:05 | SECOND AND LAST TEST LOOK (`results/test_v2/unlock.json`, code f611891). v2 macro F1 0.856 / 0.839 / 0.789 (Dec / Jul / forward) vs v1 0.738 / 0.728 / 0.686 and rule_war 0.784 / 0.783 / 0.771. v2 - v1: +0.118 / +0.112 / +0.103, all intervals above zero. v2 - rule_war: +0.072 [0.060, 0.087], +0.057 [0.040, 0.076], forward +0.018 [-0.002, 0.041] = NOT a clear win in the forward exam; say so. CV had predicted 0.85 / 0.85 / 0.80. NO THIRD LOOK. Do not change v2. | Claude |
| Sun 15 Nov 00:05 | v2 at plot level (test): status agreement 84% / 87% (v1 72% / 80%); share error 0.085 / 0.069; finds 137 of 266 and 34 of 79 intact plots (v1: 32 and 6). v1 failure windows: mistakes fall 764 -> 131, 168 -> 44, 712 -> 184, 310 -> 119. The simple new rule alone (0.78) beats the v1 model (0.74): most of the gain is the season history, learning adds +0.06-0.07 on same-date exams. | Claude (results) |
| Sun 15 Nov 00:10 | STILL TO DO for v2 (Sunday morning): write the v2 section of `docs/explainers.md` and the v2 failure reasons; switch `06_planner.py` to v2 probabilities (v2 OOF for training pixels on label months; add "greenness now vs normal" flag); Codex recompute result is in `data/processed/codex_v2.log`. Then Phase 6 app, Phase 7 write-up. | Claude |
| Sun 15 Nov 00:15 | Planner now runs on v2 (`src/06_planner.py`, outputs `results/planner_v2/` for the 5 write-up months and `data/processed/planner_v2/` for all 21 months; v1 planner results stay in `results/planner/`). New column `green_now`. Test land, top 10% UN-intact: v2 planner 43% / 21%, rule planner 44% / 19%, v1 29% / 13%, oracle 52% / 25%. After Jul 2025 the estimate falls (3,434 / 3,199 / 2,816 ha): means "looks greener", not "repaired". THE APP SHOULD USE v2 and can offer all 21 months. | Claude |
| Sun 15 Nov 00:15 | Codex v2 recompute: all numbers match. Added `damaged_ha_v1_calibrated` to the v2 report and a dated "notes added after the look" section to `docs/test_plan_v2.md`. Wording rule for the write-up: "no test pixel or label was used to fit, tune or choose anything; the test set was not blind to the designers (second look)". | Codex found, Claude fixed |
| Sun 15 Nov 00:28 | Phase 6 app. Hugging Face now requires a paid PRO plan for Gradio Spaces (402 on create). Russell chose the FREE STATIC option. LIVE: https://huggingface.co/spaces/russellbuilds/the-day-after (page: https://russellbuilds-the-day-after.static.hf.space/). Source in `web/` (index.html, app.js, rules.js, style.css, data/); built by `src/07_precompute.py` + `src/07b_static_site.py`; upload with `HfApi.upload_folder("web", ...)` (static Space README has `sdk: static`, short_description <= 60 chars). A Gradio version stays in `app/` and runs locally (`python app/app.py`); not deployed. Tests: `tests/test_app.py`, `tests/test_web.py` (needs Node and Edge). | Russell chose, Claude built |
| Sun 15 Nov 00:28 | After ANY change to results/planner or v2 outputs, re-run: `07_precompute.py`, `07b_static_site.py`, `tests/test_app.py`, `tests/test_web.py`, then re-upload `web/` to the Space. | Claude |
| Sun 15 Nov 01:05 | NEW-REGION TEST started (Russell agreed): Deir Al-Balah governorate (OCHA PS0265, 5,836 ha, ~2,590 ha UN-mapped cropland; FAO table says 2,587 ha). Area switch: env var `DAYAFTER_AREA=deir_al_balah` -> outputs in `data/processed_deir_al_balah/` and `results/area_deir_al_balah/`; Khan Younis paths unchanged. Done: outline, label maps (2 dates + 18 months). Running in background: `02_imagery.py` and `02b_monthly.py` (logs in data/processed_deir_al_balah/). NOT DONE YET, IN THIS ORDER: write + push `docs/test_plan_region.md` BEFORE scoring; then features (03_features, 03b_features_v2 under the env var), then score the frozen Khan Younis models once on all Deir al-Balah cropland. Deir al-Balah is never used for training or tuning. | Russell + Claude |
| Sun 15 Nov 01:05 | Russell's questions answered (keep for the write-up): no "fine-tuned prioritiser" (no answer key exists for 'feeds people fastest'; the ranking stays transparent rules); Khan Younis = the GOVERNORATE with the most cropland in Gaza (FAO: 4,278 of 15,052 ha), not just the city - say "governorate" clearly; no single crop (10 m imagery cannot tell crops apart, no public per-field crop map). Next: web app redesign planned with Russell in plan mode (Fable 5.2): guided story, real satellite background, before/after slider, "run the model yourself" panel, audit trail. Higgsfield / AI-generated imagery: advised against. | Russell + Claude |
| Sun 15 Nov 01:32 | WEB APP REDESIGN live (plan approved by Russell in plan mode): guided 5-step story + Explore + honesty section; real Sentinel-2 time-lapse hero. Russell chose: real imagery (no Higgsfield / generated visuals), the 'Earth from above' palette sampled from the imagery, guided story then free play. 'Check it yourself' panel runs the exported v2 model in the browser on 6,000 seeded test pixels (matches Python 6,000 of 6,000). Files: `web/index.html`, `style.css`, `app.js` (shared + explore), `story.js`, `proof.js`, `model.js`, `rules.js`. Build order matters: `src/07_precompute.py` -> `src/07c_export_model.py` -> `src/07b_static_site.py` (07b reads model.json for facts). Test: `tests/test_web.py`. Upload `web/` with HfApi.upload_folder. Page numbers come from `content.json` facts (data-fact slots). CSS class `.num` is for table cells only (step numbers use `.stepnum`). Light theme is the default; `?theme=dark` or the footer button for dark. The region card reads `results/area_deir_al_balah/region_test.json` (keys: summary, columns, rows, note). | Russell chose, Claude built |
| Sun 15 Nov 01:47 | NEW-PLACE TEST SCORED ONCE on Deir Al-Balah (`results/area_deir_al_balah/`, plan pushed first at 3cd2a54, scored at code 1128679). Frozen Khan Younis models, all 2,592 ha, 81 blocks. v2 macro F1 0.849 / 0.867 / 0.836 (Dec / Jul / forward) vs Khan Younis test 0.856 / 0.839 / 0.789. v2 - rule_war: +0.049 [0.043, 0.057], +0.046 [0.037, 0.056], forward +0.020 [0.011, 0.030] (all above zero here). v1 0.754 / 0.777 / 0.754; rule_war 0.800 / 0.821 / 0.816. Plot status agreement 83% / 92%. Weak spots: Dec 2024 damaged ha over-estimated by 7% (2,082 vs 1,941), intact recall 66%. FAO cross-check there: 81.6% vs 81.5%. We had predicted a LOWER score in the plan; say so. DO NOT rerun or tune on it. The web card reads `region_test.json`. | Claude (Russell said go) |
| Sun 15 Nov 01:50 | STILL TO DO, in order: README write-up on the official template (results v1 -> v2 -> new place, failure cases, sources, AI tools, disclaimer); `notebooks/report.ipynb` with outputs visible; 2-minute video script; Codex check of README numbers against results/; links tested logged out; Russell reviews the page and README; submit by Sun 20:00. | Claude |
| Sun 15 Nov 01:55 | WRITE-UP DONE (Phase 7, first pass). `README.md`, `docs/writeup.md` (one page, for the submission form) and `docs/video_script.md` are GENERATED by `src/10_readme.py` from `docs/*.template.md` + the results files: edit the TEMPLATES, never the outputs, then re-run. `notebooks/report.ipynb` is built and executed by `src/11_notebook.py` (recomputes the test scores from the saved answers). README has a `VIDEO_LINK` placeholder to replace once Russell uploads the video. All public links checked logged out (200). Remaining: Codex check of the write-up (log in data/processed/codex_readme.log), Russell reads the page + README and records the video, submit before Sun 20:00 Israel time. | Claude |
| Sun 15 Nov 02:08 | Impact additions (Russell asked how to push further; he asked about linking to the 2014 war: advised no, Sentinel-2 starts 2015 and no testable answer key; listed under next steps instead). Added: (1) 'Why this could help' block from `results/planner_v2/validation.json` (intact ha reached in the top 10% of test plots: 59 vs 13 ha Dec, 29 vs 4 ha Jul; facts `reach_*`); (2) Arabic shortlist `web/rules_ar.js` (toggle, RTL, CSV with BOM, `?lang=ar`): written by Claude, REVIEWED BY CODEX at Russell's request (he does not trust his own Arabic reading); 5 corrections applied; page states no professional translator checked it; (3) print sheet (`@media print`, 'Print this shortlist'). Tests: `tests/web_arabic_check.js`, checks 7 and 8 in `tests/test_web.py`. | Russell asked, Claude built, Codex reviewed |
| Sun 15 Nov 13:10 | SECOND WEB REDESIGN (Russell: the page still looked vibecoded / AI-written). Look only; no numbers, planner text or model changed; `tests/test_web.py` passes (8 checks). New system in `web/style.css`: three surfaces from the imagery (`.sea` dark #152f31, white, `.band.sand` #e9cfb0), square corners everywhere, one typeface (Archivo variable, condensed for titles; IBM Plex Sans Arabic), no cards. Sections are `<section class="band"><div class="wrap">`. Status colours now #0f8f6f / #e8a920 / #7b2d26 (in `app.js` STATUS_COLORS and CSS vars). Priority map: darkened photo + dark-to-bright blue ramp (`blues()`), white rings for the top 15. `drawMap` options: `dark`, `dim`, `bg`; maps are sized to the satellite picture. Before/after is dragged on the picture (`--split`). Benefit bars use `data-bar` / `data-max` (story.js). Dark-mode toggle and `?theme=dark` REMOVED. Print sheet: one page in English with the map; Arabic sheet has no map and may take two pages (README says 'compact field sheet'). | Russell asked, Claude built |
| Sun 15 Nov 13:30 | Russell noticed 'Run it again' gives identical counts (correct: the model is deterministic and the sample is fixed). Added a second button `#run-random` in step 4: runs the model on a fresh random 1,000 of the 6,000 sample patches each click (`run(draw)` in `web/proof.js`, `?autorun=random` for testing), plus a one-line note on the page. Verified in headless Edge: two draws gave different counts, both matched Python 1,000 of 1,000. `tests/test_web.py` still passes (the full run is unchanged). | Russell asked, Claude recommended and built |

## Models and effort per phase
| Phase | Main session | Effort | Codex role |
|---|---|---|---|
| 0 Project setup | Sonnet 5.5 | low | none |
| 1 Data loads | Opus 5.5 | high | none |
| 2 Labels + spatial split | Fable 5.2 | high | A: read-only leakage review |
| 3 Baselines + RandomForest | Opus 5.5 | high | A: check baseline tuned on train only |
| 4 Honest evaluation | Fable 5.2 | xhigh | A: recompute metrics independently |
| 5 Planner | Opus 5.5 | medium | none |
| 6 Gradio app + HF Space | Sonnet 5.5 | medium | B (optional): app UI on branch `app` |
| 7 Write-up + video script | Fable 5.2 | high | A: check README numbers vs results/ |
| 8 Improvement = model v2 (season history; Prithvi dropped) | Fable 5.2 | high / xhigh | A: recompute v2 test metrics |
| Briefing (stretch) | Claude Haiku 5.5 API, precomputed | none | none |

## Timeline (Israel time)
Hard deadline Mon Nov 16 06:59 (= Sun 11:59 PM ET). First submission Sun 20:00. Feature freeze Sun 23:00.
0 Setup Sat 16:00-16:30 · 1 Data Sat 16:30-18:30 · 2 Labels+split Sat 18:30-20:30 · 3 Baselines+RF Sat 20:30-23:30 · SLEEP 00:00-07:30 ·
4 Evaluation Sun 08:00-10:30 · 5 Planner Sun 10:30-13:00 · 6 App Sun 13:00-17:00 · 7 Write-up, submit by Sun 20:00 · 8 Improvement, resubmit by Sun 23:00

## Data sources (cite all)
- Sentinel-2 L2A via Microsoft Planetary Computer STAC (fallback: Earth Search / AWS)
- ESA WorldCover 2021 (pre-war tree vs cropland)
- UNOSAT-FAO Gaza Strip Cropland Damage Assessment geodatabases, HDX, CC BY-SA:
  - Jul 2025: https://unosat.org/static/unosat_filesystem/4160/FAO-PAL-015_UNOSAT_Cropland_DA_GDB_July2025.zip
  - Dec 2024: https://unosat.org/static/unosat_filesystem/4072/FAO-PAL-012_UNOSAT_Cropland_DA_GDB_December2024.zip
- FAO, "Land availability for cultivation in the Gaza Strip as of April 2025" (CD5442EN/1/05.25, CC BY 4.0), one-page PDF, cross-check only: https://www.un.org/unispal/wp-content/uploads/2025/08/cd5442en.pdf. It reports damage as of MARCH 2025 (Khan Younis 3,230 / 4,278 ha = 75.5%) and availability as of April 2025 (Khan Younis: 2,610 ha not accessible, 1,290 ha damaged but accessible, 378 ha available). Earlier notes in this project wrongly dated it 28 Jul 2025; that date belongs to the UNOSAT geodatabase.
- OCHA COD-AB admin boundaries (HDX), for the Khan Younis governorate outline

## Commands
- Python: `.venv/Scripts/python.exe -I` (Windows venv). Run scripts from the repo root, e.g. `.venv/Scripts/python.exe -I src/01_labels.py`
- Leakage checks (run before any training): `.venv/Scripts/python.exe -I tests/test_no_leakage.py`
- Codex review (read-only): `codex exec --sandbox read-only "<prompt>"`. Second account: set `CODEX_HOME=C:\Users\russe\.codex-b`

---

## Original brief from Russell (verbatim)

I'm Russell, a high-school student building SOLO for the Veritas AI International High School Hackathon 2026, and I want first place. Continue planning and building with me from here. Explain things simply; I'll tell you my Python level.

DEADLINE: Mon Nov 16, 06:59 Israel time (= Sun 11:59 PM ET). I'm in Israel. Target a first full submission by Sun 20:00 Israel time, then resubmit improvements (latest version counts). Feature freeze Sun 23:00.

PROJECT: "The Day After" (track: AI + Climate, building on the land-use satellite mapping brief).
Pitch: On the day after a ceasefire, The Day After helps aid agronomists (FAO / NGOs) decide which Gaza fields can feed people fastest, using satellite AI checked against UN expert damage maps.
- Layer 1, the AI, which must be tested: for each cropland pixel or field in ONE area of Gaza (start with Khan Younis), predict the current status (intact / damaged / destroyed) from Sentinel-2 imagery. Also compute pre-war productivity (how green it was in 2017-2022) and the pre-war use (evergreen orchard vs seasonal crops).
- Layer 2, the planner, rule-based and transparent: recovery priority = pre-war productivity x status weight, plus a one-line reason and a suggested use per field ("productive orchard, damaged not destroyed: replant trees first").
- Output: a Gradio app (deployed on Hugging Face Spaces). The user picks an area and a date and gets a status map, a priority map, a table of hectares by class, and the top fields with reasons. Precompute results so the demo never breaks. Optional stretch: Claude Haiku 5.5 writes a short Arabic + English briefing.

DATA (free, cite all): Sentinel-2 L2A via Microsoft Planetary Computer STAC (pystac-client, planetary-computer, odc-stac). ESA WorldCover 2021 as the pre-war cropland/tree mask. Labels: UNOSAT-FAO Gaza cropland damage assessments. Check HDX for a shapefile/GeoJSON. If it's PDF only, build me a quick labelling helper and I'll hand-label 300-500 points. Cross-check against the FAO-UNOSAT "land available for cultivation" assessment.

WHAT JUDGES SCORE (equal weight): technical quality (beat a simple baseline, test on unseen data, show where it fails), impact (a real user and a real decision), creativity (a fresh question/place/data, not a tutorial), and clarity (understandable in 2 minutes, one takeaway). Projects that just follow a brief score low. Honest modest results beat suspicious high ones.

METHOD RULES:
- Baselines next to the model: majority class, and an NDVI-drop threshold (tuned on the training area only).
- Model: RandomForest/LightGBM on band, NDVI and NDMI features (pre vs post) first. Try a pretrained geospatial model (Prithvi) only if ahead of schedule.
- SPATIAL train/test split (different blocks of land), never random, to avoid leakage.
- Report per-class precision/recall plus a confusion matrix, and show 3+ failure examples with reasons.
- Every number in the write-up comes from code output.

PHASES (Israel time; each has a fallback if stuck 30 min):
0 Setup, Sat 16:00-16:30
1 Data loads for one area (pre-war 2022 + recent composites, WorldCover mask), Sat 16:30-18:30
2 Labels + spatial split, Sat 18:30-20:30
3 Baselines + RandomForest, Sat 20:30-23:30, then I SLEEP 00:00-07:30
4 Honest evaluation + failure cases, Sun 08:00-10:30
5 Day After planner layer, Sun 10:30-13:00
6 Gradio app + HF Spaces, Sun 13:00-17:00
7 Write-up (~1 page), video script (max 2 min), submit by Sun 20:00
8 One measured improvement, resubmit by Sun 23:00

RULES I MUST FOLLOW: all work built this weekend; AI is central; public data only; cite every dataset, model, library and AI tool (including you, Claude Code). I must understand and be able to explain every part, so explain each code block in plain words. Keep the framing humanitarian and neutral, use no unsourced facts, and include "This project is a prototype, not agricultural or humanitarian advice."

SUBMISSION: public GitHub repo or Colab with outputs visible, a ~1-page write-up (problem, what we built, what we added, how it works, results vs baseline, where it fails, next steps, sources), and an optional demo video of 2 minutes or less.
