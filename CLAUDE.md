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
| Sat 14 Nov 21:45 | Planner design, decided from training land and confirmed on test: plot damaged share = mean CALIBRATED probability (unbiased hectares, MAE 0.11-0.14), plots need >= 20 cropland px. It rarely calls a plot "intact" (finds 8-12% of intact plots), so Phase 5 should rank with a CONTINUOUS status weight from the damaged share and always tell the user to verify on the ground. STATUS_WEIGHT in config still needs updating. | Claude (to confirm with Russell in Phase 5) |
| Sat 14 Nov 21:45 | Findings for the write-up: edge pixels fail most (36-39% error vs 12-18% interior); bare-but-intact land is called damaged about 2/3 of the time; fields bare in the compared month hide damage (failures A, C); the "regrowth hides old damage" guess was WRONG (oldest damage is found best, 87%); Jul 2025 plain accuracy 82.5% is below the 84% of always-damaged. Report all of these. | Claude (data findings) |
| Sat 14 Nov 21:35 | FAO cross-check passed: our Khan Younis March 2025 damaged share 75.9% (3,194 / 4,207 ha) vs FAO 75.5% (3,230 / 4,278 ha). UN labels revert in only 307 px over 18 snapshots (at the June 2024 mask revision). | Claude |

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
| 8 Prithvi improvement | Opus 5.5 | high | none |
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
