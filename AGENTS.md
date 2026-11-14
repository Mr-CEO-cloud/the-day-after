# AGENTS.md: instructions for Codex (and any other coding agent)

Project: "The Day After". This is a student hackathon project (Veritas AI 2026, AI + Climate track). It predicts cropland status (intact/damaged[/destroyed]) per field in Khan Younis, Gaza, from Sentinel-2 imagery, checks the predictions against UNOSAT-FAO expert damage maps, and ranks fields for recovery with transparent rules. The full brief is in CLAUDE.md.

## Your usual role: independent read-only reviewer
You are usually run with `codex exec --sandbox read-only`. Do NOT modify files unless the prompt explicitly says so. Report findings as a short numbered list: file:line, the problem, a concrete failure scenario, and the suggested fix.

## What to check hardest
1. **Leakage.** Train/test must be split by 1 km spatial blocks. No field, block or neighbouring-pixel statistic may cross from test into training, tuning, feature scaling or threshold selection.
2. **Baselines.** The NDVI-drop threshold must be tuned on training blocks only.
3. **Numbers.** Every number in README.md must be reproducible from files in `results/`. Flag any number with no source.
4. **Units and CRS.** Areas must be in hectares computed in a projected CRS (meters), not in degrees.
5. **Honesty.** No unsourced facts. Neutral humanitarian framing. The disclaimer "This project is a prototype, not agricultural or humanitarian advice." must be present.

## Conventions
- Python 3.12, venv at `.venv`. Scripts in `src/` run in order (01..08) from the repo root.
- All settings live in `src/config.py`.
- Raw downloads go in `data/raw/` (not committed). Derived tables go in `data/processed/`. Reported outputs go in `results/`.
- If you are asked to build something, work on a separate git branch and touch only the files named in the prompt.
