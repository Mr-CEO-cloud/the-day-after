# AI assistance log

The hackathon rules require listing every AI tool and what it did. This log records what each AI did, and what Russell decided.

| When (Israel) | Tool / model | What it did | Russell's decision / check |
|---|---|---|---|
| Sat 14 Nov, before 16:00 | Claude Code (Claude Opus 5.5) | Read the hackathon guide, briefs and sample submission. Searched HDX and found the UNOSAT-FAO cropland damage geodatabases (CC BY-SA). Flagged that the UNOSAT labels are themselves NDVI-based (circularity risk). Wrote the roadmap. | Approved the roadmap. Chose RandomForest core + Prithvi stretch over fine-tuning Nemotron on Prime Intellect Lab. |
| Sat 14 Nov, ~15:50 | Claude Code (Claude Opus 5.5) | Phase 0: created the project folder, CLAUDE.md, AGENTS.md, settings, config, stubs, venv. | Accepted the recommended folder location. |
| Sat 14 Nov, 16:00-16:30 | Claude Code (Claude Opus 5.5, high effort) | Phase 1: downloaded and inspected the UNOSAT GDBs and found the labels are binary and dissolved, with monthly snapshots. Built the label rasters, the Sentinel-2 composite pipeline (cloud mask, median, +1000 offset correction) and the quick-look check. Caught and fixed a duplicate-scene / offset bug. Chose pixel model + 1 ha planner plots. | Russell approved the 1 ha plot design with 3 status levels ("ok to whatever you think is best"). |
| Sat 14 Nov, 17:40-18:10 | Claude Code (Claude Fable 5.2, high effort) | Phase 2: built the 27-feature pixel table for 5 dates, the stratified 1 km block split with a 100 m buffer and 5 CV folds, a locked test loader, and 8 automatic leakage checks. Found that the UN labels are cumulative (no pixel returns to non-affected). Switched table storage from parquet to NumPy .npz because Windows Application Control blocked the pyarrow parquet DLL. | Russell said "go" and left the design choices to Claude. |
