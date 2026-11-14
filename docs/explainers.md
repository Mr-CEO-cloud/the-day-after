# Explainers: The Day After in plain words

One section per phase, written so Russell can explain each part to judges.

## Phase 0: Setup
- **What we did:** We made a clean project folder with its own Python "virtual environment" (`.venv`). This is a private box of libraries, so this project's versions never clash with anything else on the computer.
- **Why one settings file (`src/config.py`)?** Every date, area, weight and threshold lives in one place. If a judge asks "why 1 km blocks?" or "why weight 0.6 for damaged?", the answer is in one file.
- **Why numbered scripts (01 to 08)?** Each step reads the previous step's output and saves its own. Anyone can rerun the whole project in order and get the same numbers.
- **Why `CLAUDE.md` and `AGENTS.md`?** These are instruction files for the AI tools I used (Claude Code and Codex). They hold the rules: no leakage, cite everything, no made-up facts.

### Likely judge questions
- *"Did you use AI tools?"* Yes. Claude Code wrote most of the code with me, and Codex reviewed it independently. Everything is logged in `docs/ai_log.md`, and I can explain each part.
