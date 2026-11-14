---
title: The Day After
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
license: cc-by-sa-4.0
short_description: Which Gaza cropland could feed people fastest?
---

# The Day After

Satellite AI, checked against UN expert damage maps, that suggests which cropland in Khan Younis (Gaza) could be brought back into use first after a ceasefire.

**This project is a prototype, not agricultural or humanitarian advice.**

- Code, test plans, results and the log of every decision: https://github.com/Mr-CEO-cloud/the-day-after
- Built for the Veritas AI International High School Hackathon 2026 (AI + Climate).
- This is a static page: it only displays precomputed results (`data/plots.json`), runs in your browser, and never calls a satellite service.
- Data: Sentinel-2 (ESA, via Microsoft Planetary Computer), ESA WorldCover 2021, UNOSAT-FAO cropland damage assessment (HDX, CC BY-SA), OCHA COD-AB boundaries. Full citations are in the page's "About" tab.
