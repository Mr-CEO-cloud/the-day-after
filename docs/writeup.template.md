# The Day After: one-page write-up

**Track:** AI + Climate · **Team:** Russell Ghanayem (solo) · **App:** https://huggingface.co/spaces/russellbuilds/the-day-after · **Code:** https://github.com/Mr-CEO-cloud/the-day-after

*This project is a prototype, not agricultural or humanitarian advice.*

**The problem.** After a ceasefire, aid agronomists must choose which fields to help first, with limited seeds, water and staff. Khan Younis governorate has the most cropland in the Gaza Strip (4,278 ha; FAO, 2025). By July 2025, UN analysts had marked {{un_share_last}} of it as damaged. There are thousands of fields, and they cannot all be checked at once.

**What we built.** A tool that reads ten years of satellite pictures and gives a shortlist. You pick a month. For each of {{n_plots}} one-hectare plots it shows how much the AI estimates was damaged, a recovery priority, a one-line reason and a suggested first step, on a map you can explore.

**What we added to the brief.** The brief asks for a land-cover classifier. We added a real user and decision, {{n_un_maps}} UN expert maps as the answer key, a transparent planner on top of the AI, a test on a governorate the model never saw, and a web page where anyone can run the trained model on test land in their own browser.

**How it works.** Every field has a rhythm: green in the growing season, brown after harvest. For each 10 m pixel we compute its normal for each calendar month from 2017 to 2022. Damage shows as months far below that normal. A LightGBM model ({{n_trees}} decision trees, {{n_features}} features) learns from the UN maps to answer: was this pixel damaged since October 2023? Land is split into 1 km blocks, and {{n_test_blocks}} blocks were locked away for testing. A plain rule then ranks plots: priority = pre-war productivity × (1 − penalty × damaged share), where the penalty is the user's choice.

**Results on unseen land (macro F1, 0 to 1).**

| | Khan Younis Dec 2024 | Khan Younis Jul 2025 | Deir al-Balah Dec 2024 | Deir al-Balah Jul 2025 |
|---|---|---|---|---|
| Always "damaged" | {{f1_majority_dec}} | {{f1_majority_jul}} | {{reg_f1_majority_dec}} | {{reg_f1_majority_jul}} |
| Rule: drop vs same month of 2022 | {{f1_ndvi_drop_dec}} | {{f1_ndvi_drop_jul}} | {{reg_f1_ndvi_drop_dec}} | {{reg_f1_ndvi_drop_jul}} |
| Our first model | {{f1_rf_all_dec}} | {{f1_rf_all_jul}} | {{reg_f1_rf_all_dec}} | {{reg_f1_rf_all_jul}} |
| Rule: worst month since Oct 2023 | {{f1_rule_war_dec}} | {{f1_rule_war_jul}} | {{reg_f1_rule_war_dec}} | {{reg_f1_rule_war_jul}} |
| **Our model** | **{{f1_v2_dec}}** | **{{f1_v2_jul}}** | **{{reg_f1_v2_dec}}** | **{{reg_f1_v2_jul}}** |

Deir al-Balah is a second governorate: none of its data were used to choose, fit, calibrate or set thresholds for any model. The model beats the strongest rule by {{gap_rule_dec}} and {{gap_rule_jul}} in Khan Younis and {{reg_gap_rule_dec}} and {{reg_gap_rule_jul}} in Deir al-Balah, and every 95% range is above zero. At the planner's unit, plot status matches the UN map for {{plot_status_v2_dec}} and {{plot_status_v2_jul}} of plots.

**Where it fails.** (1) Field borders: {{edge_err_dec}} of border pixels are wrong against {{interior_err_dec}} inside fields, because the UN map is finer (5 m) than the satellite pixel (10 m). (2) Bare-but-intact land is often called damaged. (3) Intact land is rare, so when the model says "intact" in July 2025 the UN agrees only {{precision_intact_v2_jul}} of the time. (4) Predicting a later date than any it learned from, its lead over the simple rule is not clear in Khan Younis ({{gap_rule_fwd}}, range {{gap_rule_fwd_range}}). (5) The right information mattered as much as the model: a one-line rule with the season history ({{f1_rule_war_dec}}) beats our whole first model ({{f1_rf_all_dec}}).

**Honesty.** Test rules were published before each test and each test ran once. The Khan Younis test land was scored twice (first model, then the improved one), which is why we added Deir al-Balah. A second AI recomputed every number. The UN maps are a reference, not ground truth. The tool does not know access, safety, soil, water or ownership: it gives a shortlist for field teams, not a decision.

**Next.** Sharper images and radar, more governorates, and checks with people who know the land.

**Sources.** Copernicus Sentinel-2 L2A via Microsoft Planetary Computer; UNOSAT-FAO Gaza Strip Cropland Damage Assessment (HDX, CC BY-SA); ESA WorldCover 2021; OCHA COD-AB boundaries; FAO (2025) *Land availability for cultivation in the Gaza Strip as of April 2025*. Software: Python, GeoPandas, rasterio, odc-stac, scikit-learn, LightGBM, matplotlib, and the others listed in the repository's README. AI tools: Claude Code (Anthropic: Claude Fable 5.2, Opus 5.5, Sonnet 5.5) and OpenAI Codex; every step is logged in `docs/ai_log.md`.
