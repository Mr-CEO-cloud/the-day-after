# Video script (2 minutes)

Read it aloud once with a timer before recording. Speak slowly: the text is about 290 words, which is 2 minutes at a calm pace. Every number below is filled in from the results files.

**How to record (Windows):** open the app in your browser at full screen. Press **Win + Alt + R** to start and stop a screen recording with your microphone (Xbox Game Bar), or use Clipchamp or OBS. Upload to YouTube as **Unlisted**, then give me the link and I will put it in the README.

| Time | On screen | You say |
|---|---|---|
| 0:00-0:17 | Top of the page. Press **Play** on the time-lapse. | "This is Khan Younis, in Gaza, seen from space between 2017 and 2026. By July 2025, UN analysts had marked {{un_share_last}} of its farmland as damaged. After a ceasefire, aid teams will have to choose which fields to help first. I built The Day After to give them a shortlist." |
| 0:17-0:37 | Scroll to step 2. Drag the before/after handle. Point at the chart. | "Every field has a rhythm: green in the growing season, brown after harvest. Damage breaks that rhythm. My model compares each patch of land with its own normal, month after month, the way the UN's analysts do." |
| 0:37-1:07 | Scroll to step 4. Click **Run the model**. Let the numbers fill in. | "You don't have to believe my scores. This button runs the trained model in your own browser, on {{n_sample}} patches of land it never saw, and compares each answer with the UN's. On the full test it scores {{f1_v2_dec}}, against {{f1_rule_war_dec}} for the best simple rule. Then I tested it once on Deir al-Balah, a governorate it had never seen. It scored {{reg_f1_v2_dec}} and {{reg_f1_v2_jul}}." |
| 1:07-1:32 | Scroll to Explore. Move the slider. Click a plot. | "The AI only says what was damaged. Choosing what to help first is a judgement, so that part is a plain rule: how productive the land was, times how undamaged it is. The slider is the user's choice. Click any plot and you see it before and now, and a suggested first step. With the same number of field visits, this list reaches {{reach_times_dec}} times more intact farmland than choosing the most productive land without a damage map." |
| 1:32-1:50 | Scroll to "Where it still fails" or "How I kept it honest". | "It is not perfect. It struggles at field borders and on bare land, and when it predicts a later date it is not clearly better than a simple rule. It does not know which land is safe or reachable." |
| 1:50-2:00 | Top of the page again, disclaimer visible. | "So it gives a shortlist for people on the ground, not a decision. The code, the test rules and every number are public. This project is a prototype, not agricultural or humanitarian advice." |

## If a judge asks you live
- **"What did the AI learn from?"** {{n_un_maps}} monthly UN damage maps of Khan Younis, on training land only.
- **"How do you know it isn't cheating?"** The land was split into 1 km blocks and {{n_test_blocks}} were locked away. The test rules were published before each test, and each test ran once.
- **"What is your biggest weakness?"** Finding intact land. When the model says "intact" in July 2025, the UN agrees {{precision_intact_v2_jul}} of the time.
- **"What surprised you?"** A one-line rule with the season history ({{f1_rule_war_dec}}) beat my whole first model ({{f1_rf_all_dec}}). The right information mattered as much as the model.
- **"Who did what?"** Claude Code wrote most of the code with me and explained each part. I chose the project, the place, and made the decisions listed in `docs/ai_log.md`.
