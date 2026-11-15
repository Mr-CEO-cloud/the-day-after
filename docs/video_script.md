# Video script (2 minutes)

The order follows the hackathon's own guide: the problem (15 s), the demo (60 s), how it works (30 s), one limit and a next step (15 s). The guide says a screen recording with your voice over it is all you need.

The words below are a draft. Say them your own way, but keep the numbers exactly as written. Every number is filled in from the results files. The limit is 2:00, so aim for 1:50 and time yourself.

| Part | Time | On screen | You say |
|---|---|---|---|
| The problem | 0:00-0:15 | Top of the page. Press **Play** on the time-lapse and let it run. | "This is Khan Younis, in Gaza, seen by satellite from 2017 to 2026. UN analysts have marked 82% of its farmland as damaged. After a ceasefire, aid teams cannot help every field at once. The Day After shows them where to look first." |
| The demo | 0:15-1:05 | Click **Open the map**. Click one of the ringed plots. Point at the two pictures and the reason. Move the damage slider. Scroll to the shortlist and switch it to Arabic and back. | "Each square is one hectare of farmland. Brighter means higher priority. This is how it looked before the war, and this is now. The AI estimates how much of it was damaged, and the tool gives a reason and a suggested first step. How much damage should count is a judgement, so the user sets it with this slider, and the list reorders. The shortlist also comes in Arabic. Does it help? On land the model never saw, a team following this list reaches 59 hectares of intact farmland. With no damage map, they reach 13." |
| How it works | 1:05-1:42 | Scroll to step 2 and show the greenness chart for two seconds. Scroll to step 4. Click **Run the model** and let the boxes fill. | "How does it work? A healthy field turns green and brown at the same times every year. My model compares each ten-metre patch of land with its own normal. It learned from 18 damage maps made by UN experts. The test land was locked away before training. You can check it yourself: this button runs the real model in your browser, on 6,000 patches it never saw. On the full test it scores 0.86. The best simple rule scores 0.78. On a governorate it had never seen, it scores 0.85." |
| A limit, and next | 1:42-1:58 | Open **Where it still fails**, then go back to the top with the disclaimer in view. | "One limit: predicting a later date, it is not clearly better than the simple rule, and it cannot see which land is safe to reach. Next I would add sharper images, and checks with people who know the land. This project is a prototype, not agricultural or humanitarian advice." |

## How to record it well
- **Use your own voice.** No AI presenter and no AI voice. Judges want to hear that you understand the project, and the finals are live.
- **Do not say your name or school.** The first judging round is blind.
- **Set up the screen:** browser at full screen (F11), zoom 110%, other tabs closed, notifications off (Windows: Do not disturb).
- **Load the page first** and click "Run the model" once before recording, so the download is already done.
- **Sound:** a quiet room and a headset or phone earbuds with a microphone. Do a ten-second test and listen to it.
- **Record in four clips,** one per row of the table (Win + Alt + R starts and stops). A mistake then costs one clip. Join them in Clipchamp, which comes with Windows.
- **Move the mouse slowly** and stop it on the thing you are talking about.
- **Captions:** Clipchamp can add them. If you use its automatic captions, tell me and I will list the tool in the README.
- **Upload** to YouTube as Unlisted, open the link in a private window to check it plays, and send it to me.

## If a judge asks you live
- **"What did the AI learn from?"** 18 monthly UN damage maps of Khan Younis, on training land only.
- **"How do you know it isn't cheating?"** The land was split into 1 km blocks and 40 were locked away. The test rules were published before each test, and each test ran once.
- **"What is your biggest weakness?"** Finding intact land. When the model says "intact" in July 2025, the UN agrees 71% of the time.
- **"What surprised you?"** A one-line rule with the season history (0.78) beat my whole first model (0.74). The right information mattered as much as the model.
- **"Who did what?"** Claude Code wrote most of the code with me and explained each part. I chose the project, the place, and made the decisions listed in `docs/ai_log.md`.
