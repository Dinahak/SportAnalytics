# User Test Protocol (Week 11, part 4)

**Goal:** find out whether someone who was not in the project group can read the tool, use it for a baseline and two what-if scenarios, and say what decision it could inform.
**Participant:** at least one person outside the group, ideally someone who knows football recruitment, transfers or scouting. If you use a substitute (for example a student or a sports-analytics colleague), say so in the results and explain the limitation: a substitute does not carry a Head of Recruitment's real priorities.
**Time:** 20 to 30 minutes. **Data:** synthetic only. Do not show any restricted file (Transfermarkt or StatsBomb data).

## Rules for the facilitator
- Read the tasks aloud. Do **not** explain the screens or hint at answers. Ask the participant to think aloud.
- Write down exactly what they say and do (quotes help), including every moment of hesitation, wrong click, or misreading.
- Do not correct a misunderstanding during a task; note it and clarify afterwards.
- Screen-record the session if the participant agrees.

## Setup
Open the app (`streamlit run app/app.py`). Start at screen 1. Choose the squad need **N-02 · MID · Summer 2026** in the sidebar. Say: "This tool helps a club's Head of Recruitment decide whether to bid for a player and up to what price. All data is made up."

## Tasks
| # | Task (read aloud) | What to look for |
|---|---|---|
| 1 | Open screen **5 What-if**, tab *KPI 1*, with the pre-selected shortlisted player. "In your own words, what is this forecast telling you?" | Do they separate the central forecast, the 80% range and "chance the fee holds"? Do they confuse the chance with a guarantee? |
| 2 | "Which factors are driving this prediction? How do you know?" | Do they find the contribution bars? Do they confuse *this player's* contributions with the *overall* importance on screen 4? |
| 3 | "Which of these inputs could you actually change as a recruiter, and which could you not?" | Do they find the *You decide / Fixed / External* tags? Do they understand that the asking price is the only lever? |
| 4 | "Create a scenario where the asking price is 20% higher. Then a second, different plausible scenario of your choice. Compare them with the baseline." | Can they use the side-by-side table and the *Pin* button? Do they read the *Change from baseline* column correctly? |
| 5 | "Try to enter something unrealistic, for example a 16-year-old with a valuation from last year, or an asking price of 100 times the value. What happens? Is it clear what to do?" | Do they understand the *blocked / Do not rely / Use with caution* messages? |
| 6 | "Read the automatic summary. Is there anything in it you would not repeat to your director? Does it claim anything it should not?" | Do they notice the *not proven effects / not guaranteed* wording? |
| 7 | Switch to the *KPI 2* tab. "The forecast is below the target style. What would you do?" | Do they treat it as a rejection? (The tool warns that only 23% of real fits are found.) |
| 8 | Open **6 Validation and limits**. "How far would you trust these forecasts? For which players would you not?" | Can they name the older-player and high-passer weaknesses? |
| 9 | "What decision, if any, could these results inform? What would you still want to know that the tool does not show?" | Open answer: record verbatim. |

## Post-test questions (1 to 5, with a comment)
1. The tool was easy to understand. 2. I understood how far to trust the forecasts. 3. I understood what I can and cannot change. 4. The summary was useful and honest about its limits. 5. I would use something like this to support a bid decision (and what for?).
Also ask: "What was the most confusing moment?" and "What is missing?"
