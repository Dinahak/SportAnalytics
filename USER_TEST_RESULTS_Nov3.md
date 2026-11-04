# User Test Results (Week 11, part 4)

> Filled from Tarun's Nov 3, 2026 session notes. Items marked [NEED] still need Tarun/the group to confirm. Nothing here is invented.

## Participant
- Role / background: Friend of Tarun's. Watches football and knows the basics of football recruitment.
- Familiar with football recruitment? Partly (basics).
- **Is this the intended decision-maker?** No — substitute. Limitation: does not carry a real budget or accountability for a bid.
- Date, duration, setting: Nov 3, 2026; 10–15 minutes (shorter than the protocol's 20–30 minute estimate); remote.
- Recorded? [NEED: confirm whether the session was recorded]
- No confidential or restricted data was shown (synthetic data only): Confirmed — the app shows synthetic data only.

## Task log
| # | Task | Completed? | Time | What the participant said / did (quotes) | Errors or misunderstandings |
|---|---|---|---|---|---|
| 1 | Interpret the baseline prediction | Yes | [not recorded] | "The market value of the player will drop by 14% in the month of January 2027." | Whether they separated the central forecast, the 80% range and "chance the fee holds" was not recorded. |
| 2 | Main factors behind the prediction | Yes | [not recorded] | "I think it's the age that's making this prediction because I can see that from the line graph." | Tester cited a line graph, not the contribution bars. |
| 3 | What can / cannot be changed | Yes | [not recorded] | "I think the input asking price is something I would change and yes, I think [that's] the only input that I can change." | None — correct: asking price is the only lever. |
| 4 | Two what-if scenarios compared | Yes | [not recorded] | Tester made both scenarios and used the Pin button, and said "yes, I do read it right" for the side-by-side comparison. | None recorded. |
| 5 | Implausible input | No — not performed by the tester | [not recorded] | No tester response. The facilitator did not know how to perform this step during the session. | Facilitator gap, not a tester finding. See note below. |
| 6 | Read the automatic summary | Attempted | [not recorded] | "I really don't know." | Tester could not answer. |
| 7 | KPI 2, forecast below target | Yes | [not recorded] | "Yes, I would treat it as an automatic rejection because the chance of reaching the target is only 31% and I would at least expect 50%." | Treats a below-target forecast as an automatic rejection, despite the tool's warning that most real fits are missed. |
| 8 | Validation and limits | No | [not recorded] | Tester did not understand the question. | Question not understood; no answer recorded. |
| 9 | Decision it could inform; what is missing | Yes | [not recorded] | "I understand you guys are beginners and doing a project like this for the very first time but I think the only KPI you've used [is] passes per 90 minutes but I would expect more if you guys were more experienced like expected goals, most chances created, shots on target etc." | None — this is the tester's own view. |

## Ratings (1 to 5)
- Q1 The tool was easy to understand: **3**
- Q2 I understood how far to trust the forecasts: **3**
- Q3 I understood what I can and cannot change: **5**
- Q4 The summary was useful and honest about its limits: **4**
- Q5 I would use something like this to support a bid decision: **4**
- Most confusing moment: **The what-if analysis page.**
- What is missing: **More KPIs** (expected goals, chances created, shots on target — see task 9).

## Findings
- **Interpretation of the baseline:** Tester read the baseline as a 14% drop in market value by January 2027.
- **Decision the results could inform (their words):** Not directly stated; tester would use the tool but would auto-reject on a below-target KPI 2 forecast.
- **Information that remains unclear or missing:** More KPIs than passes per 90 (expected goals, chances created, shots on target); the automatic summary (task 6 drew "I really don't know"); the validation-and-limits question itself was not understood.
- **Misunderstandings (each one, with the screen it happened on):**
  - What-if screen, KPI 2 tab: below-target forecast treated as an automatic rejection (31% vs a 50% personal threshold).
  - What-if screen: tester attributed the prediction to age "from the line graph" rather than the contribution bars.
  - What-if screen: named as the most confusing moment of the session.

## Changes made because of the findings
At least one is required, or an explanation of why none was warranted. **None made yet — the group must pick and implement one.**

| Finding | Change made to the app | File / commit |
|---|---|---|
| What-if page was the most confusing moment | [NEED: group to decide and implement, e.g. a short on-page explainer for the what-if screen] | [NEED] |
| KPI 2 below-target read as automatic rejection | [NEED: group to decide and implement, e.g. an explicit note on the KPI 2 tab that below-target is not an automatic rejection] | [NEED] |

## Note on task 5 (facilitator check, NOT a tester finding)
After the session, the facilitator checked the step himself: setting age to 16.5 with "Player has a valuation from one year ago" switched on gives an "inputs are impossible or contradict each other" message, asking the user either to switch off the valuation-from-one-year-ago option or to raise the age to 17 or more. This confirms the block works, but it is not a tester result and must not be presented as one. If the tester is still available, task 5 can still be run with them in about 2 minutes.

## Changes made *before* testing (our own walk-through, not user findings)
Listed so the two are not confused: input checks and "Do not rely" labels for minutes below 900; the range row of the comparison table shows both ends; the "which inputs can I change" panel. These came from our own scenario testing, not from a participant.
