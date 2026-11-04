# User Test Results (Week 11, part 4)

Protocol: `docs/USER_TEST_PROTOCOL.md`. Only synthetic data was shown; no confidential or restricted data was used.

## Participant
- **Background:** follows football regularly and understands player positions, transfers and transfer fees.
- **Intended decision-maker?** No, a **substitute**. The participant has no professional recruitment experience, so the feedback represents a football fan's perspective, not a Head of Recruitment's. A real recruiter would bring budget pressure, accountability for a bid and knowledge of the squad, which this test could not capture. Feedback on wording and usability is still informative; feedback on the decision value is weaker evidence.

## Tasks attempted
The participant explored the forecasts, the prediction drivers and the input labels; increased the asking price by 20%; created a second scenario; compared both with the baseline; tested unrealistic inputs; and reviewed the Validation and limits screen.

## Tasks completed independently
- Moved between the screens and selected the shortlisted player.
- Changed the asking price.
- Identified the predicted market value (interpreting the baseline prediction).
- Found the warning that appears when unrealistic values are entered.

## Difficulties and misunderstandings
| Where | What happened |
|---|---|
| Screen 5, KPI 1, asking price | Unsure why raising the asking price changed **"Chance fee holds"** but not the **predicted market value**. |
| Screen 5, comparing scenarios | **Hesitated at the Pin button** when comparing two scenarios with the baseline. |
| Overall | The amount of information felt **overwhelming at first**. |

## Facilitation
The facilitator read the tasks aloud, asked the participant to explain their thinking, and recorded their questions without correcting them during the tasks. Afterwards the facilitator explained that the asking price changes the financial comparison, while the market-value forecast depends on the player's predictors.

## Participant's feedback
- **Decision it could inform:** discussions about whether a player's asking price is reasonable.
- **Unclear or missing:** simpler explanations of the probabilities and of the scenario comparisons.
- **Most important suggested improvement:** a short explanation beside the asking-price input.

## Changes made because of the findings
| Finding | Change made to the app | Where |
|---|---|---|
| Did not understand why the asking price moves "Chance fee holds" but not the forecast (the participant's top suggestion) | Added the participant's suggested sentence under the asking-price input: *"Changing the asking price updates the chance that the player's future value will cover the fee. It does not change the predicted market value."* Also added a tooltip on the input. | `app/app.py`, KPI 1 what-if inputs |
| Hesitated at the Pin button | Added a one-line instruction next to the button: set up scenario A, press **Pin**, change the inputs for scenario B; the table then shows the baseline, A and B side by side. Applies to both KPI tabs. | `app/app.py`, `pin_controls` |

**Not changed, and why:** the feeling of too much information at first. Reducing it properly (for example a simplified first view that hides the detailed tables) is a larger redesign. One participant, who is not the intended user, is not enough evidence to justify it. It is listed as future work and should be retested with a recruiter.

## Limitations of this test
One participant; a substitute rather than a Head of Recruitment; the facilitator was a member of the project group. Ratings and task times were not recorded.
