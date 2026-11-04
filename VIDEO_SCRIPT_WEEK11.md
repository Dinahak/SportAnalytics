# Demo Video Outline, Week 11 (7 minutes)

Run the app first (`streamlit run app/app.py`), choose squad need **N-02 · MID · Summer 2026**, and keep `docs/WEEK11_EVALUATION_REPORT.md` open for numbers.

| Time | Screen | Show and say |
|---|---|---|
| 0:00 to 0:30 | Candidates | Decision-maker (Head of Recruitment), the decision (bid or not, up to what price), the two KPIs. Data is synthetic. |
| 0:30 to 2:00 | **6 Validation and limits**: "How it was tested" and "Against a simple baseline" | Time-ordered split, tested once on 2025. Leakage checks (14 pass, 1 finding). KPI 1: 12% typical error vs 29% and 23% for baselines; the 75%-sure price held for 82%. KPI 2: 9.3 vs 12.3 passes. Say the 2025 test is not pristine. |
| 2:00 to 3:00 | **6 Validation**: "Where it fails" | Players older than the training ages (all forecast too low); defenders and ages 26 to 29 in 2025; the model finds only 23% of real target-style fits. What a recruiter should do about each. |
| 3:00 to 3:45 | **6 Validation**: "What we improved" | Two adopted changes (widened range outside trained ages; separate goalkeeper range, show the chart) and the rejected ones. An unsuccessful attempt is reported, not hidden. |
| 3:45 to 4:15 | 5 What-if, KPI 1 | The trust strip, the *You decide / Fixed / External* panel: the club controls the price, not the player's age. |
| 4:15 to 5:45 | 5 What-if, KPI 1: **complete comparison** | Baseline (a shortlisted player). Scenario A: asking price +20%, press **Pin**. Scenario B: a different plausible change. Read the side-by-side table (inputs, forecast, range, chance the fee holds, bid reading, differences). Read the automatic summary: trade-offs, limitations, "not proven effects". Explain the weight of the contribution bars vs overall importance (screen 4). |
| 5:45 to 6:30 | 5 What-if, invalid and implausible inputs | Age 16.5 with a valuation from last year (blocked, why). Age 35 (Do not rely, widened range). Asking price 3 times value (Use with caution). |
| 6:30 to 7:00 | KPI 2 tab, then wrap-up | Forecast below target is a warning, not a rejection. User-test finding and the change it caused: **[fill in after the test]**. Remaining limitations and next step (score the June 2026 forecasts in January 2027). |
