# Setting up and checking Week 11 on your laptop

Commands are for Windows (Command Prompt). On Mac/Linux use `python3` and `source .venv/bin/activate`.

## A. Check that it works (about 15 minutes)
1. Install **Python 3.10, 3.11 or 3.12** from python.org (tick "Add Python to PATH").
2. Unzip `SportAnalytics-week11.zip`, then:
   ```
   cd SportAnalytics-main\Predictive_Model
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Run the tests. Expected: `Ran 14 tests ... OK`.
   ```
   python -m unittest discover tests -v
   ```
4. Start the app (it opens at http://localhost:8501):
   ```
   streamlit run app/app.py
   ```
5. Click through and confirm:
   - Screen **5 What-if**, KPI 1 tab: the blue "How much to trust this" box; the "Which inputs can I change?" panel; the "Baseline and scenario side by side" table.
   - Raise the asking price: the table's "Change from baseline" column and the summary update.
   - Click **Pin this scenario to compare**, change another input: a "Pinned" column appears.
   - Set age to 16.5 with "valuation from one year ago" switched on: red "cannot be forecast" message.
   - Set age to 35: "Do not rely" message.
   - KPI 2 tab: the same table; set minutes below 900: "Do not rely".
   - Screen **6 Validation and limits**: all five tabs open, the bar chart shows.
6. Optional: re-run the Week 11 evaluation (about 1.5 minutes): `python code\11_evaluate_models.py`.

If anything shows an error, copy the red text (or the terminal output) and send it back.

## B. Put it on GitHub (the repo owner does this)
Run these inside the existing cloned repo, **before** copying the new files in:
```
git pull
git tag -a week10-submission -m "Week 10 submission"
git push origin week10-submission
```
Then copy the contents of `SportAnalytics-main` from the zip over the repo folder (replace files), and:
```
git add -A
git commit -m "Week 11: evaluation, model improvements, input checks, side-by-side what-if, validation screen, report"
git push
```
Do not commit the `.venv` folder (it is already in `.gitignore`).

## C. Still to do by people (cannot be prepared in advance)
1. User test with someone outside the group: `docs/USER_TEST_PROTOCOL.md` → fill `docs/USER_TEST_RESULTS.md`, make at least one app change, update report section 7.
2. Record the 7-minute video: `docs/VIDEO_SCRIPT_WEEK11.md`. Upload (Google Drive / OneDrive / YouTube unlisted), set sharing to "anyone with the link", paste the link in `README.md`.
3. Fill in the Week 11 contributions in `docs/CONTRIBUTIONS.md`.
4. Make sure the instructor and all group members can open the repo, then submit the link.
