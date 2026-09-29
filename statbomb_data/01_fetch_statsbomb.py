"""
Step 1. Fetch the StatsBomb open data we use as the empirical base.

Scope: the four COMPLETE 2015/16 league seasons in the open data
(Premier League, La Liga, Serie A, Ligue 1). Every match of each season is
covered, so players are observed against every opponent, not just one club.

Usage:  python 01_fetch_statsbomb.py --out ../raw
Data license: StatsBomb Public Data User Agreement (non-commercial, no
redistribution). Raw files stay local and are never shipped in the database.
"""
import argparse, json, os, concurrent.futures as cf, urllib.request

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
# (competition_id, season_id) confirmed complete: 380/380/380/377 matches, 20 teams each
SCOPE = [(2, 27), (11, 27), (12, 27), (7, 27)]


def get(url, path, retries=3):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    for a in range(retries):
        try:
            urllib.request.urlretrieve(url, path)
            return path
        except Exception:
            if a == retries - 1:
                raise


def main(out):
    for d in ("matches", "events", "lineups"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    get(f"{BASE}/competitions.json", os.path.join(out, "competitions.json"))
    ids = []
    for c, s in SCOPE:
        os.makedirs(os.path.join(out, "matches", str(c)), exist_ok=True)
        p = get(f"{BASE}/matches/{c}/{s}.json", os.path.join(out, "matches", str(c), f"{s}.json"))
        ids += [m["match_id"] for m in json.load(open(p, encoding="utf-8"))]
    jobs = [(f"{BASE}/{d}/{i}.json", os.path.join(out, d, f"{i}.json"))
            for i in ids for d in ("events", "lineups")]
    with cf.ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda j: get(*j), jobs))
    print(f"{len(ids)} matches fetched to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../raw")
    main(ap.parse_args().out)
