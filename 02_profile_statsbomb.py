"""
Step 2. Turn raw StatsBomb events into empirical calibration targets.

Outputs (in --out):
  player_match.csv        one row per player per match (minutes + counting stats)
  player_season.csv       per-player season totals and per-90 rates, primary position
  team_season.csv         league table and team-level xG, for club archetypes
  calibration.json        distributions the synthetic generator samples from

Only aggregates leave this step. Raw events are never copied into the database.
"""
import argparse, json, os, glob, math, collections
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

GOAL = (120.0, 40.0)
SET_PIECE_PASS = {"Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"}

# StatsBomb position_id -> position group used by the scouting tool
POS_GROUP = {1: "GK",
             3: "CB", 4: "CB", 5: "CB",
             2: "FB", 6: "FB", 7: "FB", 8: "FB",
             9: "DM", 10: "DM", 11: "DM",
             13: "CM", 14: "CM", 15: "CM",
             12: "WM", 16: "WM", 17: "W", 21: "W",
             18: "AM", 19: "AM", 20: "AM",
             22: "ST", 23: "ST", 24: "ST", 25: "ST"}
POS_GROUP = {k: {"WM": "W"}.get(v, v) for k, v in POS_GROUP.items()}

LEAGUE = {2: "Premier League", 11: "La Liga", 12: "Serie A", 7: "Ligue 1"}

STATS = ["passes", "passes_completed", "progressive_passes", "passes_final_third",
         "passes_into_box", "crosses", "through_balls", "key_passes", "assists", "xa",
         "passes_under_pressure", "passes_under_pressure_completed",
         "shots", "np_shots", "shots_on_target", "goals", "np_goals", "xg", "npxg",
         "carries", "progressive_carries", "carries_into_box",
         "dribbles", "dribbles_completed", "receipts_in_box",
         "pressures", "counterpresses", "pressures_final_third",
         "tackles", "tackles_won", "interceptions", "ball_recoveries",
         "clearances", "blocks", "aerials_won", "aerials_lost",
         "fouls_committed", "fouls_won", "dispossessed", "miscontrols", "errors",
         "yellow_cards", "red_cards",
         "gk_shots_on_target_faced", "gk_saves", "gk_goals_conceded", "gk_xg_on_target_faced",
         "gk_claims", "gk_long_passes", "gk_long_passes_completed"]


def clock(s):
    m, sec = s.split(":")
    return int(m) * 60 + int(sec)


def dist_to_goal(loc):
    return math.hypot(GOAL[0] - loc[0], GOAL[1] - loc[1])


def progressive(start, end):
    """Moves the ball at least 25% closer to the goal centre, ending in the opponent half-ish."""
    d0, d1 = dist_to_goal(start), dist_to_goal(end)
    return end[0] >= 60 and d1 <= 0.75 * d0


def in_box(loc):
    return loc[0] >= 102 and 18 <= loc[1] <= 62


def profile_match(args):
    ev_path, lu_path, meta = args
    events = json.load(open(ev_path, encoding="utf-8"))
    lineups = json.load(open(lu_path, encoding="utf-8"))

    # match length in match-clock seconds (exclude penalty shootout)
    end = max(e["minute"] * 60 + e["second"] for e in events if e["period"] <= 4)
    rows = {}
    for team in lineups:
        for p in team["lineup"]:
            mins, pos_mins = 0.0, collections.Counter()
            for pos in p.get("positions", []):
                a = clock(pos["from"])
                b = clock(pos["to"]) if pos.get("to") else end
                seg = max(0, b - a) / 60
                mins += seg
                pos_mins[pos["position_id"]] += seg
            if mins <= 0:
                continue
            cards = [c["card_type"] for c in p.get("cards", [])]
            r = dict(match_id=meta["match_id"], team_id=team["team_id"], player_id=p["player_id"],
                     player_name=p["player_name"], country=(p.get("country") or {}).get("name"),
                     minutes=round(mins, 2),
                     started=any(x.get("start_reason") == "Starting XI" for x in p["positions"]),
                     main_position_id=pos_mins.most_common(1)[0][0],
                     pos_minutes=json.dumps({str(k): round(v, 1) for k, v in pos_mins.items()}))
            r.update({s: 0.0 for s in STATS})
            r["yellow_cards"] = sum(c in ("Yellow Card", "Second Yellow") for c in cards)
            r["red_cards"] = sum(c in ("Red Card", "Second Yellow") for c in cards)
            rows[p["player_id"]] = r

    shot_xg = {e["id"]: e["shot"].get("statsbomb_xg", 0.0) for e in events if "shot" in e}
    team_gk = {}  # team_id -> player on pitch in GK slot is resolved via position on GK events
    for e in events:
        pid = (e.get("player") or {}).get("id")
        r = rows.get(pid)
        t = e["type"]["name"]
        if r is None:
            continue
        loc = e.get("location")
        up = e.get("under_pressure", False)
        if t == "Pass":
            ps = e["pass"]
            ok = "outcome" not in ps
            ptype = (ps.get("type") or {}).get("name")
            r["passes"] += 1
            r["passes_completed"] += ok
            if up:
                r["passes_under_pressure"] += 1
                r["passes_under_pressure_completed"] += ok
            endl = ps.get("end_location")
            if ok and endl and loc and ptype not in SET_PIECE_PASS:
                r["progressive_passes"] += progressive(loc, endl)
                r["passes_final_third"] += (endl[0] >= 80 and loc[0] < 80)
                r["passes_into_box"] += (in_box(endl) and not in_box(loc))
            r["crosses"] += bool(ps.get("cross"))
            r["through_balls"] += bool(ps.get("through_ball")) or (ps.get("technique") or {}).get("name") == "Through Ball"
            if ps.get("shot_assist") or ps.get("goal_assist"):
                r["key_passes"] += 1
                r["xa"] += shot_xg.get(ps.get("assisted_shot_id"), 0.0)
            r["assists"] += bool(ps.get("goal_assist"))
            r["aerials_won"] += bool(ps.get("aerial_won"))
            if (e.get("position") or {}).get("id") == 1 and ps.get("length", 0) >= 40:
                r["gk_long_passes"] += 1
                r["gk_long_passes_completed"] += ok
        elif t == "Shot":
            sh = e["shot"]
            pen = (sh.get("type") or {}).get("name") == "Penalty"
            xg = sh.get("statsbomb_xg", 0.0)
            out = sh["outcome"]["name"]
            goal = out == "Goal"
            r["shots"] += 1; r["xg"] += xg; r["goals"] += goal
            r["shots_on_target"] += out in ("Goal", "Saved", "Saved To Post")
            if not pen:
                r["np_shots"] += 1; r["npxg"] += xg; r["np_goals"] += goal
            r["aerials_won"] += bool(sh.get("aerial_won"))
        elif t == "Carry":
            endl = e["carry"].get("end_location")
            r["carries"] += 1
            if endl and loc:
                r["progressive_carries"] += progressive(loc, endl) and dist_to_goal(loc) - dist_to_goal(endl) >= 5
                r["carries_into_box"] += (in_box(endl) and not in_box(loc))
        elif t == "Ball Receipt*":
            if loc and in_box(loc) and "outcome" not in e.get("ball_receipt", {}):
                r["receipts_in_box"] += 1
        elif t == "Dribble":
            r["dribbles"] += 1
            r["dribbles_completed"] += e["dribble"].get("outcome", {}).get("name") == "Complete"
        elif t == "Pressure":
            r["pressures"] += 1
            r["counterpresses"] += bool(e.get("counterpress"))
            r["pressures_final_third"] += bool(loc and loc[0] >= 80)
        elif t == "Duel":
            d = e["duel"]; dt = d.get("type", {}).get("name")
            if dt == "Tackle":
                r["tackles"] += 1
                r["tackles_won"] += d.get("outcome", {}).get("name") in ("Won", "Success In Play", "Success Out")
            elif dt == "Aerial Lost":
                r["aerials_lost"] += 1
        elif t == "Interception":
            r["interceptions"] += 1
        elif t == "Ball Recovery":
            r["ball_recoveries"] += not e.get("ball_recovery", {}).get("recovery_failure", False)
        elif t == "Clearance":
            r["clearances"] += 1
            r["aerials_won"] += bool(e["clearance"].get("aerial_won"))
        elif t == "Block":
            r["blocks"] += 1
        elif t == "Foul Committed":
            r["fouls_committed"] += 1
        elif t == "Foul Won":
            r["fouls_won"] += 1
        elif t == "Dispossessed":
            r["dispossessed"] += 1
        elif t == "Miscontrol":
            r["miscontrols"] += 1
            r["aerials_won"] += bool(e.get("miscontrol", {}).get("aerial_won"))
        elif t == "Error":
            r["errors"] += 1
        elif t == "Goal Keeper":
            gt = e["goalkeeper"].get("type", {}).get("name")
            if gt in ("Shot Saved", "Shot Saved to Post", "Penalty Saved", "Shot Saved Off Target", "Save"):
                r["gk_saves"] += 1
            if gt == "Collected":
                r["gk_claims"] += 1

    # GK shots faced: attribute on-target opponent shots to the keeper in goal (position 1 at the time)
    gk_events = [e for e in events if e["type"]["name"] == "Goal Keeper" and (e.get("player") or {}).get("id") in rows]
    faced = {}
    for e in gk_events:
        for rid in e.get("related_events", []):
            faced[rid] = e["player"]["id"]
    for e in events:
        if e["type"]["name"] == "Shot" and e["period"] <= 4:
            out = e["shot"]["outcome"]["name"]
            if out in ("Goal", "Saved", "Saved To Post"):
                gk = faced.get(e["id"])
                if gk in rows:
                    rows[gk]["gk_shots_on_target_faced"] += 1
                    rows[gk]["gk_xg_on_target_faced"] += e["shot"].get("statsbomb_xg", 0.0)
                    rows[gk]["gk_goals_conceded"] += out == "Goal"

    # team-level
    team_rows = {}
    for tm in lineups:
        tid = tm["team_id"]
        team_rows[tid] = dict(match_id=meta["match_id"], team_id=tid, team_name=tm["team_name"],
                              xg=0.0, shots=0, passes=0)
    for e in events:
        tid = (e.get("team") or {}).get("id")
        if tid in team_rows:
            if e["type"]["name"] == "Shot" and e["period"] <= 4:
                team_rows[tid]["xg"] += e["shot"].get("statsbomb_xg", 0.0); team_rows[tid]["shots"] += 1
            elif e["type"]["name"] == "Pass":
                team_rows[tid]["passes"] += 1
    return list(rows.values()), list(team_rows.values())


def main(raw, out, min_minutes):
    os.makedirs(out, exist_ok=True)
    matches, jobs = [], []
    for p in glob.glob(os.path.join(raw, "matches", "*", "27.json")):
        for m in json.load(open(p, encoding="utf-8")):
            cid = m["competition"]["competition_id"]
            if cid not in LEAGUE:
                continue
            meta = dict(match_id=m["match_id"], league=LEAGUE[cid], date=m["match_date"],
                        week=m["match_week"],
                        home_id=m["home_team"]["home_team_id"], home=m["home_team"]["home_team_name"],
                        away_id=m["away_team"]["away_team_id"], away=m["away_team"]["away_team_name"],
                        hs=m["home_score"], as_=m["away_score"])
            matches.append(meta)
            jobs.append((os.path.join(raw, "events", f"{m['match_id']}.json"),
                         os.path.join(raw, "lineups", f"{m['match_id']}.json"), meta))
    pm, tm = [], []
    with ProcessPoolExecutor() as ex:
        for a, b in ex.map(profile_match, jobs, chunksize=8):
            pm += a; tm += b
    mdf = pd.DataFrame(matches)
    pm = pd.DataFrame(pm).merge(mdf[["match_id", "league", "date"]], on="match_id")
    pm["position_group"] = pm["main_position_id"].map(POS_GROUP)
    pm.to_csv(os.path.join(out, "player_match.csv"), index=False)

    # ---- team season table
    tm = pd.DataFrame(tm)
    tbl = []
    for m in matches:
        for side, opp, gf, ga in (("home_id", "away_id", m["hs"], m["as_"]), ("away_id", "home_id", m["as_"], m["hs"])):
            tbl.append(dict(league=m["league"], team_id=m[side], gf=gf, ga=ga,
                            pts=3 if gf > ga else 1 if gf == ga else 0, match_id=m["match_id"], opp_id=m[opp]))
    tbl = pd.DataFrame(tbl)
    x = tm.merge(tm[["match_id", "team_id", "xg"]].rename(columns={"team_id": "opp_id", "xg": "xga"}), on="match_id")
    x = x[x.team_id != x.opp_id]
    # team names vary across matches for the same team_id (e.g. "Caen" vs "Stade Malherbe Caen"): use the mode
    canon = tm.groupby("team_id")["team_name"].agg(lambda n: n.value_counts().index[0])
    tbl = tbl.merge(x[["match_id", "team_id", "xg", "xga"]], on=["match_id", "team_id"])
    tbl["team_name"] = tbl.team_id.map(canon)
    ts = tbl.groupby(["league", "team_id", "team_name"]).agg(
        played=("pts", "size"), pts=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"),
        xg=("xg", "sum"), xga=("xga", "sum")).reset_index()
    ts["rank"] = ts.groupby("league")["pts"].rank(ascending=False, method="first").astype(int)
    ts = ts.sort_values(["league", "rank"])
    ts.to_csv(os.path.join(out, "team_season.csv"), index=False)

    # ---- player season (per player per team)
    agg = pm.groupby(["player_id", "player_name", "country", "team_id", "league"], dropna=False).agg(
        apps=("match_id", "nunique"), starts=("started", "sum"), minutes=("minutes", "sum"),
        **{s: (s, "sum") for s in STATS}).reset_index()
    posmin = pm.groupby(["player_id", "team_id", "position_group"])["minutes"].sum().reset_index()
    prim = posmin.sort_values("minutes").groupby(["player_id", "team_id"]).tail(1)[["player_id", "team_id", "position_group"]]
    ps = agg.merge(prim, on=["player_id", "team_id"])
    p90 = pd.DataFrame({f"{s}_p90": np.where(ps.minutes > 0, ps[s] / ps.minutes * 90, np.nan) for s in STATS})
    ps = pd.concat([ps, p90], axis=1).copy()
    ps["pass_pct"] = ps.passes_completed / ps.passes.replace(0, np.nan)
    ps["dribble_pct"] = ps.dribbles_completed / ps.dribbles.replace(0, np.nan)
    ps["aerial_pct"] = ps.aerials_won / (ps.aerials_won + ps.aerials_lost).replace(0, np.nan)
    ps["gk_save_pct"] = ps.gk_saves / ps.gk_shots_on_target_faced.replace(0, np.nan)
    ps["gk_goals_prevented"] = ps.gk_xg_on_target_faced - ps.gk_goals_conceded
    ps.to_csv(os.path.join(out, "player_season.csv"), index=False)

    # ---- calibration targets
    q = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    rates = [f"{s}_p90" for s in STATS] + ["pass_pct", "dribble_pct", "aerial_pct", "gk_save_pct"]
    reg = ps[ps.minutes >= min_minutes]
    cal = dict(source="StatsBomb Open Data, 2015/16 EPL, La Liga, Serie A, Ligue 1 (complete seasons)",
               n_matches=len(matches), n_player_seasons=int(len(ps)), min_minutes_for_rates=min_minutes,
               position_groups={}, league_effects={}, squad={}, nationality={})
    for g, d in reg.groupby("position_group"):
        cal["position_groups"][g] = dict(
            n=int(len(d)),
            quantiles={r: [round(float(v), 4) for v in d[r].quantile(q)] for r in rates if d[r].notna().sum() > 20},
            mean={r: round(float(d[r].mean()), 4) for r in rates if d[r].notna().sum() > 20},
            sd={r: round(float(d[r].std()), 4) for r in rates if d[r].notna().sum() > 20},
        )
        core = [r for r in ["npxg_p90", "xa_p90", "progressive_passes_p90", "progressive_carries_p90",
                            "key_passes_p90", "dribbles_completed_p90", "pressures_p90", "tackles_won_p90",
                            "interceptions_p90", "aerials_won_p90", "pass_pct"] if d[r].notna().all()]
        cal["position_groups"][g]["spearman_core"] = {"metrics": core,
            "matrix": d[core].corr(method="spearman").round(3).values.tolist()}
    cal["quantile_levels"] = q
    for lg, d in reg.groupby("league"):
        cal["league_effects"][lg] = {r: round(float(d[r].median()), 4) for r in
                                     ["npxg_p90", "progressive_passes_p90", "pressures_p90", "pass_pct"]}
    per_team = ps.groupby(["league", "team_id"]).agg(players=("player_id", "nunique"),
                                                      used_900=("minutes", lambda m: int((m >= 900).sum())))
    cal["squad"] = dict(players_used=per_team.players.describe().round(2).to_dict(),
                        players_900_plus=per_team.used_900.describe().round(2).to_dict(),
                        minutes_share_by_group=(ps.groupby("position_group").minutes.sum() /
                                                ps.minutes.sum()).round(4).to_dict(),
                        season_minutes_quantiles=[round(float(v), 1) for v in ps.minutes.quantile(q)])
    for lg, d in ps.groupby("league"):
        share = d.groupby("country").minutes.sum().sort_values(ascending=False) / d.minutes.sum()
        cal["nationality"][lg] = share.head(15).round(4).to_dict()
    cal["team_strength"] = ts[["league", "team_name", "rank", "pts", "xg", "xga"]].round(2).to_dict("records")
    json.dump(cal, open(os.path.join(out, "calibration.json"), "w"), indent=1)
    print(f"matches={len(matches)} player_match={len(pm)} player_season={len(ps)} regulars={len(reg)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../raw")
    ap.add_argument("--out", default="../calibration")
    ap.add_argument("--min-minutes", type=int, default=900)
    a = ap.parse_args()
    main(a.raw, a.out, a.min_minutes)
