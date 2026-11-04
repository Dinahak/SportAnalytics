-- =====================================================================
-- Synthetic Recruitment Database  (schema draft v0.1)
-- Organization archetype : mid-table club in a top European league
-- Decision role          : Head of Recruitment
-- Core decision          : For each squad need in a transfer window,
--                          which player do we pursue, and at what
--                          maximum fee and wage, or do we not sign?
--
-- Portable SQL: runs on SQLite 3.35+ and PostgreSQL 13+.
-- IDs are assigned by the generator (deterministic from the seed).
-- Performance distributions are calibrated to StatsBomb Open Data
-- (2015/16 EPL, La Liga, Serie A, Ligue 1). No raw StatsBomb rows are stored.
-- =====================================================================

-- ---------------------------------------------------------------
-- A. REFERENCE
-- ---------------------------------------------------------------
CREATE TABLE country (
    country_id      INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    eu_member       BOOLEAN NOT NULL,          -- drives non-EU quota constraints
    confederation   TEXT NOT NULL
);

CREATE TABLE league (
    league_id       INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    country_id      INTEGER NOT NULL REFERENCES country(country_id),
    tier            INTEGER NOT NULL CHECK (tier BETWEEN 1 AND 3),
    strength_coef   REAL NOT NULL CHECK (strength_coef > 0),   -- translates per-90 output across leagues
    n_teams         INTEGER NOT NULL,
    calibrated_from TEXT                        -- e.g. 'StatsBomb 2015/16 Serie A', or NULL if extrapolated
);

CREATE TABLE season (
    season_id       INTEGER PRIMARY KEY,
    label           TEXT NOT NULL UNIQUE,       -- fictional seasons, e.g. 'S1', 'S2'
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    CHECK (end_date > start_date)
);

CREATE TABLE position_group (
    position_group  TEXT PRIMARY KEY,           -- GK, CB, FB, DM, CM, AM, W, ST
    description     TEXT NOT NULL
);

CREATE TABLE metric (
    metric_code     TEXT PRIMARY KEY,           -- e.g. npxg_p90, progressive_passes_p90
    label           TEXT NOT NULL,
    family          TEXT NOT NULL CHECK (family IN ('attacking','creation','progression','defending','aerial','goalkeeping','discipline','volume')),
    higher_is_better BOOLEAN NOT NULL,
    statsbomb_definition TEXT NOT NULL          -- how the calibration step derives it from events
);

-- ---------------------------------------------------------------
-- B. MARKET ENTITIES (clubs, players, agents, contracts)
-- ---------------------------------------------------------------
CREATE TABLE club (
    club_id         INTEGER PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,       -- fictional names
    league_id       INTEGER NOT NULL REFERENCES league(league_id),
    archetype       TEXT NOT NULL CHECK (archetype IN ('elite','contender','mid_table','relegation','promoted')),
    selling_propensity REAL NOT NULL CHECK (selling_propensity BETWEEN 0 AND 1),
    is_focal_club   BOOLEAN NOT NULL DEFAULT FALSE   -- exactly one TRUE: the decision-maker's club
);

CREATE TABLE agent (
    agent_id        INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    agency          TEXT,
    fee_pct_typical REAL NOT NULL CHECK (fee_pct_typical BETWEEN 0 AND 0.2),
    cooperativeness REAL NOT NULL CHECK (cooperativeness BETWEEN 0 AND 1)
);

CREATE TABLE player (
    player_id       INTEGER PRIMARY KEY,
    display_name    TEXT NOT NULL,              -- synthetic names only
    birth_date      DATE NOT NULL,
    nationality_id  INTEGER NOT NULL REFERENCES country(country_id),
    second_nationality_id INTEGER REFERENCES country(country_id),
    preferred_foot  TEXT NOT NULL CHECK (preferred_foot IN ('left','right','both')),
    height_cm       INTEGER CHECK (height_cm BETWEEN 155 AND 205),
    primary_position_group TEXT NOT NULL REFERENCES position_group(position_group),
    secondary_position_group TEXT REFERENCES position_group(position_group),
    agent_id        INTEGER REFERENCES agent(agent_id),
    latent_ability  REAL NOT NULL,              -- generator ground truth; HIDDEN from the tool, used only for outcome simulation and validation
    latent_potential REAL NOT NULL
);

CREATE TABLE contract (
    contract_id     INTEGER PRIMARY KEY,
    player_id       INTEGER NOT NULL REFERENCES player(player_id),
    club_id         INTEGER NOT NULL REFERENCES club(club_id),
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    weekly_wage_eur NUMERIC(12,2) NOT NULL CHECK (weekly_wage_eur > 0),
    release_clause_eur NUMERIC(14,2),
    contract_type   TEXT NOT NULL CHECK (contract_type IN ('permanent','loan_in','youth')),
    CHECK (end_date > start_date)
);

CREATE TABLE market_value_snapshot (
    player_id       INTEGER NOT NULL REFERENCES player(player_id),
    as_of_date      DATE NOT NULL,
    market_value_eur NUMERIC(14,2) NOT NULL CHECK (market_value_eur >= 0),   -- external "crowd" estimate, noisy
    source          TEXT NOT NULL CHECK (source IN ('public_estimate','internal_model')),
    PRIMARY KEY (player_id, as_of_date, source)
);

CREATE TABLE injury (
    injury_id       INTEGER PRIMARY KEY,
    player_id       INTEGER NOT NULL REFERENCES player(player_id),
    start_date      DATE NOT NULL,
    days_out        INTEGER NOT NULL CHECK (days_out > 0),
    body_area       TEXT NOT NULL,
    severity        TEXT NOT NULL CHECK (severity IN ('minor','moderate','major'))
);

-- ---------------------------------------------------------------
-- C. PERFORMANCE (synthetic, StatsBomb-calibrated)
-- ---------------------------------------------------------------
CREATE TABLE match (
    match_id        INTEGER PRIMARY KEY,
    season_id       INTEGER NOT NULL REFERENCES season(season_id),
    league_id       INTEGER NOT NULL REFERENCES league(league_id),
    match_week      INTEGER NOT NULL,
    match_date      DATE NOT NULL,
    home_club_id    INTEGER NOT NULL REFERENCES club(club_id),
    away_club_id    INTEGER NOT NULL REFERENCES club(club_id),
    home_goals      INTEGER NOT NULL CHECK (home_goals >= 0),
    away_goals      INTEGER NOT NULL CHECK (away_goals >= 0),
    home_xg         REAL NOT NULL,
    away_xg         REAL NOT NULL,
    CHECK (home_club_id <> away_club_id)
);

CREATE TABLE player_match_stats (
    match_id        INTEGER NOT NULL REFERENCES match(match_id),
    player_id       INTEGER NOT NULL REFERENCES player(player_id),
    club_id         INTEGER NOT NULL REFERENCES club(club_id),
    position_group  TEXT NOT NULL REFERENCES position_group(position_group),
    started         BOOLEAN NOT NULL,
    minutes         REAL NOT NULL CHECK (minutes > 0 AND minutes <= 130),
    goals INTEGER NOT NULL DEFAULT 0, np_goals INTEGER NOT NULL DEFAULT 0, assists INTEGER NOT NULL DEFAULT 0,
    shots INTEGER NOT NULL DEFAULT 0, xg REAL NOT NULL DEFAULT 0, npxg REAL NOT NULL DEFAULT 0, xa REAL NOT NULL DEFAULT 0,
    key_passes INTEGER NOT NULL DEFAULT 0,
    passes INTEGER NOT NULL DEFAULT 0, passes_completed INTEGER NOT NULL DEFAULT 0,
    progressive_passes INTEGER NOT NULL DEFAULT 0, passes_into_box INTEGER NOT NULL DEFAULT 0,
    progressive_carries INTEGER NOT NULL DEFAULT 0,
    dribbles INTEGER NOT NULL DEFAULT 0, dribbles_completed INTEGER NOT NULL DEFAULT 0,
    pressures INTEGER NOT NULL DEFAULT 0, tackles_won INTEGER NOT NULL DEFAULT 0,
    interceptions INTEGER NOT NULL DEFAULT 0, ball_recoveries INTEGER NOT NULL DEFAULT 0,
    aerials_won INTEGER NOT NULL DEFAULT 0, aerials_lost INTEGER NOT NULL DEFAULT 0,
    fouls_committed INTEGER NOT NULL DEFAULT 0, yellow_cards INTEGER NOT NULL DEFAULT 0, red_cards INTEGER NOT NULL DEFAULT 0,
    gk_shots_on_target_faced INTEGER NOT NULL DEFAULT 0, gk_saves INTEGER NOT NULL DEFAULT 0,
    gk_xg_on_target_faced REAL NOT NULL DEFAULT 0, gk_goals_conceded INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (match_id, player_id),
    CHECK (passes_completed <= passes),
    CHECK (dribbles_completed <= dribbles),
    CHECK (np_goals <= goals AND goals <= shots),
    CHECK (npxg <= xg),
    CHECK (gk_saves <= gk_shots_on_target_faced)
);

-- ---------------------------------------------------------------
-- D. ORGANIZATION (the focal club and its people)
-- ---------------------------------------------------------------
CREATE TABLE staff (
    staff_id        INTEGER PRIMARY KEY,
    club_id         INTEGER NOT NULL REFERENCES club(club_id),
    name            TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('head_of_recruitment','sporting_director','head_coach',
                                                  'chief_scout','scout','data_analyst','cfo','medical_lead')),
    region_focus    TEXT,                        -- scouts only
    rating_bias     REAL NOT NULL DEFAULT 0,     -- systematic over/under-rating, hidden ground truth
    rating_noise_sd REAL NOT NULL DEFAULT 0      -- scout reliability, hidden ground truth
);

CREATE TABLE transfer_window (
    window_id       INTEGER PRIMARY KEY,
    club_id         INTEGER NOT NULL REFERENCES club(club_id),
    season_id       INTEGER NOT NULL REFERENCES season(season_id),
    window_type     TEXT NOT NULL CHECK (window_type IN ('summer','winter')),
    opens_on        DATE NOT NULL,
    closes_on       DATE NOT NULL,
    transfer_budget_eur NUMERIC(14,2) NOT NULL CHECK (transfer_budget_eur >= 0),
    wage_budget_weekly_eur NUMERIC(12,2) NOT NULL CHECK (wage_budget_weekly_eur >= 0),
    max_squad_size  INTEGER NOT NULL DEFAULT 25,
    max_non_eu      INTEGER,                     -- registration constraint
    min_homegrown   INTEGER,
    CHECK (closes_on > opens_on)
);

-- ---------------------------------------------------------------
-- E. DECISION WORKFLOW
--    need -> screen -> scout -> shortlist -> value -> decide -> negotiate -> transfer
-- ---------------------------------------------------------------
CREATE TABLE squad_need (                       -- the CASE the decision-maker resolves
    need_id         INTEGER PRIMARY KEY,
    window_id       INTEGER NOT NULL REFERENCES transfer_window(window_id),
    position_group  TEXT NOT NULL REFERENCES position_group(position_group),
    trigger_reason  TEXT NOT NULL CHECK (trigger_reason IN ('departure','contract_expiry','injury','performance_gap','depth','coach_request')),
    priority        INTEGER NOT NULL CHECK (priority BETWEEN 1 AND 5),
    requested_by    INTEGER NOT NULL REFERENCES staff(staff_id),
    role_profile    TEXT NOT NULL,              -- e.g. 'ball-playing CB', 'pressing winger'
    age_min         INTEGER NOT NULL, age_max INTEGER NOT NULL,
    max_fee_eur     NUMERIC(14,2) NOT NULL,
    max_weekly_wage_eur NUMERIC(12,2) NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('open','filled','deferred','cancelled')),
    CHECK (age_max >= age_min)
);

CREATE TABLE need_metric_weight (               -- what "good" means for this need
    need_id         INTEGER NOT NULL REFERENCES squad_need(need_id),
    metric_code     TEXT NOT NULL REFERENCES metric(metric_code),
    weight          REAL NOT NULL CHECK (weight > 0),
    min_percentile  REAL CHECK (min_percentile BETWEEN 0 AND 100),
    PRIMARY KEY (need_id, metric_code)
);

CREATE TABLE screen_run (                       -- automated data screen (the tool)
    screen_run_id   INTEGER PRIMARY KEY,
    need_id         INTEGER NOT NULL REFERENCES squad_need(need_id),
    run_at          TIMESTAMP NOT NULL,
    model_version   TEXT NOT NULL,
    min_minutes     INTEGER NOT NULL,
    leagues_included TEXT NOT NULL              -- comma-separated league_ids
);

CREATE TABLE screen_result (
    screen_run_id   INTEGER NOT NULL REFERENCES screen_run(screen_run_id),
    player_id       INTEGER NOT NULL REFERENCES player(player_id),
    fit_score       REAL NOT NULL CHECK (fit_score BETWEEN 0 AND 100),
    rank_in_run     INTEGER NOT NULL,
    minutes_observed REAL NOT NULL,              -- information quality
    passed_filters  BOOLEAN NOT NULL,
    PRIMARY KEY (screen_run_id, player_id)
);

CREATE TABLE scouting_assignment (
    assignment_id   INTEGER PRIMARY KEY,
    need_id         INTEGER NOT NULL REFERENCES squad_need(need_id),
    player_id       INTEGER NOT NULL REFERENCES player(player_id),
    scout_id        INTEGER NOT NULL REFERENCES staff(staff_id),
    assigned_on     DATE NOT NULL,
    due_on          DATE NOT NULL
);

CREATE TABLE scout_report (
    report_id       INTEGER PRIMARY KEY,
    assignment_id   INTEGER NOT NULL REFERENCES scouting_assignment(assignment_id),
    match_id        INTEGER REFERENCES match(match_id),    -- NULL for video-only reports
    method          TEXT NOT NULL CHECK (method IN ('live','video','data_only')),
    submitted_on    DATE NOT NULL,
    technical_rating INTEGER NOT NULL CHECK (technical_rating BETWEEN 1 AND 10),
    physical_rating  INTEGER NOT NULL CHECK (physical_rating BETWEEN 1 AND 10),
    tactical_rating  INTEGER NOT NULL CHECK (tactical_rating BETWEEN 1 AND 10),
    mental_rating    INTEGER NOT NULL CHECK (mental_rating BETWEEN 1 AND 10),
    overall_grade    TEXT NOT NULL CHECK (overall_grade IN ('A','B','C','D')),
    recommendation   TEXT NOT NULL CHECK (recommendation IN ('sign','monitor','reject')),
    confidence       REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    notes            TEXT
);

CREATE TABLE shortlist_entry (                  -- the ALTERNATIVES
    need_id         INTEGER NOT NULL REFERENCES squad_need(need_id),
    player_id       INTEGER NOT NULL REFERENCES player(player_id),
    added_on        DATE NOT NULL,
    source          TEXT NOT NULL CHECK (source IN ('data_screen','scout_network','agent_offer','coach_request')),
    stage           TEXT NOT NULL CHECK (stage IN ('long_list','short_list','priority','dropped','signed')),
    drop_reason     TEXT,
    PRIMARY KEY (need_id, player_id)
);

CREATE TABLE valuation (                        -- model output the decision-maker sees
    valuation_id    INTEGER PRIMARY KEY,
    need_id         INTEGER NOT NULL REFERENCES squad_need(need_id),
    player_id       INTEGER NOT NULL REFERENCES player(player_id),
    valued_on       DATE NOT NULL,
    model_version   TEXT NOT NULL,
    fair_fee_eur    NUMERIC(14,2) NOT NULL,
    fee_low_eur     NUMERIC(14,2) NOT NULL,
    fee_high_eur    NUMERIC(14,2) NOT NULL,
    expected_wage_weekly_eur NUMERIC(12,2) NOT NULL,
    total_cost_of_ownership_eur NUMERIC(14,2) NOT NULL,   -- fee + wages over contract + agent fee
    expected_resale_eur NUMERIC(14,2),
    risk_score      REAL NOT NULL CHECK (risk_score BETWEEN 0 AND 1),   -- injury, adaptation, data sparsity
    CHECK (fee_low_eur <= fair_fee_eur AND fair_fee_eur <= fee_high_eur)
);

CREATE TABLE recruitment_decision (             -- the DECISION, one per need per window
    decision_id     INTEGER PRIMARY KEY,
    need_id         INTEGER NOT NULL UNIQUE REFERENCES squad_need(need_id),
    decided_by      INTEGER NOT NULL REFERENCES staff(staff_id),   -- head_of_recruitment
    approved_by     INTEGER REFERENCES staff(staff_id),            -- sporting_director / cfo
    decided_on      DATE NOT NULL,
    outcome         TEXT NOT NULL CHECK (outcome IN ('pursue','no_signing','defer','promote_internal')),
    chosen_player_id INTEGER REFERENCES player(player_id),
    walk_away_fee_eur NUMERIC(14,2),
    walk_away_wage_weekly_eur NUMERIC(12,2),
    followed_model_top_pick BOOLEAN,
    rationale       TEXT NOT NULL,
    CHECK ((outcome = 'pursue') = (chosen_player_id IS NOT NULL))
);

CREATE TABLE negotiation_event (
    event_id        INTEGER PRIMARY KEY,
    decision_id     INTEGER NOT NULL REFERENCES recruitment_decision(decision_id),
    event_seq       INTEGER NOT NULL,
    event_date      DATE NOT NULL,
    event_type      TEXT NOT NULL CHECK (event_type IN ('bid','counter','rejected','accepted','player_terms_agreed','medical_passed','medical_failed','collapsed')),
    counterparty    TEXT NOT NULL CHECK (counterparty IN ('selling_club','player_agent','focal_club')),
    fee_eur         NUMERIC(14,2),
    weekly_wage_eur NUMERIC(12,2),
    UNIQUE (decision_id, event_seq)
);

CREATE TABLE transfer (                         -- executed moves, focal club and wider market
    transfer_id     INTEGER PRIMARY KEY,
    player_id       INTEGER NOT NULL REFERENCES player(player_id),
    from_club_id    INTEGER REFERENCES club(club_id),
    to_club_id      INTEGER NOT NULL REFERENCES club(club_id),
    transfer_date   DATE NOT NULL,
    fee_eur         NUMERIC(14,2) NOT NULL CHECK (fee_eur >= 0),
    agent_fee_eur   NUMERIC(14,2) NOT NULL DEFAULT 0,
    decision_id     INTEGER REFERENCES recruitment_decision(decision_id),   -- NULL for market transfers
    CHECK (from_club_id IS NULL OR from_club_id <> to_club_id)
);

-- ---------------------------------------------------------------
-- F. OUTCOMES (observed after the decision, for evaluation)
-- ---------------------------------------------------------------
CREATE TABLE signing_outcome (
    transfer_id     INTEGER NOT NULL REFERENCES transfer(transfer_id),
    season_id       INTEGER NOT NULL REFERENCES season(season_id),
    minutes_share   REAL NOT NULL CHECK (minutes_share BETWEEN 0 AND 1),
    performance_index REAL NOT NULL,             -- position-adjusted composite, 0-100
    market_value_end_eur NUMERIC(14,2) NOT NULL,
    success_label   TEXT NOT NULL CHECK (success_label IN ('hit','adequate','miss')),
    PRIMARY KEY (transfer_id, season_id)
);

-- Convenience view: season per-90 profile the screening tool reads
CREATE VIEW player_season_profile AS
SELECT  s.season_id, pms.player_id, pms.club_id,
        MAX(pms.position_group)                        AS position_group,
        COUNT(*)                                       AS apps,
        SUM(pms.minutes)                               AS minutes,
        90.0 * SUM(pms.npxg)              / SUM(pms.minutes) AS npxg_p90,
        90.0 * SUM(pms.xa)                / SUM(pms.minutes) AS xa_p90,
        90.0 * SUM(pms.progressive_passes)/ SUM(pms.minutes) AS progressive_passes_p90,
        90.0 * SUM(pms.progressive_carries)/SUM(pms.minutes) AS progressive_carries_p90,
        90.0 * SUM(pms.pressures)         / SUM(pms.minutes) AS pressures_p90,
        90.0 * SUM(pms.tackles_won)       / SUM(pms.minutes) AS tackles_won_p90,
        90.0 * SUM(pms.interceptions)     / SUM(pms.minutes) AS interceptions_p90,
        90.0 * SUM(pms.aerials_won)       / SUM(pms.minutes) AS aerials_won_p90,
        1.0 * SUM(pms.passes_completed) / NULLIF(SUM(pms.passes), 0)          AS pass_pct,
        1.0 * SUM(pms.gk_saves) / NULLIF(SUM(pms.gk_shots_on_target_faced), 0) AS gk_save_pct
FROM player_match_stats pms
JOIN match m  ON m.match_id = pms.match_id
JOIN season s ON s.season_id = m.season_id
GROUP BY s.season_id, pms.player_id, pms.club_id;
