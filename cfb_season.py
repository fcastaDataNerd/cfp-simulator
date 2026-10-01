"""cfb_season - the app's back end: real results + your picks -> everything on screen.

No Streamlit in here, so it can be tested on its own (check_cfb_app).

    import cfb_season as cs
    base  = cs.load_base(BASE_DIR)                       # once: data, model, Elo through 2025
    state = cs.build_state(base, picks, ccg_picks, po_picks)
    picks, ccg_picks, po_picks = cs.simulate(base, picks, ccg_picks, po_picks, "rest", rng)

PICKS
  picks      {game id: True/False}   True = home team wins     (regular-season games)
  ccg_picks  {conference: winner}                              (championship games)
  po_picks   {"R1-1" .. "FINAL": winner}                       (playoff games)
A real result always wins over a pick. Championship and playoff picks are stored by winner
name, so if earlier picks change who is in a game, a stale pick is simply ignored.

THE SEASON, IN ORDER
  1. regular season (weeks 1-13): Elo and every team's resume update after every result
  2. the poll: release 1 after week 9 (hidden until every game through week 9 has a
     result), then one per week through week 13; the next release is shown as PROVISIONAL
     while its week is being picked
  3. after week 13: conference standings + tiebreakers -> the ten championship games
  4. Selection Day (after the championship games) -> final ranking -> 12-team bracket
  5. playoff games: only Elo moves; the ranking is frozen
"""
import datetime as dt

import numpy as np
import pandas as pd

import cfb_engine as eng
import cfb_standings as st
import cfb_model as cm
import cfb_bracket as br

SEASON = 2026
RELEASE_WEEK = {1: 9, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14}       # release -> last game week it has seen
LAST_REGULAR_WEEK = 13
CCG_WEEK = 14
SELECTION_DAY = dt.date(2026, 12, 6)
CCG_ID0, PO_ID0 = 990_000_000, 991_000_000
CCG_START = "2026-12-05T20:00:00.000Z"
PO_CODES = [f"R1-{i}" for i in range(1, 5)] + [f"QF-{i}" for i in range(1, 5)] + ["SF-1", "SF-2", "FINAL"]
PO_START = {"R1": "2026-12-19T20:00:00.000Z", "QF": "2026-12-31T20:00:00.000Z",
            "SF": "2027-01-14T20:00:00.000Z", "FI": "2027-01-25T20:00:00.000Z"}
PO_ROUND = {"R1": "First round", "QF": "Quarterfinal", "SF": "Semifinal", "FI": "Championship"}
POOL_N = 45                                                    # Stage 2 pair pool: Elo top N + SOR top N


class Base:
    pass


def load_base(base_dir):
    """Everything that does not depend on picks. Elo is played once through 2025 and saved,
    so each click only replays the 2026 season."""
    b = Base()
    data, master = base_dir / "data", base_dir / "data" / "master"
    games = pd.read_excel(data / "games.xlsx")
    b.P = eng.load_elo_params(data)
    b.M = cm.load_model(master)
    b.DIV = st.load_divisions(data)
    hist = eng.prepare_games(games[games["season"] < SEASON])
    _, b.ratings_start = eng.run_elo(hist, b.P)
    b.conf_hist = eng.conference_map(hist)
    b.start_season = int(hist["season"].max())
    b.sched = games[games["season"] == SEASON].copy().reset_index(drop=True)
    b.week_of = dict(zip(b.sched["id"], b.sched["week"]))
    b.played = set(b.sched.loc[b.sched["completed"].fillna(False).astype(bool), "id"])
    gd = pd.to_datetime(b.sched["startDate"], utc=True).dt.tz_convert("America/New_York").dt.date
    b.cutoff = gd.groupby(b.sched["week"]).max().to_dict()
    b.cutoff[CCG_WEEK] = SELECTION_DAY
    b.weeks = sorted(int(w) for w in b.sched["week"].unique())
    b.real_through = max([int(w) for w in b.sched.loc[b.sched["id"].isin(b.played), "week"]] or [0])
    # real CFP polls for this season, if any exist yet: release j has seen games through
    # week RELEASE_WEEK[j], and CFBD labels that poll with the NEXT week
    cfp = pd.read_excel(data / "cfp_rankings.xlsx", sheet_name="cfp")
    cfp = cfp[cfp["season"] == SEASON]
    b.real_polls = {}
    for j, w in RELEASE_WEEK.items():
        c = cfp[cfp["week"] == w + 1]
        if len(c):
            b.real_polls[j] = dict(zip(c["team"], c["rank"]))
    info = pd.concat([b.sched[["homeTeam", "homeConference", "homeClassification"]].set_axis(["team", "conf", "cls"], axis=1),
                      b.sched[["awayTeam", "awayConference", "awayClassification"]].set_axis(["team", "conf", "cls"], axis=1)])
    info = info.drop_duplicates("team").set_index("team")
    b.conf_of, b.is_fbs = info["conf"].to_dict(), (info["cls"] == "fbs").to_dict()
    b.fbs_teams = sorted(t for t, v in b.is_fbs.items() if v)
    b.cache = {}
    return b


# =============================================================================
# Elo + resume for the season, from the saved end-of-last-season state
# =============================================================================
def _run(base, sched, picks):
    g = eng.prepare_games(sched, picks)
    ge, ratings = eng.run_elo(g, base.P, start_ratings=base.ratings_start, start_season=base.start_season,
                              conf=base.conf_hist)
    return g, ge, ratings, eng.team_games(g, ge)


def _rating(base, ratings, team):
    return ratings.get(team, base.P["FCS_ELO"]) if base.is_fbs.get(team, True) else base.P["FCS_ELO"]


def _release(base, j, tg, sig, is_sd):
    """Release j's inputs, cached on the results it depends on."""
    key = (j, sig)
    if key not in base.cache:
        if len(base.cache) > 400:
            base.cache.clear()
        cut = base.cutoff[RELEASE_WEEK[j]]
        feat = eng.features_at(tg, SEASON, cut, base.P, is_selection_day=is_sd)
        pool = set(feat.nsmallest(POOL_N, "elo_rank")["team"]) | set(feat.nlargest(POOL_N, "sor_elo")["team"])
        base.cache[key] = cm.Release(j, feat, is_first=(j == 1), is_sd=is_sd,
                                     pairs=cm.pair_evidence(tg, SEASON, cut, pool))
    return base.cache[key]


def _sig(base, picks, through_week):
    return tuple(sorted((i, v) for i, v in picks.items() if base.week_of.get(i, 99) <= through_week))


def _new_row(gid, week, season_type, start, home, away, neutral, base, ccg=False):
    return dict(id=gid, season=SEASON, week=week, seasonType=season_type, startDate=start, neutralSite=neutral,
                conferenceGame=ccg, homeTeam=home, awayTeam=away, homeConference=base.conf_of.get(home),
                awayConference=base.conf_of.get(away), homeClassification="fbs", awayClassification="fbs",
                homePoints=np.nan, awayPoints=np.nan, completed=False, isConfChampionship=ccg, notes=None)


# =============================================================================
# The whole state
# =============================================================================
def build_state(base, picks, ccg_picks=None, po_picks=None):
    ccg_picks, po_picks = ccg_picks or {}, po_picks or {}
    picks = {i: v for i, v in picks.items() if i in base.week_of and i not in base.played}
    S = dict(picks=picks)
    sched = base.sched
    g, ge, ratings, tg = _run(base, sched, picks)

    # ---- progress through the regular season ------------------------------------------------
    has = g.set_index("id")["hasResult"]
    wk = sched.assign(done=sched["id"].map(has).values).groupby("week")["done"].agg(["size", "sum"])
    S["week_status"] = wk.rename(columns={"size": "games", "sum": "with_result"})
    through = 0
    for w in range(1, LAST_REGULAR_WEEK + 1):
        if w in wk.index and wk.loc[w, "sum"] < wk.loc[w, "size"]:
            break
        through = w
    S["complete_through"] = through
    regular_done = through >= LAST_REGULAR_WEEK

    # ---- releases 1-5 -------------------------------------------------------------------------
    def release_status(j):
        w = RELEASE_WEEK[j]
        if through >= w:
            return "final", 0
        left = int(wk.loc[w, "size"] - wk.loc[w, "sum"]) if w in wk.index else 0
        started = w in wk.index and wk.loc[w, "sum"] > 0
        # release 1 is never provisional: before week 9 is complete the model has nothing it
        # was trained on. Later releases show provisionally while their week is being picked.
        if j >= 2 and through == w - 1 and started:
            return "provisional", left
        return None, left

    rels, rel_info = [], {}
    for j in range(1, 6):
        status, left = release_status(j)
        rel_info[j] = dict(release=j, week=RELEASE_WEEK[j], status=status, games_left=left,
                           real=j in base.real_polls)
        if status:
            rels.append(_release(base, j, tg, _sig(base, picks, RELEASE_WEEK[j]), False))

    # ---- resume "as of now" (always shown) ----------------------------------------------------
    last_date = max(tg.loc[tg["season"] == SEASON, "gdate"])
    S["as_of"] = last_date
    S["resume_now"] = eng.features_at(tg, SEASON, last_date, base.P)

    preds = cm.predict_season(base.M, rels, base.real_polls) if rels else {}

    # ---- standings and the championship matchups ------------------------------------------------
    if 5 in preds and rel_info[5]["status"] == "final":
        rating_rank = dict(zip(preds[5]["team"], preds[5]["rank"]))
        rating_src = "our predicted ranking after week 13"
    else:
        rating_rank = dict(zip(S["resume_now"]["team"], S["resume_now"]["elo_rank"]))
        rating_src = "current Elo rank (the predicted ranking is used once week 13 is complete)"
    if 4 in base.real_polls:
        cfp_rank, cfp_src = base.real_polls[4], "the real CFP poll after week 12"
    elif 4 in preds and rel_info[4]["status"] == "final":
        p4 = preds[4]
        cfp_rank = dict(zip(p4.loc[p4["rank"] <= 25, "team"], p4.loc[p4["rank"] <= 25, "rank"]))
        cfp_src = "our predicted poll after week 12"
    else:
        cfp_rank, cfp_src = {}, "none yet"
    mu = st.championship_matchups(tg, SEASON, None, rating_rank, cfp_rank, base.DIV)
    S["matchups"], S["matchups_final"] = mu, regular_done
    S["tiebreak_inputs"] = dict(rating=rating_src, cfp=cfp_src)

    # ---- championship games ---------------------------------------------------------------------
    extra, ccg_meta = [], []
    all_picks = dict(picks)
    if regular_done:
        conf_order = list(st.RULES)
        for r in mu.itertuples(index=False):
            gid = CCG_ID0 + conf_order.index(r.conference)
            neutral = r.home_team is None or (isinstance(r.home_team, float) and np.isnan(r.home_team))
            extra.append(_new_row(gid, CCG_WEEK, "regular", CCG_START, r.seed1, r.seed2, bool(neutral), base, ccg=True))
            w = ccg_picks.get(r.conference)
            if w in (r.seed1, r.seed2):
                all_picks[gid] = (w == r.seed1)
            ccg_meta.append(dict(id=gid, conference=r.conference, winner=w if w in (r.seed1, r.seed2) else None))
    S["ccg"] = pd.DataFrame(ccg_meta)
    ccg_done = regular_done and len(ccg_meta) > 0 and all(m["winner"] for m in ccg_meta)

    if extra:
        sched = pd.concat([sched, pd.DataFrame(extra)], ignore_index=True)
        g, ge, ratings, tg = _run(base, sched, all_picks)
        n_picked = sum(1 for m in ccg_meta if m["winner"])
        status = "final" if ccg_done else ("provisional" if n_picked else None)
        rel_info[6] = dict(release=6, week=CCG_WEEK, status=status, games_left=len(ccg_meta) - n_picked,
                           real=6 in base.real_polls)
        if status:
            sig = _sig(base, picks, 99) + tuple(sorted((m["conference"], m["winner"]) for m in ccg_meta))
            rels.append(_release(base, 6, tg, sig, True))
            preds = cm.predict_season(base.M, rels, base.real_polls)
    else:
        rel_info[6] = dict(release=6, week=CCG_WEEK, status=None, games_left=len(st.RULES), real=6 in base.real_polls)
    S["release_info"] = pd.DataFrame(rel_info.values())
    S["releases"], S["preds"] = rels, preds

    # ---- bracket and playoff games ----------------------------------------------------------------
    S["bracket"], S["po_results"] = None, {}
    po_rows, po_meta = [], []
    if ccg_done and 6 in preds:
        champs = br.champions(tg, SEASON)
        b = br.build_bracket(preds[6][["team", "rank", "conference"]], champs, rules="2026")
        s2t = dict(zip(b["field"]["seed"], b["field"]["team"]))
        slot, won = {}, {}

        def settle(code, home, away, neutral):
            slot[code] = (home, away)
            if home is None or away is None:
                return
            gid = PO_ID0 + PO_CODES.index(code)
            po_rows.append(_new_row(gid, 1, "postseason", PO_START[code[:2]], home, away, neutral, base))
            w = po_picks.get(code)
            if w in (home, away):
                won[code] = w
                all_picks[gid] = (w == home)
            po_meta.append(dict(id=gid, code=code, round=PO_ROUND[code[:2]], winner=won.get(code)))

        for i, (h, a) in enumerate(br.FIRST_ROUND):
            settle(f"R1-{i + 1}", s2t[h], s2t[a], False)
        for i, (bye, g_) in enumerate(br.QUARTERS):
            settle(f"QF-{i + 1}", s2t[bye], won.get(f"R1-{g_ + 1}"), True)
        for i, (a, b_) in enumerate(br.SEMIS):
            settle(f"SF-{i + 1}", won.get(f"QF-{a + 1}"), won.get(f"QF-{b_ + 1}"), True)
        settle("FINAL", won.get("SF-1"), won.get("SF-2"), True)
        if po_rows:
            sched = pd.concat([sched, pd.DataFrame(po_rows)], ignore_index=True)
            g, ge, ratings, tg = _run(base, sched, all_picks)
        elo_now = {t: _rating(base, ratings, t) for t in b["field"]["team"]}
        b["odds"] = br.advance_odds(b, elo_now, base.P, results=won)
        b["slots"] = slot
        S["bracket"], S["po_results"] = b, won
    S["playoff"] = pd.DataFrame(po_meta)
    S["champion"] = S["po_results"].get("FINAL")

    # ---- standings: ties broken by each conference's own procedure; overall record includes
    #      the championship game once it has a result
    S["standings"] = st.ordered_standings(tg, SEASON, None, rating_rank, cfp_rank, base.DIV)

    # ---- every game, with its Elo win probability ---------------------------------------------------
    # played / picked games: the probability BEFORE kickoff, from both teams' Elo at that time.
    # open games: from today's Elo.
    pre = ge.set_index("gameId")["homeWinProb"].to_dict()
    gm = g[["id", "week", "seasonType", "gdate", "homeTeam", "awayTeam", "neutral", "hasResult", "isPick",
            "homeWon", "isConfChampionship", "homePoints", "awayPoints"]].copy()
    gm["p_home"] = [pre[i] if i in pre else eng.win_prob(_rating(base, ratings, h), _rating(base, ratings, a), n, base.P)
                    for i, h, a, n in zip(gm["id"], gm["homeTeam"], gm["awayTeam"], gm["neutral"])]
    gm["status"] = np.where(~gm["hasResult"], "open", np.where(gm["isPick"], "pick", "final"))
    gm["winner"] = np.where(gm["hasResult"], np.where(gm["homeWon"].fillna(False).astype(bool), gm["homeTeam"], gm["awayTeam"]), None)
    code_of = {m["id"]: m["code"] for m in po_meta}
    conf_of_ccg = {m["id"]: m["conference"] for m in ccg_meta}
    gm["code"] = gm["id"].map(code_of)
    gm["ccg_conference"] = gm["id"].map(conf_of_ccg)
    gm["stage"] = np.where(gm["seasonType"] == "postseason", "playoff",
                           np.where(gm["isConfChampionship"], "championship", "regular"))
    S["games"] = gm.sort_values(["gdate", "id"]).reset_index(drop=True)
    S["ratings"], S["tg"] = ratings, tg
    S["schedule"], S["all_picks"] = sched, all_picks        # exactly what the engine was given
    return S


def ranges(base, state, seed=0):
    """Uncertainty about the COMMITTEE, given the results so far: 500 simulated polls per shown
    release (100 bootstrap coefficient sets x 5 Plackett-Luce draws). Fixed seed, so the same
    picks always give the same numbers.
    -> {j: DataFrame(team, median_rank, range_lo, range_hi, p_top4, p_top12, p_top25)}
    On Selection Day, once every championship game has a winner, each simulated poll is also
    run through the bracket rules -> p_playoff (in the 12-team field) and p_bye (seed 1-4)."""
    if not state["releases"]:
        return {}
    sims = cm.simulate_season(base.M, state["releases"], base.real_polls, rng=np.random.default_rng(seed))
    out = {j: cm.summarize_sims(t, r) for j, (t, r) in sims.items()}
    if state["bracket"] is not None and 6 in sims:
        teams, ranks = sims[6]
        conf = dict(zip(state["preds"][6]["team"], state["preds"][6]["conference"]))
        champs = br.champions(state["tg"], SEASON)
        made, bye = dict.fromkeys(teams, 0), dict.fromkeys(teams, 0)
        for row in ranks:
            order = [teams[i] for i in np.argsort(row, kind="stable")]
            seeded = br.select_field(order, dict(zip(teams, row)), conf, champs)[0]
            for sd_, t in enumerate(seeded, start=1):
                made[t] += 1
                bye[t] += sd_ <= 4
        out[6]["p_playoff"] = out[6]["team"].map(made) / len(ranks)
        out[6]["p_bye"] = out[6]["team"].map(bye) / len(ranks)
    return out


# =============================================================================
# The forecast - only ever as of a COMPLETED stage
# =============================================================================
def forecast_inputs(base, state, ccg_picks):
    """What the forecast is conditioned on. It moves only when a whole week has a result
    (user decision): picks in a week that is still being picked are ignored until the week is
    complete; the title games count once all ten are picked. Playoff picks never enter - after
    Selection Day the playoff odds live on the Bracket tab.
    -> (picks, ccg_picks, fixed_matchups, label, key)"""
    through = state["complete_through"]
    picks = {i: v for i, v in state["picks"].items() if base.week_of[i] <= through}
    ccg, fixed, label = {}, None, f"the end of week {through}"
    if state["matchups_final"]:
        m = state["matchups"]
        fixed = {r.conference: (r.seed1, r.seed2) for r in m.itertuples(index=False)}
        done = len(state["ccg"]) and state["ccg"]["winner"].notna().all()
        if done:
            ccg = {c: w for c, w in zip(state["ccg"]["conference"], state["ccg"]["winner"])}
            label = "Selection Day (every title game decided)"
        else:
            label = "the end of the regular season"
    key = (tuple(sorted(picks.items())), tuple(sorted(ccg.items())),
           tuple(sorted(fixed.items())) if fixed else None)
    return picks, ccg, fixed, label, key


# =============================================================================
# Simulation - draw winners from the Elo win probabilities
# =============================================================================
def simulate(base, picks, ccg_picks, po_picks, scope, rng, week=None):
    """scope "week": the open games of regular week `week` (14 = championship games,
    "playoff" = the next playoff round). scope "rest": everything still open, in order,
    week by week so Elo updates before the next week is drawn."""
    picks, ccg_picks, po_picks = dict(picks), dict(ccg_picks), dict(po_picks)

    def sim_regular(w):
        gq = eng.prepare_games(base.sched, picks)
        opn = gq[(gq["week"] == w) & ~gq["hasResult"]]
        if opn.empty:
            return
        _, ratings = eng.run_elo(gq, base.P, start_ratings=base.ratings_start, start_season=base.start_season,
                                 conf=base.conf_hist)
        for i, h, a, n in zip(opn["id"], opn["homeTeam"], opn["awayTeam"], opn["neutral"]):
            p = eng.win_prob(_rating(base, ratings, h), _rating(base, ratings, a), n, base.P)
            picks[i] = bool(rng.random() < p)

    def sim_ccg():
        stt = build_state(base, picks, ccg_picks, po_picks)
        gm = stt["games"]
        for r in gm[(gm["stage"] == "championship") & (gm["status"] == "open")].itertuples(index=False):
            ccg_picks[r.ccg_conference] = r.homeTeam if rng.random() < r.p_home else r.awayTeam

    def sim_playoff_round():
        stt = build_state(base, picks, ccg_picks, po_picks)
        gm = stt["games"]
        opn = gm[(gm["stage"] == "playoff") & (gm["status"] == "open")]
        for r in opn.itertuples(index=False):
            po_picks[r.code] = r.homeTeam if rng.random() < r.p_home else r.awayTeam
        return len(opn)

    if scope == "week":
        if week == "playoff":
            sim_playoff_round()
        elif week == CCG_WEEK:
            sim_ccg()
        else:
            sim_regular(int(week))
    elif scope == "rest":
        for w in [w for w in base.weeks if w <= LAST_REGULAR_WEEK]:
            sim_regular(w)
        sim_ccg()
        for w in [w for w in base.weeks if w > CCG_WEEK]:              # Army-Navy
            sim_regular(w)
        for _ in range(4):
            if not sim_playoff_round():
                break
    else:
        raise ValueError(scope)
    return picks, ccg_picks, po_picks
