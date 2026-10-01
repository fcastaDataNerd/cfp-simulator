"""cfb_standings - conference standings and who plays in each championship game.

Back end for the app: after any set of results/picks it gives the live conference
standings, and once the regular season is complete, the two participants (seeded 1 and 2)
of every conference championship game, with a plain-language reason for each choice.

    import cfb_standings as st
    st.standings(tg, season, cutoff)                       -> one row per team
    st.championship_matchups(tg, season, cutoff, rating_rank, cfp_rank, divisions)

Inputs
  tg           cfb_engine.team_games(...)   (one row per FBS team per game with a result)
  cutoff       last date of the regular season to use
  rating_rank  {team: rank, 1 = best}. OUR stand-in for every step a conference decides with
               something we cannot compute (SportSource ratings, computer composites, the
               SEC's capped scoring margin, the ACC's Team Success Ranking). In the app this
               is our predicted ranking of all FBS teams after the regular season.
  cfp_rank     {team: rank} from the CFP poll released BEFORE the final regular-season
               weekend (real poll if it exists, else our predicted one). Used by the
               conferences whose rule is "highest-ranked team that wins its last game".
  divisions    {(season, team): division}  (Sun Belt only)

SECTION 1 is the rules table - one entry per conference, with the source of each. Every
procedure is a list of the shared steps defined in SECTION 3.
"""
import numpy as np
import pandas as pd

import cfb_common as cc

# =============================================================================
# 1. THE RULES  (edit here; nothing below hard-codes a conference)
# =============================================================================
# Step names (SECTION 3):
#   h2h         two teams: winner of their game(s)
#   rr          3+ teams, ALL played each other: record in games among the tied teams
#   beat_all    3+ teams: a team that beat every other tied team advances
#   swept       beat_all, PLUS a team that lost to every other tied team drops out
#               (SEC, ACC 2026, American, Mountain West, C-USA, Sun Belt word it both ways;
#               Big Ten, Big 12, MAC only advance the team that beat everyone - in 2025
#               Miami (OH) lost to both other tied MAC teams and still advanced)
#   common      win % against conference opponents every tied team played
#   walk        record against the next-highest-placed COMMON opponent (one every tied team
#               played), working down the standings; common opponents tied with each other
#               count as one unit
#   opp_pct     combined conference win % of each team's conference opponents
#   total_wins  regular-season wins, at most one FCS win counted
#   overall     overall regular-season win %, at most one FCS win counted
#   div_pct     win % against division opponents              (Sun Belt)
#   common_nondiv  win % against common conference opponents outside the division (Sun Belt)
#   cfp_win     the highest team in `cfp_rank` among those tied, IF it won its final
#               regular-season game; otherwise the step is skipped
#   cfp         higher in `cfp_rank` (no win condition)       (Pac-12)
#   rating      our stand-in ranking (see `rating_rank`) - always separates
#
# After any step separates the tied group, the leaders restart from the first step (two
# teams -> the two-team list). When one team is seeded, the rest restart for the next seat.
_P5 = dict(two=["h2h", "common", "walk", "opp_pct", "rating"],
           multi=["rr", "beat_all", "common", "walk", "opp_pct", "rating"])
_P5_SWEPT = dict(two=_P5["two"], multi=["rr", "swept", "common", "walk", "opp_pct", "rating"])
_G5 = dict(two=["h2h", "cfp_win", "rating"], multi=["rr", "swept", "cfp_win", "rating"])

RULES = {
    "SEC": dict(format="top2", site="neutral", **_P5_SWEPT,
                note="last real step is capped scoring margin (SportSource) -> rating",
                source="ESPN 2024 summary; PFN SEC tiebreakers; 9 conference games from 2026"),
    "Big Ten": dict(format="top2", site="neutral", **_P5,
                    note="step 5 is the SportSource Team Rating Score -> rating",
                    source="ESPN 2024 summary; CBS Sports (no 2026 change found)"),
    "Big 12": dict(format="top2", site="neutral",
                   two=["h2h", "common", "walk", "opp_pct", "total_wins", "rating"],
                   multi=["rr", "beat_all", "common", "walk", "opp_pct", "total_wins", "rating"],
                   note="2026 uses the same policy as 2025; SportSource rating -> rating",
                   source="Wikipedia 2026 Big 12 Championship Game; big12ology.com"),
    "ACC": {
        (2000, 2025): dict(format="top2", site="neutral", **_P5,
                           note="pre-2026 six-step policy", source="ESPN 2024 summary"),
        (2026, 2099): dict(format="top2", site="neutral", tied="acc2026",
                           two=["h2h", "rating"], multi=["rr", "swept", "rating"],
                           note="July 2026 policy: head-to-head, then SportSource Team Success Ranking "
                                "(-> rating). 12 teams play 9 conference games and 5 play 8: a team with a "
                                "different number of games counts as TIED with the leader if it has the same "
                                "number of wins OR the same number of losses.",
                           source="ACC Football Tiebreaker Policy, as amended July 1 2026 (official PDF)"),
    },
    "Mid-American": dict(format="top2", site="neutral", **_P5,
                         note="neutral site (Ford Field, Detroit); SportSource rating -> rating",
                         source="MAC 2024 tiebreaker release; ESPN 2024 summary"),
    "American Athletic": dict(format="top2", site="seed1_home",
                              two=["h2h", "cfp_win", "rating", "common", "overall"],
                              multi=["rr", "swept", "cfp_win", "rating", "common", "overall"],
                              nonconf=[("Army", "Navy")],
                              note="composite of SP+, SportSource SOR, ESPN SOR, KPI -> rating. "
                                   "Army-Navy is not a conference game.",
                              source="theamerican.org 2025 tiebreaker scenarios; ESPN 2024 summary"),
    "Mountain West": dict(format="top2", site="seed1_home",
                          two=["h2h", "cfp_win", "rating", "overall", "walk", "common"],
                          multi=["rr", "swept", "cfp_win", "rating", "overall", "walk", "common"],
                          nonconf=[(2026, "San José State", "North Dakota State")],
                          note="computer-metric average -> rating. 2026: the Nov 27 San Jose State - "
                               "North Dakota State game is a NON-conference game (every team plays 8).",
                          source="PFN Mountain West tiebreakers; ESPN 2024 summary; FBSchedules 2026"),
    "Conference USA": dict(format="top2", site="seed1_home", **_G5,
                           note="computer rankings, then APR -> rating",
                           source="PFN Conference USA tiebreakers; ESPN 2024 summary"),
    "Sun Belt": dict(format="divisions", site="seed1_home",
                     two=["h2h", "div_pct", "walk", "common_nondiv", "cfp_win", "rating", "overall"],
                     multi=["rr", "swept", "div_pct", "walk", "common_nondiv", "cfp_win", "rating", "overall"],
                     note="East and West division winners; host = better conference record. "
                          "Computer composite -> rating",
                     source="PFN Sun Belt tiebreakers; ESPN 2024 summary"),
    "Pac-12": {
        (2000, 2023): dict(format="top2", site="neutral", **_P5,
                           note="old Pac-12 (divisions through 2021 - see DIVISION ERA below)",
                           source="historical"),
        (2026, 2099): dict(format="top2", site="seed1_home", last_week=12,
                           two=["h2h", "walk", "common", "cfp", "rating"],
                           multi=["rr", "common", "walk", "opp_pct", "rating"],
                           note="8 teams, 7-game round robin; standings through week 12 only - the week-13 "
                                "flex game between members is non-conference. Game at the #1 seed's home.",
                           source="pac-12.com, Sept 3 2026 operational update (official)"),
    },
}
# DIVISION ERA (history only, used by the 2014-2023 replay). Before the conferences dropped
# divisions, the title game was division winner vs division winner. Whenever the division
# file lists divisions for a conference in a season, that format and this generic division
# procedure are used instead of the top-two lists above. The Sun Belt (still divisional) keeps
# its own list.
_DIV_ERA = dict(two=["h2h", "div_pct", "walk", "common_nondiv", "cfp", "rating"],
                multi=["rr", "swept", "div_pct", "walk", "common_nondiv", "cfp", "rating"])

# Teams that could not play in the title game (postseason-ineligible). Their games still
# count in everyone's record.
INELIGIBLE = {(2022, "James Madison"), (2023, "James Madison")}      # FBS transition years

STEP_TEXT = {
    "h2h": "head-to-head", "rr": "record among the tied teams", "beat_all": "beat every other tied team",
    "swept": "beat (or lost to) every other tied team",
    "common": "record vs common opponents", "walk": "record vs the next-highest common opponent",
    "opp_pct": "opponents' conference win %", "total_wins": "total wins", "overall": "overall win %",
    "div_pct": "division record", "common_nondiv": "record vs common non-division opponents",
    "cfp_win": "highest CFP-ranked team that won its last game", "cfp": "CFP ranking",
    "rating": "our ranking (stand-in for the conference's rating step)",
}


def rules_for(conf, season):
    r = RULES.get(conf)
    if r is None:
        return None
    if "format" in r:
        return r
    for (lo, hi), v in r.items():
        if lo <= season <= hi:
            return v
    return None


def rules_table(season):
    """The rules as a table, for display and review."""
    rows = []
    for conf in RULES:
        r = rules_for(conf, season)
        if r is None:
            continue
        rows.append(dict(conference=conf, participants="division winners" if r["format"] == "divisions" else "top two",
                         site="#1 seed's home" if r["site"] == "seed1_home" else "neutral",
                         two_team_tie=" -> ".join(r["two"]), three_plus_tie=" -> ".join(r["multi"]),
                         note=r.get("note", ""), source=r.get("source", "")))
    return pd.DataFrame(rows)


# =============================================================================
# 2. Standings
# =============================================================================
def _season_rows(tg, season, cutoff):
    d = tg[(tg["season"] == season) & ~tg["is_ccg"]]
    return d if cutoff is None else d[d["gdate"] <= cutoff]


def conference_rows(tg, season, cutoff=None):
    """The team-game rows that count in the conference standings."""
    d = _season_rows(tg, season, cutoff)
    d = d[(d["conference"] == d["opp_conference"]) & d["conf_flag"] & ~d["opp_is_fcs"]
          & (d["conference"] != "FBS Independents")]
    keep = np.ones(len(d), bool)
    for i, (conf, wk, t, o) in enumerate(zip(d["conference"], d["week"], d["team"], d["opponent"])):
        r = rules_for(conf, season) or {}
        if "last_week" in r and wk > r["last_week"]:
            keep[i] = False
        for pair in r.get("nonconf", []):
            if len(pair) == 3 and pair[0] != season:
                continue
            if {t, o} == set(pair[-2:]):
                keep[i] = False
    return d[keep]


def standings(tg, season, cutoff=None, divisions=None):
    """One row per FBS team: conference and overall record, sorted within conference."""
    allr = _season_rows(tg, season, cutoff)
    cr = conference_rows(tg, season, cutoff)
    conf = allr.groupby("team")["conference"].agg(lambda x: x.mode().iloc[0] if len(x.dropna()) else None)
    out = pd.DataFrame({"conference": conf})
    out["conf_w"] = cr.groupby("team")["win"].sum().reindex(out.index).fillna(0).astype(int)
    out["conf_g"] = cr.groupby("team")["win"].size().reindex(out.index).fillna(0).astype(int)
    out["conf_l"] = out["conf_g"] - out["conf_w"]
    out["conf_pct"] = np.where(out["conf_g"] > 0, out["conf_w"] / out["conf_g"].replace(0, np.nan), np.nan)
    # overall record INCLUDES the championship game (the conference record never does)
    full = tg[tg["season"] == season]
    full = full if cutoff is None else full[full["gdate"] <= cutoff]
    out["w"] = full.groupby("team")["win"].sum().reindex(out.index).fillna(0).astype(int)
    out["l"] = (full.groupby("team")["win"].size().reindex(out.index).fillna(0) - out["w"]).astype(int)
    out = out.reset_index()
    out["division"] = [None if divisions is None else divisions.get((season, t)) for t in out["team"]]
    return out.sort_values(["conference", "conf_pct", "conf_w", "team"],
                           ascending=[True, False, False, True]).reset_index(drop=True)


def ordered_standings(tg, season, cutoff=None, rating_rank=None, cfp_rank=None, divisions=None):
    """standings() with a `place` column: every tie broken by that conference's own procedure,
    so the table reads in the same order the championship game is seeded (within each division
    where a conference has them). Teams yet to play a conference game go last."""
    sd = standings(tg, season, cutoff, divisions)
    place = {}
    for conf in RULES:
        if rules_for(conf, season) is None:
            continue
        c = _Ctx(conf, season, tg, cutoff, rating_rank, cfp_rank, divisions)
        if not c.teams:
            continue
        groups = ([[t for t in c.teams if c.div[t] == dv] for dv in sorted({d for d in c.div.values() if d})]
                  if c.rules["format"] == "divisions" else [c.teams])
        for pool in groups:
            for i, (t, _) in enumerate(_seed(pool, len(pool), c), start=1):
                place[t] = i
    sd["place"] = sd["team"].map(place)
    sd["_d"] = sd["division"].fillna("")
    sd = sd.sort_values(["conference", "_d", "place", "conf_pct", "team"],
                        ascending=[True, True, True, False, True], na_position="last")
    return sd.drop(columns="_d").reset_index(drop=True)


def load_divisions(data_dir):
    t = pd.read_excel(data_dir / "teams_conferences.xlsx")
    t = t[t["division"].notna()]
    return {(int(s), sc): dv for s, sc, dv in zip(t["season"], t["school"], t["division"])}


# =============================================================================
# 3. The tiebreak steps. Each takes the tied teams and returns them in tiers,
#    best first (a list of lists), or None when the step does not separate them.
# =============================================================================
class _Ctx:
    """Everything the steps need for one conference in one season."""

    def __init__(self, conf, season, tg, cutoff, rating_rank, cfp_rank, divisions):
        self.conf, self.season = conf, season
        self.rules = rules_for(conf, season)
        cr = conference_rows(tg, season, cutoff)
        cr = cr[cr["conference"] == conf]
        self.teams = sorted(cr["team"].unique())
        self.rec = {}                                   # (team, opponent) -> [wins, games]
        for t, o, w in zip(cr["team"], cr["opponent"], cr["win"]):
            r = self.rec.setdefault((t, o), [0, 0])
            r[0] += int(bool(w))
            r[1] += 1
        allr = _season_rows(tg, season, cutoff)
        allr = allr[allr["team"].isin(self.teams)].sort_values(["team", "gdate"])
        self.total_wins, self.overall, self.final_win = {}, {}, {}
        for t, d in allr.groupby("team"):
            fcs_w = int((d["win"].astype(bool) & d["opp_is_fcs"]).sum())
            extra = max(0, fcs_w - 1)                    # at most one FCS win counts
            wins, games = int(d["win"].sum()) - extra, len(d) - extra
            self.total_wins[t], self.overall[t] = wins, (wins / games if games else 0.0)
            self.final_win[t] = bool(d["win"].iloc[-1])
        self._finish(rating_rank, cfp_rank, divisions)

    def _finish(self, rating_rank, cfp_rank, divisions):
        """Everything derived from self.rec / self.teams (shared by both constructors)."""
        season = self.season
        self.w, self.g, self.opps = dict.fromkeys(self.teams, 0), dict.fromkeys(self.teams, 0), {t: set() for t in self.teams}
        for (a, o), v in self.rec.items():
            self.w[a] += v[0]
            self.g[a] += v[1]
            self.opps[a].add(o)
        self.pct = {t: (self.w[t] / self.g[t] if self.g[t] else 0.0) for t in self.teams}
        self.rating = rating_rank or {}
        self.cfp = cfp_rank or {}
        self.div = {t: (divisions or {}).get((season, t)) for t in self.teams}
        if self.rules["format"] != "divisions" and any(self.div.values()):
            self.rules = dict(self.rules, format="divisions", **_DIV_ERA)      # division era
        self.eligible = [t for t in self.teams if (season, t) not in INELIGIBLE]

    @classmethod
    def from_records(cls, conf, season, teams, games, total_wins, overall, final_win,
                     rating_rank=None, cfp_rank=None, divisions=None):
        """The same context without a table: `games` = (home, away, home_won) for every game
        that counts in this conference's standings; the three dicts are per team. Used by the
        fast season simulator (cfb_forecast); checked against the table version there."""
        c = cls.__new__(cls)
        c.conf, c.season, c.rules, c.teams = conf, season, rules_for(conf, season), sorted(teams)
        c.rec = {}
        for h, a, hw in games:
            r = c.rec.setdefault((h, a), [0, 0])
            r[0] += int(hw)
            r[1] += 1
            r = c.rec.setdefault((a, h), [0, 0])
            r[0] += int(not hw)
            r[1] += 1
        c.total_wins, c.overall, c.final_win = total_wins, overall, final_win
        c._finish(rating_rank, cfp_rank, divisions)
        return c

    def record_vs(self, t, others):
        w = sum(self.rec[(t, o)][0] for o in others if (t, o) in self.rec)
        g = sum(self.rec[(t, o)][1] for o in others if (t, o) in self.rec)
        return w, g


def _tiers(T, score, higher=True):
    """Group teams by score; None if they all share one score."""
    vals = sorted({round(score[t], 9) for t in T}, reverse=higher)
    if len(vals) <= 1:
        return None
    return [[t for t in T if round(score[t], 9) == v] for v in vals]


def _all_played(T, c):
    return all((a, b) in c.rec for a in T for b in T if a != b)


def s_h2h(T, c):
    a, b = T
    if (a, b) not in c.rec:
        return None
    wa, g = c.rec[(a, b)]
    return None if wa * 2 == g else ([[a], [b]] if wa * 2 > g else [[b], [a]])


def s_rr(T, c):
    if not _all_played(T, c):
        return None
    sc = {}
    for t in T:
        w, g = c.record_vs(t, [o for o in T if o != t])
        sc[t] = w / g
    return _tiers(T, sc)


def _sweep(T, c, drop_loser):
    def swept(t, win):
        for o in T:
            if o == t:
                continue
            if (t, o) not in c.rec:
                return False
            w, g = c.rec[(t, o)]
            if (w != g) if win else (w != 0):
                return False
        return True
    sup = [t for t in T if swept(t, True)]
    inf = [t for t in T if swept(t, False)]
    if len(sup) == 1:
        rest = [t for t in T if t != sup[0]]
        return [sup, rest]
    if drop_loser and len(inf) == 1:
        return [[t for t in T if t != inf[0]], inf]
    return None


def s_beat_all(T, c):
    return _sweep(T, c, drop_loser=False)


def s_swept(T, c):
    return _sweep(T, c, drop_loser=True)


def _pct_vs(T, c, pool):
    sc = {}
    for t in T:
        w, g = c.record_vs(t, pool)
        if g == 0:
            return None
        sc[t] = w / g
    return _tiers(T, sc)


def s_common(T, c):
    common = set.intersection(*[c.opps[t] for t in T]) - set(T)
    return _pct_vs(T, c, common) if common else None


def s_walk(T, c):
    """Down the standings, one COMMON opponent at a time (opponents every tied team played);
    common opponents with the same record are compared as one unit."""
    pool = set.intersection(*[c.opps[t] for t in T]) - set(T)
    if c.rules["format"] == "divisions":                    # Sun Belt walks the division standings
        pool = {t for t in pool if c.div[t] == c.div[T[0]]}
    for v in sorted({round(c.pct[t], 9) for t in pool}, reverse=True):
        out = _pct_vs(T, c, [t for t in pool if round(c.pct[t], 9) == v])
        if out:
            return out
    return None


def s_opp_pct(T, c):
    sc = {}
    for t in T:
        w = sum(c.w[o] for o in c.opps[t])
        g = sum(c.g[o] for o in c.opps[t])
        sc[t] = w / g if g else 0.0
    return _tiers(T, sc)


def s_total_wins(T, c):
    return _tiers(T, {t: c.total_wins[t] for t in T})


def s_overall(T, c):
    return _tiers(T, {t: c.overall[t] for t in T})


def s_div_pct(T, c):
    sc = {}
    for t in T:
        if c.div[t] is None:
            return None
        w, g = c.record_vs(t, [o for o in c.teams if c.div[o] == c.div[t] and o != t])
        if g == 0:
            return None
        sc[t] = w / g
    return _tiers(T, sc)


def s_common_nondiv(T, c):
    if c.div[T[0]] is None:
        return None
    common = {o for o in set.intersection(*[c.opps[t] for t in T]) if c.div[o] != c.div[T[0]]}
    return _pct_vs(T, c, common) if common else None


def s_cfp_win(T, c):
    ranked = [t for t in T if t in c.cfp]
    if not ranked:
        return None
    best = min(ranked, key=lambda t: c.cfp[t])
    if not c.final_win.get(best, False):
        return None                                         # it lost its last game -> next step
    return [[best], [t for t in T if t != best]]


def s_cfp(T, c):
    return _tiers(T, {t: c.cfp.get(t, 999) for t in T}, higher=False)


def s_rating(T, c):
    return _tiers(T, {t: c.rating.get(t, 9999) for t in T}, higher=False)


STEPS = {"h2h": s_h2h, "rr": s_rr, "beat_all": s_beat_all, "swept": s_swept, "common": s_common, "walk": s_walk,
         "opp_pct": s_opp_pct, "total_wins": s_total_wins, "overall": s_overall, "div_pct": s_div_pct,
         "common_nondiv": s_common_nondiv, "cfp_win": s_cfp_win, "cfp": s_cfp, "rating": s_rating}


# =============================================================================
# 4. Resolving ties and seeding
# =============================================================================
def pick_best(T, c, trail=None):
    """The best of the tied teams, and how it was decided."""
    trail = [] if trail is None else trail
    T = list(T)
    if len(T) == 1:
        return T[0], trail
    for name in (c.rules["two"] if len(T) == 2 else c.rules["multi"]):
        tiers = STEPS[name](T, c)
        if tiers:
            top = tiers[0]
            trail = trail + [f"{STEP_TEXT[name]} ({len(T)} tied -> {len(top)})"]
            return (top[0], trail) if len(top) == 1 else pick_best(top, c, trail)   # leaders restart
    best = sorted(T)[0]                                     # draw: never reached while `rating` is listed
    return best, trail + ["draw (alphabetical)"]


def _fmt(c, t):
    return f"{c.w[t]}-{c.g[t] - c.w[t]}"


def _tied_with_leader(pool, c):
    """Teams tied for the best conference record in `pool`."""
    best = max(c.pct[t] for t in pool)
    lead = [t for t in pool if abs(c.pct[t] - best) < 1e-9]
    if c.rules.get("tied") == "acc2026":
        # a team with a different number of conference games is tied with the leader if it
        # has the same number of wins OR the same number of losses
        for t in pool:
            if t in lead:
                continue
            for q in list(lead):
                if c.g[t] != c.g[q] and (c.w[t] == c.w[q] or (c.g[t] - c.w[t]) == (c.g[q] - c.w[q])):
                    lead.append(t)
                    break
    return lead


def _seed(pool, n, c):
    """Seed the top n teams of `pool`. Returns [(team, reason)]."""
    pool, out = list(pool), []
    while len(out) < n and pool:
        tied = _tied_with_leader(pool, c)
        if len(tied) == 1:
            t, why = tied[0], f"best conference record ({_fmt(c, tied[0])})"
        else:
            t, trail = pick_best(tied, c)
            others = ", ".join(x for x in tied if x != t)
            why = f"{len(tied)}-way tie at {_fmt(c, t)} with {others}: " + "; then ".join(trail)
        out.append((t, why))
        pool.remove(t)
    return out


def championship_matchups(tg, season, cutoff=None, rating_rank=None, cfp_rank=None, divisions=None):
    """Seeds 1 and 2 of every conference championship game, with the site and the reasons."""
    rows = []
    for conf in RULES:
        if rules_for(conf, season) is None or conf not in cc.ccg_conferences(season):
            continue
        c = _Ctx(conf, season, tg, cutoff, rating_rank, cfp_rank, divisions)
        if not c.teams:
            continue
        seeds = seeds_for(c)
        if seeds is None:
            continue
        (s1, why1), (s2, why2) = seeds
        r = c.rules
        rows.append(dict(season=season, conference=conf, seed1=s1, seed2=s2,
                         seed1_record=_fmt(c, s1), seed2_record=_fmt(c, s2),
                         site=("at " + s1) if r["site"] == "seed1_home" else "neutral",
                         home_team=s1 if r["site"] == "seed1_home" else None,
                         seed1_why=why1, seed2_why=why2))
    return pd.DataFrame(rows)


def seeds_for(c):
    """[(seed 1, why), (seed 2, why)] for one conference context, or None if undecidable yet."""
    r = c.rules
    if r["format"] == "divisions":
        winners = []
        for dv in sorted({d for d in c.div.values() if d}):
            (t, why), = _seed([t for t in c.eligible if c.div[t] == dv], 1, c)
            winners.append((t, f"{dv} division - {why}"))
        if len(winners) < 2:
            return None                                  # a division has no conference results yet
        a, b = winners[0][0], winners[1][0]
        if abs(c.pct[a] - c.pct[b]) > 1e-9:
            first = a if c.pct[a] > c.pct[b] else b
            host_why = "better conference record"
        else:
            first, trail = pick_best([a, b], c)
            host_why = "same record: " + "; then ".join(trail)
        seeds = sorted(winners, key=lambda x: x[0] != first)
        seeds[0] = (seeds[0][0], seeds[0][1] + f" | hosts: {host_why}")
    else:
        seeds = _seed(c.eligible, 2, c)
    return seeds if len(seeds) == 2 else None
