"""Shared CFB definitions - the SINGLE source of truth for rules every notebook uses.

Import this module instead of re-implementing any of these. A second copy of a
rule that silently drifts is how leaks and mismatches get back in.

    import sys
    sys.path.insert(0, str(BASE_DIR))
    import cfb_common as cc

Contents
  1. Power conferences
  2. Conference championship games
  3. ALIGNMENT RULE - which games a CFP (or AP) poll has seen
"""
import datetime as _dt

import pandas as pd

# =============================================================================
# 1. Power conferences
# =============================================================================
# Power Five through 2023 (incl. Pac-12); Power Four from 2024 after the
# Pac-12 collapsed. Notre Dame is treated as power.
POWER_TO_2023 = {"ACC", "Big Ten", "Big 12", "SEC", "Pac-12"}
POWER_FROM_2024 = {"ACC", "Big Ten", "Big 12", "SEC"}


def power_conferences(season):
    return POWER_TO_2023 if season <= 2023 else POWER_FROM_2024


def is_power_conf(conf, team, season):
    """True if `team` (in conference `conf`) is a power-conference team in `season`."""
    if team == "Notre Dame":
        return True
    return conf in power_conferences(season)


# =============================================================================
# 2. Conference championship games
# =============================================================================
# CFBD's `notes` field ("SEC Championship", ...) is only populated from 2022 on,
# so for 2008-2021 the championship game is identified by rule:
#
#   A conference's championship game is its LAST same-conference regular-season
#   game of the season, for conferences that HELD a championship game that
#   season (fixed history below). Army-Navy is excluded explicitly: both teams
#   are in the AAC from 2024 and it is played the week AFTER championship week.
#
# VALIDATED: on 2022-2025 the rule reproduces the notes exactly (38 of 38, no
# extras); every season's count matches the table below; every game falls on
# championship weekend. From 2022 on (incl. the in-progress season, where the
# rule would wrongly fire mid-season) the notes are used directly.
NOTES_FROM_SEASON = 2022

# (season, conference) -> the two teams that actually played the championship game
CCG_OVERRIDES = {
    (2018, "Pac-12"): ("Washington", "Utah"),     # Cal-Stanford moved to Dec 1 (wildfire smoke)
    (2021, "Pac-12"): ("Utah", "Oregon"),         # Cal-USC moved to Dec 4 (COVID postponement)
}


def ccg_conferences(season):
    """FBS conferences that held a championship game in `season`."""
    c = {"ACC", "SEC", "Conference USA", "Mid-American"}
    if season <= 2010 or season >= 2017:
        c.add("Big 12")                 # no Big 12 championship 2011-2016
    if season >= 2011:
        c |= {"Big Ten", "Pac-12"}      # both began in 2011
    if season >= 2013:
        c.add("Mountain West")
    if season >= 2015:
        c.add("American Athletic")
    if season >= 2018:
        c.add("Sun Belt")
    if 2024 <= season <= 2025:
        c.discard("Pac-12")             # two-team Pac-12, no title game; it returns in 2026 (8 teams)
    return c


def flag_conference_championships(games):
    """Boolean Series aligned to `games`: True for FBS conference championship games.

    `games` needs: id, season, seasonType, startDate, homeTeam, awayTeam,
    homeConference, awayConference, notes, homePoints.
    """
    g = games
    reg = ((g["seasonType"] == "regular") & g["homePoints"].notna()
           & (g["homeClassification"] == "fbs"))
    same = (g["homeConference"] == g["awayConference"]) & g["homeConference"].notna()
    army_navy = ((g["homeTeam"].isin(["Army", "Navy"]))
                 & (g["awayTeam"].isin(["Army", "Navy"])))
    flag = pd.Series(False, index=g.index)

    # 2022+ : CFBD notes (ground truth)
    by_notes = (reg & same & (g["season"] >= NOTES_FROM_SEASON)
                & g["notes"].fillna("").str.contains("Championship"))
    flag |= by_notes

    # 2008-2021 : rule
    cand = g[reg & same & ~army_navy & (g["season"] < NOTES_FROM_SEASON)].copy()
    cand = cand[[c in ccg_conferences(s)
                 for c, s in zip(cand["homeConference"], cand["season"])]]
    cand = cand.assign(_d=pd.to_datetime(cand["startDate"], utc=True))
    last = cand.sort_values("_d").groupby(["season", "homeConference"]).tail(1)
    flag.loc[last.index] = True

    # Known exceptions to the rule: a postponed regular-season game was played on (or after)
    # championship weekend, so it - not the title game - was the conference's last game.
    # Found by cfb_standings' replay (our computed participants disagreed with the flag).
    for (season, conf), pair in CCG_OVERRIDES.items():
        in_conf = (g["season"] == season) & (g["homeConference"] == conf) & (g["awayConference"] == conf)
        flag.loc[in_conf & reg] = False
        m = cand[(cand["season"] == season) & cand["homeTeam"].isin(pair) & cand["awayTeam"].isin(pair)]
        assert len(m) >= 1, (season, conf, pair)
        flag.loc[m.sort_values("_d").index[-1]] = True       # their last meeting = the title game
    return flag


# =============================================================================
# 3. ALIGNMENT RULE  (verified; binding for every join between a poll and games)
# =============================================================================
# CFBD labels each CFP poll AND each AP poll by the UPCOMING week. The poll
# labelled "week N" was built from games through GAME WEEK N-1.
#   Proof: Georgia lost to Ole Miss on Sat Nov 9 2024 (game week 11). The polls
#   labelled week 11 still have Georgia #3 (CFP) / #2 (AP); the polls labelled
#   week 12 drop them to #12 / #11.
# Joining poll week N to games through game week N leaks a week of results the
# pollsters had not seen.
#
# Do NOT hardcode "week 14" for championship weekend: it is game week 14 in some
# seasons (2021, 2023) and game week 15 in others (2019, 2024, 2025).
def game_week_for_poll(poll_week):
    """Last GAME week a poll labelled `poll_week` has seen."""
    return int(poll_week) - 1


def poll_week_for_game_week(game_week):
    """The poll that first sees games through `game_week`."""
    return int(game_week) + 1


def build_week_dates(games):
    """(season, week) -> first and last local game date, regular season only.

    `games` needs: season, week, seasonType, startDate.
    """
    g = games[games["seasonType"] == "regular"].copy()
    g["gdate"] = (pd.to_datetime(g["startDate"], utc=True)
                  .dt.tz_convert("America/New_York").dt.date)
    return g.groupby(["season", "week"])["gdate"].agg(["min", "max"])


def poll_cutoff_date(season, poll_week, week_dates):
    """Last game DATE the poll labelled `poll_week` has seen.

    = last date of game week N-1, capped at that week's first date + 6 days.
    The cap excludes Army-Navy, played the Saturday AFTER Selection Day but
    which CFBD sometimes files in the same game week as the conference
    championships (e.g. 2019 game week 15 runs Dec 6 to Dec 14).
    """
    lo, hi = week_dates.loc[(season, game_week_for_poll(poll_week))]
    return min(hi, lo + _dt.timedelta(days=6))
