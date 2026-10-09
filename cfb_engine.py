"""cfb_engine - the live engine: game results in, every model input out.

ONE implementation of "results -> Elo -> resume features", shared by the pipeline checks
and the app. The app must compute a team's inputs exactly the way the model was trained on
them; `check_cfb_engine` proves that by reproducing data/cfb_elo.xlsx and
data/master/team_week_resume.xlsx from the raw games.

    import cfb_engine as eng
    P   = eng.load_elo_params(DATA_DIR)
    g   = eng.prepare_games(games, picks={game_id: True/False})   # True = home team wins
    ge, ratings = eng.run_elo(g, P)                               # per-game Elo, before/after
    tg  = eng.team_games(g, ge)                                   # one row per FBS team per game
    f   = eng.features_at(tg, season, cutoff_date, P, is_selection_day=False)

Everything is win/loss only - no margins. A pick is just "home team won" True/False.

Logic mirrors build_cfb_elo.run_elo and build_cfb_resume CELL 2 / CELL 5. If either of
those changes, re-run check_cfb_engine.
"""
import math

import numpy as np
import pandas as pd

import cfb_common as cc

SCALE = 400.0
INIT_ELO = 1500.0
FCS_KEY = "__FCS__"
ELO_TREND_WEIGHTS = [1.0, 0.5, 0.25]      # most recent first: 3-game weighted mean of Elo changes


# =============================================================================
# 1. Parameters and game preparation
# =============================================================================
def load_elo_params(data_dir):
    """The frozen Elo parameters, from the sheet build_cfb_elo writes."""
    p = pd.read_excel(data_dir / "cfb_elo.xlsx", sheet_name="parameters").set_index("parameter")["tuned"]
    return {k: float(p[k]) for k in ["K", "HFA", "ALPHA", "CROSS_CONF", "FCS_ELO", "INIT_GAP"]}


def prepare_games(games, picks=None):
    """Add the columns the engine needs. `games` has the games.xlsx layout.

    homeWon : True/False for a played game (from the score) or a picked game (from `picks`,
              {game id: home team won}); missing when the game has no result yet.
    A real result always wins over a pick - a user cannot change a game already played.
    """
    g = games.copy()
    g["startDate"] = pd.to_datetime(g["startDate"], errors="coerce", utc=True)
    g["gdate"] = g["startDate"].dt.tz_convert("America/New_York").dt.date
    played = g["completed"].fillna(False).astype(bool) & (g["homePoints"] != g["awayPoints"])
    hw = pd.Series(np.where(played, g["homePoints"] > g["awayPoints"], np.nan), index=g.index, dtype=object)
    hw[~played] = np.nan
    if picks:
        pick = g["id"].map(picks)
        use = ~played & pick.notna()
        hw[use] = pick[use].astype(bool)
    g["homeWon"] = hw
    g["hasResult"] = g["homeWon"].notna()
    g["isPick"] = g["hasResult"] & ~played
    g["homeKey"] = np.where(g["homeClassification"] == "fbs", g["homeTeam"], FCS_KEY)
    g["awayKey"] = np.where(g["awayClassification"] == "fbs", g["awayTeam"], FCS_KEY)
    g["neutral"] = g["neutralSite"].fillna(False).astype(bool)
    g["confGame"] = g["conferenceGame"].fillna(False).astype(bool)
    if "isConfChampionship" not in g:
        g["isConfChampionship"] = False
    g["isConfChampionship"] = g["isConfChampionship"].fillna(False).astype(bool)
    return g


# =============================================================================
# 2. Elo  (binary win/loss, flat K, regression toward the conference mean each offseason)
# =============================================================================
def conference_map(g):
    """{(season, team): conference} from every game that has a result."""
    seq = g[g["hasResult"]]
    conf = {}
    for s, hk, ak, hc, ac in seq[["season", "homeKey", "awayKey", "homeConference",
                                  "awayConference"]].itertuples(index=False, name=None):
        if hk != FCS_KEY and isinstance(hc, str):
            conf[(s, hk)] = hc
        if ak != FCS_KEY and isinstance(ac, str):
            conf[(s, ak)] = ac
    return conf


def run_elo(g, p, start_ratings=None, start_season=None, conf=None):
    """Walk every game that has a result, in time order.

    Returns (game_elo, ratings): game_elo has one row per game with both teams' rating before
    and after and the home win probability; ratings is the final rating of every team.

    start_ratings / start_season / conf let the app resume from a saved state - the ratings
    and season as they stood after the last game of an earlier run, plus that run's
    conference_map (needed for the offseason regression) - instead of replaying history.
    """
    seq = g[g["hasResult"] & ~((g["homeKey"] == FCS_KEY) & (g["awayKey"] == FCS_KEY))]
    seq = seq.sort_values(["season", "startDate", "id"])
    cols = ["season", "id", "homeKey", "awayKey", "homeWon", "neutral", "confGame",
            "homeConference", "awayConference"]

    # conference membership by season, read from the games (handles realignment)
    conf = dict(conf) if conf else {}
    for s, hk, ak, hc, ac in seq[["season", "homeKey", "awayKey", "homeConference",
                                  "awayConference"]].itertuples(index=False, name=None):
        if hk != FCS_KEY and isinstance(hc, str):
            conf[(s, hk)] = hc
        if ak != FCS_KEY and isinstance(ac, str):
            conf[(s, ak)] = ac

    def regress_group(team, season):
        if team == "Notre Dame":
            return "ACC"            # Notre Dame: de facto ACC for the offseason pull only (fixed 2026-10-09; was a group of one = no pull)
        c = conf.get((season, team))
        if c is None or c == "FBS Independents":
            return "__OTHER_INDEP__"
        return c

    k, hfa, alpha = p["K"], p["HFA"], p["ALPHA"]
    boost, fcs_elo = p["CROSS_CONF"], p["FCS_ELO"]
    start_power = INIT_ELO + p["INIT_GAP"] / 2.0
    start_other = INIT_ELO - p["INIT_GAP"] / 2.0

    ratings = dict(start_ratings) if start_ratings else {}
    cur_season = start_season
    rows = []
    for season, gid, hk, ak, hw, neutral, conf_game, hconf, aconf in seq[cols].itertuples(index=False, name=None):
        if season != cur_season:
            if cur_season is not None and ratings:
                groups = {}
                for t, v in ratings.items():
                    groups.setdefault(regress_group(t, cur_season), []).append(v)
                means = {grp: sum(v) / len(v) for grp, v in groups.items()}
                for t in ratings:
                    ratings[t] = alpha * ratings[t] + (1 - alpha) * means[regress_group(t, cur_season)]
            cur_season = season

        if hk == FCS_KEY:
            rh = fcs_elo
        else:
            rh = ratings.get(hk, start_power if cc.is_power_conf(hconf, hk, season) else start_other)
        if ak == FCS_KEY:
            ra = fcs_elo
        else:
            ra = ratings.get(ak, start_power if cc.is_power_conf(aconf, ak, season) else start_other)

        adj = 0.0 if neutral else hfa
        e_home = 1.0 / (1.0 + 10.0 ** ((ra - (rh + adj)) / SCALE))
        mult = boost if (not conf_game and hconf and aconf and hconf != aconf) else 1.0
        delta = k * mult * ((1.0 if hw else 0.0) - e_home)
        if hk != FCS_KEY:
            ratings[hk] = rh + delta
        if ak != FCS_KEY:
            ratings[ak] = ra - delta                     # the pooled FCS rating never moves
        rows.append((season, gid, hk, ak, rh, ra, ratings.get(hk, rh), ratings.get(ak, ra), e_home, bool(hw)))

    game_elo = pd.DataFrame(rows, columns=["season", "gameId", "homeKey", "awayKey", "homePreElo",
                                           "awayPreElo", "homePostElo", "awayPostElo", "homeWinProb",
                                           "homeWon"])
    return game_elo, ratings


def win_prob(elo_home, elo_away, neutral, p):
    """Home team's win probability - what the 'simulate' buttons draw from."""
    adj = 0.0 if neutral else p["HFA"]
    return 1.0 / (1.0 + 10.0 ** ((elo_away - (elo_home + adj)) / SCALE))


# =============================================================================
# 3. One row per FBS team per regular-season game
# =============================================================================
def team_games(g, game_elo):
    """Regular-season games with a result, seen from each FBS team's side."""
    d = g[(g["seasonType"] == "regular") & g["hasResult"]]
    d = d.merge(game_elo[["gameId", "homePreElo", "awayPreElo", "homePostElo", "awayPostElo", "homeWinProb"]],
                left_on="id", right_on="gameId", how="inner")

    def side(df, s):
        o = "away" if s == "home" else "home"
        hw = df["homeWon"].astype(bool)
        out = pd.DataFrame({
            "season": df["season"], "week": df["week"], "gdate": df["gdate"], "gameId": df["id"],
            "team": df[f"{s}Team"], "conference": df[f"{s}Conference"],
            "team_class": df[f"{s}Classification"],
            "opponent": df[f"{o}Team"], "opp_conference": df[f"{o}Conference"],
            "opp_is_fcs": df[f"{o}Classification"] != "fbs",
            "elo_pre": df[f"{s}PreElo"], "opp_elo_pre": df[f"{o}PreElo"], "elo_post": df[f"{s}PostElo"],
            "p_win": df["homeWinProb"] if s == "home" else 1 - df["homeWinProb"],
            "win": hw if s == "home" else ~hw,
            "is_ccg": df["isConfChampionship"], "is_pick": df["isPick"],
            "conf_flag": df["confGame"],          # CFBD: counts in the conference standings
        })
        out["site"] = np.where(df["neutral"], "neutral", s)
        return out

    tg = pd.concat([side(d, "home"), side(d, "away")], ignore_index=True)
    tg = tg[tg["team_class"] == "fbs"].drop(columns="team_class")
    tg["elo_change"] = tg["elo_post"] - tg["elo_pre"]
    return tg.sort_values(["season", "team", "gdate"]).reset_index(drop=True)


# =============================================================================
# 4. Resume features at a cutoff date
# =============================================================================
def pb_tail(ps, k):
    """P(at least k wins) for independent games with win probabilities ps."""
    dp = np.zeros(len(ps) + 1)
    dp[0] = 1.0
    for q in ps:
        new = dp * (1 - q)
        new[1:] += dp[:-1] * q
        dp = new
    return float(dp[k:].sum())


def _p_ref_wins(ref_elo, opp_elo, site, hfa):
    adj = hfa if site == "home" else (-hfa if site == "away" else 0.0)
    return 1.0 / (1.0 + 10.0 ** ((opp_elo - (ref_elo + adj)) / SCALE))


def _ewma_last(changes, weights=ELO_TREND_WEIGHTS):
    last = list(changes)[-len(weights):][::-1]
    if not last:
        return np.nan
    w = np.array(weights[:len(last)])
    return float(np.dot(w, last) / w.sum())


def features_at(tg, season, cutoff, p, is_selection_day=False):
    """Every team's resume using only games on or before `cutoff` (a date).

    Opponent strength is judged AT THE CUTOFF: a win over a team that has since collapsed is
    no longer a quality win, and strength of record uses opponents' current Elo. So every
    team is recomputed each week whether or not it played.
    """
    hfa, fcs_elo = p["HFA"], p["FCS_ELO"]
    season_tg = tg[tg["season"] == season]
    d = season_tg[season_tg["gdate"] <= cutoff]
    teams = sorted(season_tg["team"].unique())
    team_conf = season_tg.groupby("team")["conference"].agg(
        lambda x: x.mode().iloc[0] if len(x.dropna()) else None)

    elo_at = d.groupby("team")["elo_post"].last()
    pre = season_tg.groupby("team")["elo_pre"].first()
    elo_at = elo_at.reindex(teams).fillna(pre.reindex(teams))
    elo_rank = elo_at.rank(ascending=False, method="min")
    ref_elo = float(elo_at[elo_rank[elo_rank <= 25].index].mean())     # the "average top-25 team"

    dd = d.copy()
    dd["opp_elo_cut"] = np.where(dd["opp_is_fcs"], fcs_elo, dd["opponent"].map(elo_at))
    dd["opp_rank_elo"] = np.where(dd["opp_is_fcs"], np.nan, dd["opponent"].map(elo_rank))
    by_team = dict(tuple(dd.groupby("team")))

    rows = []
    for tm in teams:
        t = by_team.get(tm, dd.iloc[0:0])
        wins = int(t["win"].sum())
        losses = int((~t["win"].astype(bool)).sum())
        ps = [_p_ref_wins(ref_elo, o, st, hfa) for o, st in zip(t["opp_elo_cut"], t["site"])]
        conf = team_conf.get(tm)
        row = {
            "season": season, "team": tm, "conference": conf,
            "is_power": cc.is_power_conf(conf, tm, season),
            "is_independent": conf == "FBS Independents",
            "games_played": len(t), "wins": wins, "losses": losses,
            "elo": float(elo_at[tm]), "elo_rank": int(elo_rank[tm]),
            "elo_trend": _ewma_last(t["elo_change"]),
            "sor_elo": max(0.0, -math.log(min(1.0, max(pb_tail(ps, wins), 1e-12)))) if len(t) else np.nan,
            "qual_win_top10_elo": int((t["win"].astype(bool) & (t["opp_rank_elo"] <= 10)).sum()),
            "qual_win_11_25_elo": int((t["win"].astype(bool) & t["opp_rank_elo"].between(11, 25)).sum()),
        }
        if is_selection_day:
            ccg = t[t["is_ccg"]]
            row["ccg_status"] = "none" if ccg.empty else ("won" if bool(ccg["win"].iloc[-1]) else "lost")
        else:
            row["ccg_status"] = "not_selection_day"
        rows.append(row)
    f = pd.DataFrame(rows)

    # conference strength = mean Elo of the conference's members now (independents: all-FBS mean)
    conf_mean = f[~f["is_independent"]].groupby("conference")["elo"].mean()
    f["conf_elo_mean"] = f["conference"].map(conf_mean).where(~f["is_independent"]).fillna(f["elo"].mean())
    return f.sort_values("elo_rank").reset_index(drop=True)
