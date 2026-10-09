"""CFP committee simulator - local Streamlit app.

    run_app.bat                                   (double-click), or
    python -m streamlit run app.py --server.address localhost

All logic lives in cfb_season.py (and the modules it uses). This file is only the screens.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import cfb_standings as stn
import cfb_season as cs
import cfb_forecast as fc

st.set_page_config(page_title="CFP Committee Simulator", layout="wide")

# every coefficient of the model, in plain words
FEATURE_TEXT = {
    "loss_ge1": "has at least 1 loss", "loss_ge2": "has at least 2 losses", "loss_ge3": "has at least 3 losses",
    "loss_ge4": "has at least 4 losses", "loss_ge5": "has 5 or more losses",
    "elo": "Elo rating", "sor_log": "strength of record, as log(SOR + 0.05)", "is_power": "Power 4 team (or Notre Dame)",
    "qual_win_top10_elo": "wins over teams now in the Elo top 10",
    "qual_win_11_25_elo": "wins over teams now ranked 11-25 by Elo",
    "elo_trend_later": "Elo trend, last 3 games (counts from release 2)",
    "ccg_won": "won its conference championship game (Selection Day)",
    "ccg_lost": "lost its conference championship game (offsets the extra loss - Selection Day)",
    "ccg_won_x_conf": "won the title game x (conference Elo mean - 1500)",
    "prior_real_in25": "in the top 25 of the real previous CFP poll",
    "prior_real_pts": "points in the real previous CFP poll (26 - rank)",
    "prior_A_pts": "points in OUR previous predicted poll (no real poll exists yet)",
    "prior_B_pts": "points in OUR previous predicted poll (a real poll exists, 2+ releases old)",
    "anchor_pts": "points in the last real CFP poll (anchor)",
}


@st.cache_resource(show_spinner="Loading data, the model and Elo through last season (about 20 seconds, once)...",
                   max_entries=1)
def get_base(data_stamp):
    """Loaded once per version of the data. `data_stamp` (each data file's size and modification
    time) changes whenever a weekly update is published, so the server reloads on its own instead
    of serving the copy it loaded at start-up; max_entries=1 drops the old copy."""
    b = cs.load_base(BASE_DIR)
    b.plan = fc.build_plan(b)
    return b


def data_stamp():
    files = sorted((BASE_DIR / "data").glob("*.xlsx")) + sorted((BASE_DIR / "data" / "master").glob("*.json")) + \
        sorted((BASE_DIR / "winprob" / "data" / "app").glob("*.csv"))
    return tuple((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files if f.exists())


N_SIMS = 1000                                  # simulated seasons per forecast


base = get_base(data_stamp())
WP_ASOF = f" ({base.wp.as_of})" if base.wp is not None and base.wp.as_of else ""
ss = st.session_state
for k in ("picks", "ccg_picks", "po_picks"):
    ss.setdefault(k, {})
ss.setdefault("undo", None)
ss.setdefault("fc_cache", {})


def snapshot():
    ss.undo = (dict(ss.picks), dict(ss.ccg_picks), dict(ss.po_picks))


state = cs.build_state(base, ss.picks, ss.ccg_picks, ss.po_picks)
G = state["games"]
RI = state["release_info"].set_index("release")
now = state["resume_now"].set_index("team")
SD = state["standings"]
REC = {t: f"{w}-{l}" for t, w, l in zip(SD["team"], SD["w"], SD["l"])}        # includes title games
shown = sorted(state["preds"])                                # releases that have a poll

# ---- the forecast: only re-run when a whole week (or all ten title games) gets a result ----
f_picks, f_ccg, f_fixed, f_label, f_key = cs.forecast_inputs(base, state, ss.ccg_picks)
if f_key not in ss.fc_cache:
    with st.spinner(f"Updating the forecast: {N_SIMS} simulated seasons from {f_label}..."):
        if len(ss.fc_cache) > 40:
            ss.fc_cache.clear()
        ss.fc_cache[f_key] = fc.summarize(base.plan, fc.run(base, base.plan, f_picks, f_ccg, None, n_sims=N_SIMS,
                                                            seed=0, fixed_matchups=f_fixed))
FC = ss.fc_cache[f_key].set_index("team")
n_ignored = int(sum(1 for i in state["picks"] if i not in f_picks))
FC_NOTE = (f"Forecast: {N_SIMS} simulated seasons played out from {f_label}."
           + (f" {n_ignored} pick(s) in weeks that are not complete yet are not in it - it updates when the week is "
              "fully picked." if n_ignored else ""))
latest = state["preds"][shown[-1]].set_index("team") if shown else None
# the poll shown next to teams: the real CFP poll for that release if it exists, else ours
POLL_RANK, POLL_LABEL = {}, ""
if shown:
    _j = shown[-1]
    if _j in base.real_polls:
        POLL_RANK, POLL_LABEL = dict(base.real_polls[_j]), f"the real CFP poll ({'Selection Day' if _j == 6 else 'release ' + str(_j)})"
    else:
        POLL_RANK = dict(zip(latest.index, latest["rank"]))
        POLL_LABEL = (f"our predicted {'Selection Day ranking' if _j == 6 else 'poll, release ' + str(_j)}"
                      + (" (provisional)" if state["release_info"].set_index("release").loc[_j, "status"] == "provisional" else ""))


def pct(p):
    return f"{100 * p:.0f}%"


def tag(team):
    """Team with its current poll rank (once a poll exists) and record."""
    if team not in REC:
        return f"{team} (FCS)"
    rk = f"#{int(latest.loc[team, 'rank'])} " if latest is not None and latest.loc[team, "rank"] <= 25 else ""
    return f"{rk}{team} ({REC[team]})"


def release_name(j):
    return "Selection Day" if j == 6 else f"Release {j}"


# =============================================================================
# Sidebar - status and simulation
# =============================================================================
with st.sidebar:
    st.title("CFP Committee Simulator")
    st.caption(f"{cs.SEASON} season. Real results are loaded through week {base.real_through} and cannot be "
               f"changed. Every game has a result through week {state['complete_through']}.")
    n_pick = int((G["status"] == "pick").sum())
    n_open = int((G["status"] == "open").sum())
    c1, c2 = st.columns(2)
    c1.metric("Your picks", n_pick)
    c2.metric("Games open", n_open)
    ws = state["week_status"]
    if not shown:
        left9 = int((ws.loc[ws.index <= 9, "games"] - ws.loc[ws.index <= 9, "with_result"]).sum())
        st.info(f"The predicted poll appears once every game through week 9 has a result "
                f"({left9} to go). First release: Nov 3.")
    if state["champion"]:
        st.success(f"Champion: {state['champion']}")

    st.subheader("Simulate one season")
    st.caption("One random season, not a forecast: each open game's winner is drawn once from the game model's "
               "win probability (Elo for games against FCS teams), week by week. Your own picks are kept. Press "
               "again for a different season.")
    open_weeks = sorted(int(w) for w in G.loc[(G["status"] == "open") & (G["stage"] == "regular"), "week"].unique())
    targets = {f"Week {w}": w for w in open_weeks}
    if ((G["stage"] == "championship") & (G["status"] == "open")).any():
        targets["Championship games"] = cs.CCG_WEEK
    if ((G["stage"] == "playoff") & (G["status"] == "open")).any():
        targets["Next playoff round"] = "playoff"
    if targets:
        which = st.selectbox("Simulate one stage", list(targets))
        if st.button("Simulate this stage", width="stretch"):
            snapshot()
            ss.picks, ss.ccg_picks, ss.po_picks = cs.simulate(base, ss.picks, ss.ccg_picks, ss.po_picks, "week",
                                                               np.random.default_rng(), week=targets[which])
            st.rerun()
    if n_open or not state["champion"]:
        if st.button("Simulate the rest of the season", type="primary", width="stretch"):
            snapshot()
            with st.spinner("Simulating through the playoff..."):
                ss.picks, ss.ccg_picks, ss.po_picks = cs.simulate(base, ss.picks, ss.ccg_picks, ss.po_picks, "rest",
                                                                   np.random.default_rng())
            st.rerun()
    st.divider()
    if ss.undo and st.button("Undo last simulation / clear", width="stretch"):
        ss.picks, ss.ccg_picks, ss.po_picks = ss.undo
        ss.undo = None
        st.rerun()
    if (ss.picks or ss.ccg_picks or ss.po_picks) and st.button("Clear all my picks", width="stretch"):
        snapshot()
        ss.picks, ss.ccg_picks, ss.po_picks = {}, {}, {}
        st.rerun()
    st.caption(f"Model: {base.M['version']}")


# =============================================================================
# Pick widgets
# =============================================================================
def _on_pick(kind, key, ident, home):
    v = ss.get(key)
    if kind == "regular":
        if v is None:
            ss.picks.pop(ident, None)
        else:
            ss.picks[ident] = (v == home)
    else:
        target = ss.ccg_picks if kind == "ccg" else ss.po_picks
        if v is None:
            target.pop(ident, None)
        else:
            target[ident] = v


def _clear_pick(kind, ident):
    {"regular": ss.picks, "ccg": ss.ccg_picks, "po": ss.po_picks}[kind].pop(ident, None)


def game_row(r, kind, ident):
    """One game: matchup, the game model's spread and win probability, and the pick control (or the result)."""
    c1, c2, c3 = st.columns([5, 3, 4], vertical_alignment="center")
    c1.markdown(f"{tag(r.awayTeam)} &nbsp;at&nbsp; **{tag(r.homeTeam)}**" if not r.neutral
                else f"{tag(r.awayTeam)} &nbsp;vs&nbsp; {tag(r.homeTeam)} &nbsp;*(neutral)*")
    fav, p = (r.homeTeam, r.p_home) if r.p_home >= 0.5 else (r.awayTeam, 1 - r.p_home)
    spread = "" if pd.isna(r.home_margin) else f" by {abs(r.home_margin):.1f}"
    c2.progress(float(r.p_home), text=f"{fav}{spread} · {pct(p)}")
    if r.status == "final":
        score = "" if pd.isna(r.homePoints) else f" {int(r.awayPoints)}-{int(r.homePoints)}"
        c3.markdown(f"Final{score}: **{r.winner}**")
        return
    key = f"pick_{r.id}"
    ss[key] = r.winner if r.status == "pick" else None
    ctl, clr = c3.columns([6, 1], vertical_alignment="center")
    ctl.segmented_control("pick", [r.awayTeam, r.homeTeam], key=key, label_visibility="collapsed",
                          on_change=_on_pick, args=(kind, key, ident, r.homeTeam))
    if r.status == "pick":
        clr.button("x", key=f"clear_{r.id}", help="Clear this pick", on_click=_clear_pick, args=(kind, ident))


(t_sched, t_rank, t_fc, t_stand, t_team, t_ccg, t_brk, t_mu, t_how) = st.tabs(
    ["Schedule", "Rankings", "Forecast", "Standings", "Team", "Championship games", "Bracket",
     "Matchups & power ranking", "How it works"])

# =============================================================================
# Schedule  (regular season weeks + week 14 = the championship games)
# =============================================================================
with t_sched:
    sched = G[G["stage"].isin(["regular", "championship"])]
    weeks = sorted(int(w) for w in sched["week"].unique())
    first_open = next((w for w in weeks if (sched.loc[sched["week"] == w, "status"] == "open").any()), weeks[-1])

    def week_name(w):
        if w == cs.CCG_WEEK:
            return "Week 14 - conference championship games"
        return f"Week {w}" + (" (played)" if w <= base.real_through else "")

    a, b_, c = st.columns([3, 3, 3])
    wk = a.selectbox("Week", weeks, index=weeks.index(first_open), format_func=week_name)
    confs = sorted({base.conf_of[t] for t in base.fbs_teams if isinstance(base.conf_of.get(t), str)})
    pick_conf = b_.multiselect("Conference", confs, placeholder="All conferences")
    only_top = c.toggle("Only games with an Elo top-25 team", value=False)
    d = sched[sched["week"] == wk]
    if pick_conf:
        d = d[d["homeTeam"].map(base.conf_of).isin(pick_conf) | d["awayTeam"].map(base.conf_of).isin(pick_conf)]
    if only_top:
        top = set(now.index[now["elo_rank"] <= 25])
        d = d[d["homeTeam"].isin(top) | d["awayTeam"].isin(top)]
    done = int((d["status"] != "open").sum())
    st.caption(f"{week_name(wk)}: {done} of {len(d)} games shown have a result. Home team in bold. The bar is the "
               "game model's prediction: the favourite, the predicted margin (the spread) and the win probability "
               "that margin implies. Played games show the prediction made before kickoff; games not yet played "
               f"use every team's inputs as of the last update{WP_ASOF}. Games against FCS teams use Elo (no spread). "
               "Click a team to pick it, the x to clear a pick.")
    if wk == cs.CCG_WEEK:
        for r in d.itertuples(index=False):
            st.markdown(f"**{r.ccg_conference}**")
            game_row(r, "ccg", r.ccg_conference)
        st.caption("How each team got here is on the Championship games tab.")
    else:
        for r in d.itertuples(index=False):
            game_row(r, "regular", r.id)
    if d.empty:
        st.write("No games match the filters.")
    if cs.CCG_WEEK not in weeks:
        st.caption("Week 14 (the ten conference championship games) appears here once every game through "
                   "week 13 has a result.")

# =============================================================================
# Rankings
# =============================================================================
RESUME_COLS = {"elo": st.column_config.NumberColumn("Elo", format="%.0f"),
               "elo_rank": st.column_config.NumberColumn("Elo rank", format="%d"),
               "sor_elo": st.column_config.NumberColumn("strength of record", format="%.2f",
                                                        help="Higher = harder for an average top-25 team to match. "
                                                             "Shown raw because it is easier to read; the model "
                                                             "itself uses log(strength of record + 0.05)."),
               "qual_win_top10_elo": st.column_config.NumberColumn("wins v top 10", format="%d"),
               "qual_win_11_25_elo": st.column_config.NumberColumn("wins v 11-25", format="%d"),
               "elo_trend": st.column_config.NumberColumn("Elo trend", format="%+.1f"),
               "is_power": st.column_config.CheckboxColumn("Power 4"),
               "conf_elo_mean": st.column_config.NumberColumn(
                   "conf. Elo mean", format="%.0f",
                   help="Mean Elo of the team's conference right now. Blank for independents (they have no "
                        "conference); the model only uses it for a team that wins its title game.")}


def blank_independents(df):
    """Independents have no conference: show nothing instead of the all-FBS placeholder."""
    df = df.copy()
    df["conf_elo_mean"] = df["conf_elo_mean"].where(df["conference"] != "FBS Independents")
    return df

with t_rank:
    if not shown:
        st.info("No poll yet. The model was trained on committee polls, the first of which comes after week 9, "
                "so the predicted poll and model scores are shown only once every game through week 9 has a "
                "result. Below: every team's resume inputs as they stand now.")
        st.subheader(f"Resume inputs as of {state['as_of']} (sorted by Elo - not the committee model)")
        t = blank_independents(state["resume_now"])
        t["record"] = t["team"].map(REC)
        st.dataframe(t[["elo_rank", "team", "record", "conference", "elo", "sor_elo", "qual_win_top10_elo",
                        "qual_win_11_25_elo", "elo_trend", "is_power", "conf_elo_mean"]],
                     hide_index=True, width="stretch", height=920, column_config=RESUME_COLS)
    else:
        def lab(j):
            i = RI.loc[j]
            s = "Selection Day - after the championship games" if j == 6 else f"Release {j} - after week {int(i['week'])}"
            if i["status"] == "provisional":
                s += f"  (PROVISIONAL: {int(i['games_left'])} games not picked)"
            return s
        j = st.selectbox("Release", shown, index=len(shown) - 1, format_func=lab)
        P_ = state["preds"][j]
        sit = P_["situation"].iloc[0]
        how = {"first": "First release: predicted cold from the resumes (no prior poll).",
               "real": "Prior = the real CFP poll from the week before.",
               "A": "Prior = our own predicted poll from the week before (no real poll exists yet).",
               "B": "Prior = our own previous prediction, anchored to the last real CFP poll."}[sit]
        st.caption(how + ("  Provisional: it will change as the remaining games of this week are picked."
                          if RI.loc[j, "status"] == "provisional" else ""))
        rg = cs.ranges(base, state).get(j).drop(columns=["p_playoff", "p_bye"], errors="ignore")
        T = P_.merge(rg, on="team", how="left").merge(FC[["p_playoff", "p_bye", "p_champion"]], left_on="team",
                                                      right_index=True, how="left")
        T = blank_independents(T)
        if j - 1 in state["preds"]:
            prev = state["preds"][j - 1].set_index("team")["rank"]
            T["move"] = (T["team"].map(prev) - T["rank"]).fillna(0).astype(int)
        else:
            T["move"] = 0
        T["record"] = T["wins"].astype(str) + "-" + T["losses"].astype(str)
        T["range"] = T["range_lo"].round().astype(int).astype(str) + "-" + T["range_hi"].round().astype(int).astype(str)
        T["prior"] = T["prior_rank"].map(lambda v: "" if pd.isna(v) else str(int(v)))
        if j in base.real_polls:
            T["real poll"] = T["team"].map(base.real_polls[j])
        n_show = st.radio("Show", [25, 40, len(T)], horizontal=True,
                          format_func=lambda n: f"Top {n}" if n < len(T) else "All teams")
        cols = (["rank", "team", "record", "conference", "move"] + (["real poll"] if j in base.real_polls else [])
                + ["score", "gap_to_12th", "range", "p_playoff", "p_bye", "p_champion", "prior", "elo", "elo_rank", "sor_elo",
                   "qual_win_top10_elo", "qual_win_11_25_elo", "elo_trend", "is_power", "conf_elo_mean"]
                + (["ccg_status"] if j == 6 else []))
        cfg = dict(RESUME_COLS)
        cfg.update({
            "move": st.column_config.NumberColumn("move", format="%+d", help="Places gained since the previous release"),
            "score": st.column_config.NumberColumn("model score", format="%.2f",
                                                   help="The model's committee score. Only comparable WITHIN a release: "
                                                        "release 1 has no prior-poll term, so every score is lower there."),
            "gap_to_12th": st.column_config.NumberColumn("vs #12 line", format="%+.2f",
                                                         help="Score minus the 12th team's score"),
            "range": st.column_config.TextColumn("range in this poll", help="Where the committee would rank the team in "
                                                 "THIS poll 8 times out of 10 (10th-90th percentile of 500 simulated polls)"),
            "prior": st.column_config.TextColumn("prior rank", help="Rank in the prior poll the model used (blank = unranked / none)"),
            "ccg_status": st.column_config.TextColumn("title game"),
            "p_playoff": st.column_config.ProgressColumn("P(make playoff)", format="percent", min_value=0, max_value=1,
                                                         help="Forecast: share of simulated seasons in the 12-team field"),
            "p_bye": st.column_config.ProgressColumn("P(bye)", format="percent", min_value=0, max_value=1,
                                                     help="Forecast: share of simulated seasons seeded 1-4"),
            "p_champion": st.column_config.ProgressColumn("P(title)", format="percent", min_value=0, max_value=1)})
        st.dataframe(T.head(n_show)[cols], hide_index=True, width="stretch",
                     height=min(35 * n_show + 40, 920), column_config=cfg)
        st.caption("P(make playoff), P(bye) and P(title) are the season FORECAST (end of season, after the bracket "
                   "rules) - see the Forecast tab. " + FC_NOTE)
        st.caption("'Range in this poll' is about this week's poll only: 500 simulated committees (100 re-estimated "
                   "versions of the model x 5 draws of the committee's own randomness) given the results so far. "
                   "Same picks always give the same numbers.")

# =============================================================================
# Forecast
# =============================================================================
with t_fc:
    st.subheader(f"Season forecast as of {f_label}")
    st.caption(FC_NOTE + " In each simulated season every open game is drawn from the game model's win probability "
               "(Elo for FCS games; Elo still updates for the committee's inputs), the conference tiebreakers set the "
               "title games, the committee ranks the teams "
               "(with its own randomness), the bracket rules pick the field, and the playoff is played. The same "
               "random draws are reused every time, so when you change results the odds move because of the "
               "results, not because of simulation noise.")
    F = FC.reset_index()
    F["record"] = F["team"].map(REC)
    F["final rank"] = (F["final_rank_median"].round().astype(int).astype(str) + "  (" + F["final_rank_lo"].round().astype(int).astype(str)
                       + "-" + F["final_rank_hi"].round().astype(int).astype(str) + ")")
    a, b_ = st.columns([2, 5])
    view = a.radio("Show", ["Playoff contenders", "All teams"], horizontal=True)
    fconf = b_.multiselect("Conference", sorted({c_ for c_ in F["conference"] if isinstance(c_, str)}),
                           placeholder="All conferences", key="fc_conf")
    if fconf:
        F = F[F["conference"].isin(fconf)]
    if view == "Playoff contenders":
        F = F[(F["p_playoff"] >= 0.005) | (F["p_conf_champ"] >= 0.05)]
    pc = lambda label, help_=None: st.column_config.ProgressColumn(label, format="percent", min_value=0, max_value=1, help=help_)
    st.dataframe(
        F[["team", "record", "conference", "p_playoff", "p_bye", "p_title_game", "p_conf_champ", "p_semifinal", "p_final",
           "p_champion", "avg_seed", "final rank"]],
        hide_index=True, width="stretch", height=min(35 * len(F) + 40, 920),
        column_config={"p_playoff": pc("make playoff", "In the 12-team field after the bracket rules"),
                       "p_bye": pc("bye (seed 1-4)"), "p_title_game": pc("in conf. title game"),
                       "p_conf_champ": pc("win conference"), "p_semifinal": pc("reach semifinal"),
                       "p_final": pc("reach final"), "p_champion": pc("national title"),
                       "avg_seed": st.column_config.NumberColumn("avg seed", format="%.1f",
                                                                 help="Average seed in the seasons it makes the field"),
                       "final rank": st.column_config.TextColumn("Selection Day rank",
                                                                 help="Median committee rank on Selection Day and the 80% range")})
    st.caption(f"With {N_SIMS} seasons a probability near 50% is good to about +/- 3 points, near 10% to about +/- 2. "
               "The forecast updates when a week is completely picked - not on every click - because it takes "
               "several seconds.")

# =============================================================================
# Standings
# =============================================================================
def standings_table(t):
    t = t.copy()
    t["place"] = t["place"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    t["conf"] = t["conf_w"].astype(str) + "-" + t["conf_l"].astype(str)
    t["overall"] = t["w"].astype(str) + "-" + t["l"].astype(str)
    t["Elo rank"] = t["team"].map(now["elo_rank"])
    cols = ["place", "team", "conf", "overall"]
    if latest is not None:                                  # once a poll exists, show the CFP rank too
        t["CFP rank"] = t["team"].map(lambda x: str(int(POLL_RANK[x])) if POLL_RANK.get(x, 99) <= 25 else "")
        cols.append("CFP rank")
    st.dataframe(t[cols + ["Elo rank"]], hide_index=True, width="stretch", height=35 * len(t) + 40,
                 column_config={"CFP rank": st.column_config.TextColumn(
                     "CFP rank", help=f"Top 25 of {POLL_LABEL} (blank = not ranked)")})


with t_stand:
    conf_list = [c_ for c_ in stn.RULES if c_ in set(SD["conference"])] + ["FBS Independents"]
    cf = st.selectbox("Conference", conf_list)
    t = SD[SD["conference"] == cf]
    if t["division"].notna().any():
        for dv, part in zip(st.columns(t["division"].nunique()), sorted(t["division"].dropna().unique())):
            with dv:
                st.subheader(part)
                standings_table(t[t["division"] == part])
    else:
        standings_table(t)
    st.caption("Order = the conference's own tiebreak procedure, so tied teams appear in the order they would be "
               "seeded. Conference record never includes the title game; the overall record does."
               + (f" CFP rank = {POLL_LABEL}." if latest is not None else " The CFP rank column appears once the "
                  "first poll exists (every game through week 9 has a result)."))
    m = state["matchups"]
    m = m[m["conference"] == cf]
    if len(m):
        r = m.iloc[0]
        st.subheader("Championship game" if state["matchups_final"] else "If the regular season ended now")
        st.markdown(f"**#1 {r['seed1']}** ({r['seed1_record']}) vs **#2 {r['seed2']}** ({r['seed2_record']}) - {r['site']}")
        st.markdown(f"- #1: {r['seed1_why']}\n- #2: {r['seed2_why']}")
        ti = state["tiebreak_inputs"]
        st.caption(f"Where a rule needs a rating we cannot compute, the stand-in is {ti['rating']}. "
                   f"CFP-ranking steps use: {ti['cfp']}.")
    rt = stn.rules_table(cs.SEASON)
    rr = rt[rt["conference"] == cf]
    if len(rr):
        with st.expander("This conference's rules"):
            q = rr.iloc[0]
            st.markdown(f"- **Participants:** {q['participants']}, site: {q['site']}\n"
                        f"- **Two teams tied:** {q['two_team_tie']}\n- **Three or more tied:** {q['three_plus_tie']}\n"
                        f"- {q['note']}\n- *Source: {q['source']}*")

# =============================================================================
# Team
# =============================================================================
with t_team:
    order = list(now.sort_values("elo_rank").index)
    tm = st.selectbox("Team", order)
    r = now.loc[tm]
    CONF_MEAN_TXT = ("none (independent)" if r["conference"] == "FBS Independents" else f"{r['conf_elo_mean']:.0f}")
    c = st.columns(6)
    c[0].metric("Record", REC[tm])
    c[1].metric("Elo", f"{r['elo']:.0f}", f"rank {int(r['elo_rank'])}", delta_color="off")
    c[2].metric("Strength of record", f"{r['sor_elo']:.2f}",
                help="Raw value, easier to read. The model uses log(strength of record + 0.05) - see the inputs table.")
    c[3].metric("Wins vs Elo top 10", int(r["qual_win_top10_elo"]))
    c[4].metric("Wins vs Elo 11-25", int(r["qual_win_11_25_elo"]))
    c[5].metric("Elo trend", f"{r['elo_trend']:+.1f}" if pd.notna(r["elo_trend"]) else "-")
    st.caption(f"{r['conference']} | {'Power 4' if r['is_power'] else 'not Power 4'} | regular-season inputs as of "
               f"{state['as_of']}. Opponents are judged by where they stand NOW, so these move even in a week the "
               "team is idle.")
    fr_ = FC.loc[tm]
    c = st.columns(6)
    c[0].metric("Make playoff", pct(fr_["p_playoff"]))
    c[1].metric("Bye (seed 1-4)", pct(fr_["p_bye"]))
    c[2].metric("Win conference", pct(fr_["p_conf_champ"]))
    c[3].metric("Reach semifinal", pct(fr_["p_semifinal"]))
    c[4].metric("National title", pct(fr_["p_champion"]))
    c[5].metric("Selection Day rank", f"{fr_['final_rank_median']:.0f}",
                f"{fr_['final_rank_lo']:.0f}-{fr_['final_rank_hi']:.0f}", delta_color="off")
    st.caption(FC_NOTE)

    if shown:
        st.subheader("Every model input, and what it is worth")
        if ss.get("_team_latest") != shown[-1] or ss.get("team_release") not in shown:
            ss["team_release"] = shown[-1]                  # a new release appeared: show it
            ss["_team_latest"] = shown[-1]
        jt = st.selectbox("Release", shown, format_func=release_name, key="team_release")
        row = state["preds"][jt].set_index("team").loc[tm]
        beta = dict(zip(base.M["feats"], base.M["beta"]))
        sit_t = row["situation"]

        def used(f):
            """Is this input switched on at this release? (otherwise its value is 0 by design)"""
            if f.startswith("prior_real"):
                return sit_t == "real"
            if f == "prior_A_pts":
                return sit_t == "A"
            if f in ("prior_B_pts", "anchor_pts"):
                return sit_t == "B"
            if f.startswith("ccg_"):
                return jt == 6
            if f == "elo_trend_later":
                return jt > 1
            return True
        why_off = {"first": "release 1 has no prior poll", "real": "the real prior poll is used instead",
                   "A": "no real poll exists yet", "B": "a real poll exists but is 2+ releases old"}[sit_t]
        inp = pd.DataFrame([dict(input=FEATURE_TEXT.get(f, f), value=row["x_" + f], weight=beta[f],
                                 points=row["x_" + f] * beta[f],
                                 note="" if used(f) else ("only on Selection Day" if f.startswith("ccg_") else
                                                          "from release 2" if f == "elo_trend_later" else
                                                          f"not used here: {why_off}"))
                            for f in base.M["feats"]])
        st.dataframe(inp, hide_index=True, width="stretch", height=35 * len(inp) + 40,
                     column_config={"value": st.column_config.NumberColumn("this team's value", format="%.3f"),
                                    "note": st.column_config.TextColumn("switched off?", help="Inputs that do not apply "
                                                                        "at this release are 0 for every team"),
                                    "weight": st.column_config.NumberColumn("model weight", format="%.4f"),
                                    "points": st.column_config.NumberColumn("points = value x weight", format="%+.2f")})
        moved = "" if row["rank"] == row["rank_stage1"] else (
            f" The head-to-head / common-opponent step then moved it from {int(row['rank_stage1'])} to "
            f"{int(row['rank'])}, so on the Rankings tab it carries the score of the place it moved into, "
            f"{row['score']:.2f}.")
        st.markdown(f"**Model score {row['score_stage1']:.2f}** (the points column added up) -> rank "
                    f"**{int(row['rank'])}** in {release_name(jt)}, {row['gap_to_12th']:+.2f} against the #12 line.{moved}")
        st.caption(f"For reading, not model inputs: record at this release {int(row['wins'])}-{int(row['losses'])} | "
                   f"raw strength of record "
                   f"{row['sor_elo']:.3f} | Elo rank {int(row['elo_rank'])} | conference Elo mean "
                   f"{'none (independent)' if row['conference'] == 'FBS Independents' else format(row['conf_elo_mean'], '.0f')} | "
                   f"prior rank used: {'none' if pd.isna(row['prior_rank']) else int(row['prior_rank'])}"
                   + ("" if pd.isna(row["anchor_rank"]) else f" | last real poll: {int(row['anchor_rank'])}")
                   + ". Scores are comparable only within a release.")
        hist = pd.DataFrame([dict(release=release_name(q), status=RI.loc[q, "status"],
                                  rank=int(state["preds"][q].set_index("team").loc[tm, "rank"]),
                                  score=state["preds"][q].set_index("team").loc[tm, "score"],
                                  vs_12_line=state["preds"][q].set_index("team").loc[tm, "gap_to_12th"],
                                  record=f"{int(state['preds'][q].set_index('team').loc[tm, 'wins'])}-"
                                         f"{int(state['preds'][q].set_index('team').loc[tm, 'losses'])}")
                             for q in shown])
        st.subheader("Predicted committee rank by release")
        st.dataframe(hist, hide_index=True, width="stretch",
                     column_config={"score": st.column_config.NumberColumn("model score", format="%.2f"),
                                    "vs_12_line": st.column_config.NumberColumn("vs #12 line", format="%+.2f")})
    else:
        st.subheader("Every model input (values as of now)")
        L = int(r["losses"])
        vals = [("losses", L)] + [(FEATURE_TEXT[f"loss_ge{k}"], int(L >= k)) for k in range(1, 6)] + [
            (FEATURE_TEXT["elo"], round(float(r["elo"]), 1)),
            ("strength of record, raw (for reading only - the model uses the log version below)",
             round(float(r["sor_elo"]), 3)),
            (FEATURE_TEXT["sor_log"], round(float(np.log(r["sor_elo"] + base.M["sor_shift"])), 3)),
            (FEATURE_TEXT["is_power"], int(bool(r["is_power"]))),
            (FEATURE_TEXT["qual_win_top10_elo"], int(r["qual_win_top10_elo"])),
            (FEATURE_TEXT["qual_win_11_25_elo"], int(r["qual_win_11_25_elo"])),
            (FEATURE_TEXT["elo_trend_later"], round(float(r["elo_trend"]), 2) if pd.notna(r["elo_trend"]) else 0.0),
            ("conference Elo mean (used with a title-game win)", CONF_MEAN_TXT),
            ("prior poll / title game inputs", "not defined yet")]
        st.dataframe(pd.DataFrame(vals, columns=["input", "value"]).astype({"value": str}), hide_index=True, width="stretch",
                     height=35 * len(vals) + 40)
        st.caption("The model's weights and the team's score appear here once the first poll exists "
                   "(every game through week 9 has a result).")

    st.subheader("Schedule")
    tg = state["tg"]
    mine = G[(G["homeTeam"] == tm) | (G["awayTeam"] == tm)].copy()
    elo_move = tg[tg["team"] == tm].set_index("gameId")[["elo_pre", "opp_elo_pre", "elo_post"]]
    rows = []
    for g_ in mine.itertuples(index=False):
        home = g_.homeTeam == tm
        opp = g_.awayTeam if home else g_.homeTeam
        p = g_.p_home if home else 1 - g_.p_home
        mg = g_.home_margin if home else -g_.home_margin
        res = "-" if g_.status == "open" else (("W" if g_.winner == tm else "L") + (" (pick)" if g_.status == "pick" else ""))
        wk_lab = {"regular": f"Week {int(g_.week)}", "championship": "Title game", "playoff": "Playoff"}[g_.stage]
        rows.append(dict(when=wk_lab, site="neutral" if g_.neutral else ("home" if home else "away"),
                         opponent=tag(opp), spread=None if pd.isna(mg) else round(float(mg), 1), win_prob=p, result=res,
                         elo_before=elo_move["elo_pre"].get(g_.id), opp_elo_before=elo_move["opp_elo_pre"].get(g_.id),
                         elo_after=elo_move["elo_post"].get(g_.id)))
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=35 * len(rows) + 40,
                 column_config={"spread": st.column_config.NumberColumn("predicted margin", format="%+.1f",
                                                                        help="Game model: + = this team favoured by"),
                                "win_prob": st.column_config.ProgressColumn("win probability", format="percent",
                                                                            min_value=0, max_value=1),
                                "elo_before": st.column_config.NumberColumn("Elo at kickoff", format="%.0f"),
                                "opp_elo_before": st.column_config.NumberColumn("opponent Elo at kickoff", format="%.0f"),
                                "elo_after": st.column_config.NumberColumn("Elo after", format="%.0f")})
    st.caption("Predicted margin and win probability: the game model - its prediction before kickoff for played "
               "games, inputs as of the last update for the rest (FCS opponents: Elo, no margin). The Elo columns "
               "are the committee model's Elo, at kickoff for games with a result. The opponent's record and rank "
               "are as of now.")

# =============================================================================
# Championship games  (the picks themselves are under Schedule -> Week 14)
# =============================================================================
with t_ccg:
    m = state["matchups"]
    if not state["matchups_final"]:
        left = int((ws.loc[ws.index <= cs.LAST_REGULAR_WEEK, "games"] - ws.loc[ws.index <= cs.LAST_REGULAR_WEEK, "with_result"]).sum())
        st.info(f"The ten championship games are set once every regular-season game has a result ({left} to go). "
                "Below: who would play if the season ended now.")
    else:
        st.caption("Pick the winners under Schedule -> Week 14. Selection Day follows these games.")
        m = m.merge(state["ccg"][["conference", "winner"]], on="conference", how="left")
    cols = ["conference", "seed1", "seed1_record", "seed2", "seed2_record", "site"] + (["winner"] if "winner" in m else [])
    st.dataframe(m[cols], hide_index=True, width="stretch")
    st.subheader("How each team got there")
    for r in m.itertuples(index=False):
        st.markdown(f"**{r.conference}**\n- #1 {r.seed1} ({r.seed1_record}): {r.seed1_why}\n"
                    f"- #2 {r.seed2} ({r.seed2_record}): {r.seed2_why}")

# =============================================================================
# Bracket
# =============================================================================
with t_brk:
    b = state["bracket"]
    if b is None:
        if not state["matchups_final"]:
            st.info("The bracket is built from the Selection Day ranking - finish the regular season and the "
                    "championship games first.")
        else:
            left = int(state["ccg"]["winner"].isna().sum())
            st.info(f"Pick the remaining {left} championship game(s) (Schedule -> Week 14) to produce the Selection "
                    "Day ranking and the bracket.")
    else:
        if state["champion"]:
            st.success(f"National champion: {state['champion']}")
        f = b["field"].merge(b["odds"], on=["seed", "team"])
        f["rank"] = f["rank"].astype(int)
        st.subheader("The field")
        st.dataframe(f[["seed", "team", "rank", "conference", "bid", "p_quarterfinal", "p_semifinal", "p_final", "p_champion"]],
                     hide_index=True, width="stretch", height=35 * 12 + 40,
                     column_config={k: st.column_config.ProgressColumn(v, format="percent", min_value=0, max_value=1)
                                    for k, v in [("p_quarterfinal", "quarterfinal"), ("p_semifinal", "semifinal"),
                                                 ("p_final", "final"), ("p_champion", "champion")]})
        st.caption(("Left out from the top 12: " + ", ".join(b["bumped"]) + ". ") * bool(b["bumped"]) +
                   "This is the field from our single most likely ranking; the Rankings tab (Selection Day) shows each "
                   "team's probability of making it. Seeds 1-4 have byes. The round-by-round odds are exact, from "
                   "current Elo, and update as you pick games.")
        pg = G[G["stage"] == "playoff"].set_index("code")
        for rnd, codes, note in [("First round", [f"R1-{i}" for i in range(1, 5)], "at the higher seed's campus"),
                                 ("Quarterfinals", [f"QF-{i}" for i in range(1, 5)], "neutral site"),
                                 ("Semifinals", ["SF-1", "SF-2"], "neutral site"),
                                 ("Championship", ["FINAL"], "neutral site")]:
            st.subheader(f"{rnd} - {note}")
            any_ = False
            for code in codes:
                if code in pg.index:
                    any_ = True
                    game_row(SimpleNamespace(**pg.loc[code].to_dict()), "po", code)
            if not any_:
                st.caption("Set once the previous round is picked.")

# =============================================================================
# Matchups & power ranking - the game model, read from its precomputed tables (no computation here)
# =============================================================================
with t_mu:
    if base.wp is None:
        st.info("The game model's tables are not available.")
    else:
        st.subheader("Head to head")
        st.caption(f"Any two FBS teams, with every team's inputs as of the last update{WP_ASOF}. The game model "
                   "predicts the margin; the win probability is the one that margin implies. Real results only - "
                   "your picks do not change these numbers.")
        teams_mu = sorted(base.wp.rankings["team"])
        c1, c2, c3 = st.columns([4, 4, 3])
        ta = c1.selectbox("Team A", teams_mu, index=teams_mu.index("Notre Dame") if "Notre Dame" in teams_mu else 0,
                          key="mu_a")
        tb = c2.selectbox("Team B", teams_mu, index=teams_mu.index("Georgia") if "Georgia" in teams_mu else 1,
                          key="mu_b")
        site = c3.radio("Where", ["Neutral site", f"at {ta}", f"at {tb}"], key="mu_site")
        if ta == tb:
            st.warning("Pick two different teams.")
        else:
            if site == "Neutral site":
                p_a, m_a = base.wp.pair(ta, tb, True)
            elif site == f"at {ta}":
                p_a, m_a = base.wp.pair(ta, tb, False)
            else:
                p_b, m_b = base.wp.pair(tb, ta, False)
                p_a, m_a = 1 - p_b, -m_b
            fav, pf, mf = (ta, p_a, m_a) if m_a >= 0 else (tb, 1 - p_a, -m_a)
            k1, k2, k3 = st.columns(3)
            k1.metric(f"{ta} win probability", pct(p_a))
            k2.metric(f"{tb} win probability", pct(1 - p_a))
            k3.metric("Predicted margin", f"{fav} by {mf:.1f}")
            rows_mu = []
            for lab, (p_, m_) in (("Neutral site", base.wp.pair(ta, tb, True)), (f"at {ta}", base.wp.pair(ta, tb, False)),
                                  (f"at {tb}", tuple(1 - v if i == 0 else -v for i, v in enumerate(base.wp.pair(tb, ta, False))))):
                rows_mu.append({"site": lab, f"{ta} win probability": p_, f"{ta} predicted margin": round(m_, 1)})
            st.dataframe(pd.DataFrame(rows_mu), hide_index=True, width="stretch",
                         column_config={f"{ta} win probability": st.column_config.ProgressColumn(
                             f"{ta} win probability", format="percent", min_value=0, max_value=1),
                             f"{ta} predicted margin": st.column_config.NumberColumn(format="%+.1f")})

        st.subheader("Power ranking")
        st.caption("The game model's ranking of every FBS team (the method from our college basketball model): every "
                   "possible neutral-site matchup is predicted, and a team scores by being likely to beat teams that "
                   "are themselves hard to beat (a Markov chain over those win probabilities). Expected win % = its "
                   "average chance of beating every other FBS team on a neutral field. This ranks how GOOD teams are "
                   "right now - not their resume, and not what the committee will do (that is the Rankings tab).")
        R_mu = base.wp.rankings.copy()
        n_mu = st.radio("Show", [25, 50, len(R_mu)], horizontal=True, key="mu_n",
                        format_func=lambda n: "All teams" if n == len(R_mu) else f"Top {n}")
        R_mu["record"] = R_mu["team"].map(REC) if "REC" in globals() else None
        cols_mu = ["MR_Rank", "team", "record", "conference", "MR_Score", "Exp_Wins_pct", "neutral_margin_vs_average",
                   "mov_elo"]
        st.dataframe(R_mu.head(n_mu)[[c for c in cols_mu if c in R_mu]], hide_index=True, width="stretch",
                     height=35 * min(n_mu, 30) + 40,
                     column_config={"MR_Rank": st.column_config.NumberColumn("rank"),
                                    "MR_Score": st.column_config.NumberColumn("power score", format="%.2f",
                                                                              help="Markov score; 1.00 = an average FBS team"),
                                    "Exp_Wins_pct": st.column_config.NumberColumn("expected win % vs all FBS", format="%.1f"),
                                    "neutral_margin_vs_average": st.column_config.NumberColumn(
                                        "avg neutral margin vs FBS", format="%+.1f"),
                                    "mov_elo": st.column_config.NumberColumn("margin Elo", format="%.0f")})

# =============================================================================
# How it works
# =============================================================================
with t_how:
    st.markdown(f"""
**What this is.** A model of the CFP selection committee. You choose who wins; it shows what the committee's
ranking would look like, and the 12-team bracket that follows.

**The data.** Every real result of the {cs.SEASON} season so far (through week {base.real_through}) is loaded and
locked; Elo is carried over from every season since 2008. To pull in newly played games, run
`python run_pipeline.py --data` and restart the app.

**The season in order**
1. **Regular season (weeks 1-13).** Every result or pick updates Elo and every team's resume - for all teams,
   because opponents are judged by where they stand now.
2. **The poll.** Release 1 comes after week 9 (Nov 3); then one per week. Nothing is shown before every game
   through week 9 has a result - the model has never seen an earlier situation. After that the next release is
   shown as *provisional* while you pick its week.
3. **Which prior poll is used.** Release 1: none. Later releases: the real CFP poll from the week before if it
   exists; otherwise our own previous prediction (with the last real poll as an anchor once one exists).
4. **Championship games (week 14).** After week 13 the standings and each conference's tiebreakers set the ten
   title games. Where a rule needs a rating we cannot compute, our predicted ranking stands in.
5. **Selection Day and the bracket.** The four Power-4 champions are in whatever their rank; so is the
   highest-ranked Group of Six team (champion or not); Notre Dame cannot be bumped if it is in the top 12; the
   rest by rank. Seeded in ranking order - automatic-bid teams from outside the top 12 take the last seeds.
6. **Playoff.** Only Elo moves; the ranking is frozen. First round on campus, then neutral sites.

**Four different things - do not mix them up**
- *Simulate buttons*: ONE random season. Each open game is drawn once from the game model's win probability.
- *The forecast (Forecast tab; P(make playoff), P(bye), P(title) everywhere)*: {N_SIMS} seasons played out from
  the last completed week - games, tiebreakers, title games, the committee with its own randomness, the bracket
  rules, the playoff. It re-runs only when a week is completely picked, and reuses the same random draws so the
  odds move because of results, not simulation noise.
- *Range in this poll*: uncertainty about the COMMITTEE in one release, given the results as they stand - 500
  simulated polls. No future games involved.
- *Bracket odds*: exact probabilities of advancing, from the game model, given the field on the Bracket tab.

**The game model (win probabilities and spreads).** A LightGBM model of the point margin (built like our college
basketball model), from both teams' Elo and margin-of-victory Elo, opponent-adjusted efficiency (EPA per play,
success rate, explosiveness), talent, returning production, the poll, Power 4 membership, conference strength and
home field. The win probability is the one its predicted margin implies (a 7-point favourite wins about 68%).
Future games use every team's inputs frozen as of the last weekly update, so a week-12 game is predicted as if it
were played next week. FCS games use Elo.

**The model score is additive**: each input's value times its weight, summed (Team tab). The ODDS are
multiplicative: the chance the committee ranks A ahead of B is e^A / (e^A + e^B), so one extra point multiplies
a team's odds of being picked first by about 2.7.

**What the numbers mean**
- *model score*: the committee score the model assigns. Only comparable within one release.
- *vs #12 line*: score minus the 12th team's score. A gap of 1.0 = about a 73% chance the committee ranks the
  higher team first.
- *strength of record*: how hard it would be for an average top-25 team to have this record against this schedule.
  Tables show the raw value because it reads more easily; the model uses log(strength of record + 0.05).
- *Elo trend*: the weighted average of a team's Elo change over its last three GAMES. An idle week (including a
  championship weekend a team does not play in) leaves it unchanged.

Model: `{base.M['version']}` - trained on {base.M['trained_on'][0]}-{base.M['trained_on'][-1]} committee polls.
""")
