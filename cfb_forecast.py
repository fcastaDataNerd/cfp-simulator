"""cfb_forecast - the season FORECAST: play the rest of the season many times.

    import cfb_forecast as fc
    plan = fc.build_plan(base)                                   # once (schedule -> arrays)
    out  = fc.run(base, plan, picks, ccg_picks, po_picks, n_sims=500, seed=0)
    tab  = fc.summarize(plan, out)                               # one row per team

Each simulated season, in order:
  1. every open regular-season game is drawn from its Elo win probability, in date order,
     with Elo updating after each game (real results and your picks are fixed);
  2. the committee's poll is produced at every release - one of the 100 bootstrap versions of
     the model per season, plus the committee's own randomness (Plackett-Luce noise), each
     poll feeding the next as the prior exactly as in cfb_model;
  3. standings + each conference's tiebreakers set the ten championship games (cfb_standings);
  4. those games are drawn, then Selection Day's poll, then the bracket rules (cfb_bracket);
  5. the playoff is drawn from Elo, round by round.

It is the same logic as cfb_season, written on arrays so that ALL the seasons are played at
once: `cfb_season.build_state` costs 1-2 seconds per season, this costs a few milliseconds.
check_cfb_forecast proves the two give identical results for identical game outcomes.

COMMON RANDOM NUMBERS. Every random number is drawn up front from `seed`, in arrays whose
shape does not depend on the picks. So the same picks always give the same forecast, and
after you change one pick every other game in every simulated season is decided by the same
draw as before - the odds move because of the pick, not because of simulation noise.

Stage 2 (head-to-head re-ranking) is not applied inside the forecast: it moves a team less
than one spot on average and the committee noise is far larger (as in cfb_model.simulate_season).
"""
import numpy as np
import pandas as pd

import cfb_common as cc
import cfb_engine as eng
import cfb_standings as st
import cfb_model as cm
import cfb_bracket as br
import cfb_season as cs

TREND_W = np.array(eng.ELO_TREND_WEIGHTS)
R1 = br.FIRST_ROUND                       # (home seed, away seed)


class Plan:
    pass


def build_plan(base):
    """Everything about the season that does not depend on results: the schedule as arrays."""
    pl = Plan()
    P, season = base.P, cs.SEASON
    pl.teams = list(base.fbs_teams)
    pl.n = n = len(pl.teams)
    pl.idx = {t: i for i, t in enumerate(pl.teams)}
    FCS = n                                                     # one extra column: the pooled FCS team
    conf = [base.conf_of.get(t) for t in pl.teams]
    pl.conf = conf
    pl.conf_of = dict(zip(pl.teams, conf))
    pl.is_power = np.array([cc.is_power_conf(c, t, season) for t, c in zip(pl.teams, conf)], float)

    # ---- ratings entering the season: the engine's offseason regression toward the conference mean
    ratings = dict(base.ratings_start)

    def group(t):
        if t == "Notre Dame":
            return "__POWER_INDEP__"
        c = base.conf_hist.get((base.start_season, t))
        return "__OTHER_INDEP__" if (c is None or c == "FBS Independents") else c
    groups = {}
    for t, v in ratings.items():
        groups.setdefault(group(t), []).append(v)
    means = {k: sum(v) / len(v) for k, v in groups.items()}
    a = P["ALPHA"]
    ratings = {t: a * v + (1 - a) * means[group(t)] for t, v in ratings.items()}
    hi, lo = eng.INIT_ELO + P["INIT_GAP"] / 2.0, eng.INIT_ELO - P["INIT_GAP"] / 2.0
    pl.r0 = np.array([ratings.get(t, hi if pl.is_power[i] else lo) for i, t in enumerate(pl.teams)] + [P["FCS_ELO"]])

    # ---- the schedule, in the engine's order ---------------------------------------------------------
    g = eng.prepare_games(base.sched)
    g = g[~((g["homeKey"] == eng.FCS_KEY) & (g["awayKey"] == eng.FCS_KEY))].sort_values(["startDate", "id"])
    played = g["completed"].fillna(False).astype(bool) & (g["homePoints"] != g["awayPoints"])
    g = g.assign(real=np.where(played, (g["homePoints"] > g["awayPoints"]).astype(int), -1))
    mult = [P["CROSS_CONF"] if (not cg and hc and ac and hc != ac) else 1.0
            for cg, hc, ac in zip(g["confGame"], g["homeConference"], g["awayConference"])]
    g = g.assign(mult=mult, h=[pl.idx.get(k, FCS) for k in g["homeKey"]], a=[pl.idx.get(k, FCS) for k in g["awayKey"]])
    pre = g[g["week"] <= cs.LAST_REGULAR_WEEK].reset_index(drop=True)      # weeks 1-13
    post = g[g["week"] > cs.CCG_WEEK].reset_index(drop=True)               # after Selection Day (Army-Navy)
    for name, d in (("pre", pre), ("post", post)):
        setattr(pl, name, dict(id=d["id"].to_numpy(), h=d["h"].to_numpy(), a=d["a"].to_numpy(),
                               adj=np.where(d["neutral"], 0.0, P["HFA"]), mult=d["mult"].to_numpy(float),
                               real=d["real"].to_numpy(), n=len(d), home=d["homeTeam"].to_numpy(),
                               away=d["awayTeam"].to_numpy()))
    pl.bnd = {j: int((pre["gdate"] <= base.cutoff[w]).sum()) for j, w in cs.RELEASE_WEEK.items() if j <= 5}

    # ---- each team's games, as slots -------------------------------------------------------------------
    slots = [[] for _ in range(n)]
    for q, (h, a_, neu) in enumerate(zip(pre["h"], pre["a"], pre["neutral"])):
        if h != FCS:
            slots[h].append((q, True, a_, 0.0 if neu else P["HFA"]))
        if a_ != FCS:
            slots[a_].append((q, False, h, 0.0 if neu else -P["HFA"]))
    Gm = max(len(s) for s in slots)
    pl.G = Gm
    pl.TG = np.zeros((n, Gm), int)
    pl.ISHOME = np.zeros((n, Gm), bool)
    pl.OPP = np.full((n, Gm), FCS)
    pl.ADJ = np.zeros((n, Gm))
    pl.VALID = np.zeros((n, Gm), bool)
    for i, s in enumerate(slots):
        for k, (q, ih, o, adj) in enumerate(s):
            pl.TG[i, k], pl.ISHOME[i, k], pl.OPP[i, k], pl.ADJ[i, k], pl.VALID[i, k] = q, ih, o, adj, True
    pl.OPPFCS = pl.OPP == FCS
    pl.V = {j: pl.VALID & (np.arange(pre.shape[0])[pl.TG] < pl.bnd[j]) for j in pl.bnd}
    pl.GP = pl.VALID.sum(1)
    pl.LAST = np.maximum(pl.GP - 1, 0)

    # ---- conferences ------------------------------------------------------------------------------------
    all_confs = sorted({c for c in conf if isinstance(c, str) and c != "FBS Independents"})
    pl.MEMB = np.array([[1.0 if conf[i] == c else 0.0 for i in range(n)] for c in all_confs])
    pl.conf_id = np.array([all_confs.index(c) if c in all_confs else -1 for c in conf])
    pl.ccg = []
    for ci, cname in enumerate(st.RULES):
        r = st.rules_for(cname, season)
        if r is None or cname not in cc.ccg_conferences(season):
            continue
        members = [t for t in pl.teams if pl.conf_of[t] == cname]
        games = []
        for q, (ht, at, hc, ac, cg, wk) in enumerate(zip(pre["homeTeam"], pre["awayTeam"], pre["homeConference"],
                                                         pre["awayConference"], pre["confGame"], pre["week"])):
            if hc != cname or ac != cname or not cg or ht not in pl.idx or at not in pl.idx:
                continue
            if "last_week" in r and wk > r["last_week"]:
                continue
            if any((len(pr) == 2 or pr[0] == season) and {ht, at} == set(pr[-2:]) for pr in r.get("nonconf", [])):
                continue
            games.append((q, ht, at))
        pl.ccg.append(dict(name=cname, order=ci, members=members, games=games,
                           neutral=(r["site"] != "seed1_home"), midx=np.array([pl.idx[t] for t in members])))
    pl.n_conf = len(pl.ccg)
    return pl


def _known(pl, part, picks):
    """Result of each game: 1 home won, 0 away won, -1 open (real results beat picks)."""
    k = part["real"].copy()
    for q, gid in enumerate(part["id"]):
        if k[q] < 0 and gid in picks:
            k[q] = int(bool(picks[gid]))
    return k


def _ranks(score):
    order = np.argsort(-score, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(1, score.shape[1] + 1)[None, :], axis=1)
    return rank, order


def _features(pl, P, E, WIN, CH, OPP, ADJ, valid):
    """Resume features for every team in every simulated season at one cutoff.
    E (S, n) Elo at the cutoff; the rest are per game slot, broadcastable to (S, n, slots)."""
    S, n = E.shape
    rank, order = _ranks(E)
    ref = np.take_along_axis(E, order[:, :25], axis=1).mean(1)             # the average top-25 team
    E_ext = np.concatenate([E, np.full((S, 1), P["FCS_ELO"])], axis=1)
    rk_ext = np.concatenate([rank, np.full((S, 1), 9999)], axis=1)
    if OPP.ndim == 2:
        OE, ORK = E_ext[:, OPP], rk_ext[:, OPP]
    else:
        ar = np.arange(S)[:, None, None]
        OE, ORK = E_ext[ar, OPP], rk_ext[ar, OPP]
    valid = np.broadcast_to(valid, WIN.shape)
    wv = WIN & valid
    wins = wv.sum(2)
    losses = valid.sum(2) - wins
    # strength of record: P(an average top-25 team wins at least this many of these games)
    ps = np.where(valid, 1.0 / (1.0 + 10.0 ** ((OE - (ref[:, None, None] + ADJ)) / 400.0)), 0.0)
    Gs = WIN.shape[2]
    dp = np.zeros((S, n, Gs + 1))
    dp[..., 0] = 1.0
    for k in range(Gs):
        p = ps[..., k][..., None]
        new = dp * (1 - p)
        new[..., 1:] += dp[..., :-1] * p
        dp = new
    tail = np.cumsum(dp[..., ::-1], axis=2)[..., ::-1]                      # P(at least k wins)
    t = np.take_along_axis(tail, wins[..., None], axis=2)[..., 0]
    sor = np.maximum(0.0, -np.log(np.minimum(1.0, np.maximum(t, 1e-12))))
    q10 = (wv & (ORK <= 10)).sum(2)
    q25 = (wv & (ORK >= 11) & (ORK <= 25)).sum(2)
    # Elo trend: weighted mean of the last three Elo changes, most recent first
    back = np.cumsum(valid[..., ::-1], axis=2)[..., ::-1]                   # 1 = most recent valid game
    wt = np.where(valid & (back <= len(TREND_W)), TREND_W[np.clip(back, 1, len(TREND_W)) - 1], 0.0)
    den = wt.sum(2)
    trend = np.where(den > 0, (wt * CH).sum(2) / np.where(den > 0, den, 1.0), 0.0)
    return dict(elo=E, elo_rank=rank, wins=wins, losses=losses, sor_elo=sor, qual_win_top10_elo=q10,
                qual_win_11_25_elo=q25, elo_trend=trend)


def run(base, pl, picks=None, ccg_picks=None, po_picks=None, n_sims=500, seed=0, deterministic=False, debug=False,
        fixed_matchups=None):
    """Play the rest of the season n_sims times. deterministic=True: no committee noise and the
    main coefficients (used by the parity check). fixed_matchups {conference: (seed 1, seed 2)}:
    once the regular season is complete the title games are known - use them in every season
    instead of re-deriving them (the tiebreak stand-in would otherwise vary with the noise)."""
    picks, ccg_picks, po_picks = picks or {}, ccg_picks or {}, po_picks or {}
    P, M, n, S = base.P, base.M, pl.n, n_sims
    FCS, K = n, P["K"]
    ix = M["idx"]
    rng = np.random.default_rng(seed)
    U = rng.random((S, pl.pre["n"]))
    Upost = rng.random((S, max(pl.post["n"], 1)))
    Ucc = rng.random((S, pl.n_conf))
    Upo = rng.random((S, 11))
    GUM = rng.gumbel(size=(S, 6, n))
    BI = rng.integers(0, len(M["boot_betas"]), S)
    if deterministic:
        GUM = np.zeros_like(GUM)
        BETA = np.tile(M["beta"], (S, 1))
    else:
        BETA = M["boot_betas"][BI]
    ar = np.arange(S)

    def play_fixed(part, Ux, R, snaps=None):
        """A list of scheduled games, in order. Returns home-won and Elo-change arrays."""
        kn = _known(pl, part, picks)
        HW = np.empty((S, part["n"]), bool)
        D = np.empty((S, part["n"]))
        for q in range(part["n"]):
            if snaps is not None:
                for j, b in pl.bnd.items():
                    if b == q:
                        snaps[j] = R[:, :n].copy()
            h, a = part["h"][q], part["a"][q]
            e = 1.0 / (1.0 + 10.0 ** ((R[:, a] - (R[:, h] + part["adj"][q])) / 400.0))
            hw = (Ux[:, q] < e) if kn[q] < 0 else np.full(S, kn[q] == 1)
            d = K * part["mult"][q] * (hw - e)
            if h != FCS:
                R[:, h] += d
            if a != FCS:
                R[:, a] -= d
            HW[:, q], D[:, q] = hw, d
        if snaps is not None:
            for j, b in pl.bnd.items():
                if b == part["n"]:
                    snaps[j] = R[:, :n].copy()
        return HW, D

    # ---- 1. regular season -----------------------------------------------------------------------------
    R = np.tile(pl.r0, (S, 1))
    snaps = {}
    HW, D = play_fixed(pl.pre, U, R, snaps)
    HWt = HW[:, pl.TG]
    WIN = np.where(pl.ISHOME[None], HWt, ~HWt) & pl.VALID[None]
    CH = np.where(pl.ISHOME[None], D[:, pl.TG], -D[:, pl.TG])

    # ---- 2. the polls, release by release -----------------------------------------------------------------
    real = base.real_polls
    real_arr = {j: np.array([p_.get(t, np.nan) for t in pl.teams], float) for j, p_ in real.items()}

    def pts(rank):
        r = np.nan_to_num(np.asarray(rank, float), nan=99.0)
        in25 = (r <= 25).astype(float)
        return in25, np.where(in25 > 0, 26.0 - r, 0.0)

    RANK, SCORE, FEAT = {}, {}, {}

    def poll(j, f, extra=None):
        X = np.zeros((S, n, len(M["feats"])))
        for k in range(1, 6):
            X[..., ix[f"loss_ge{k}"]] = f["losses"] >= k
        X[..., ix["elo"]] = f["elo"]
        X[..., ix["sor_log"]] = np.log(f["sor_elo"] + M["sor_shift"])
        X[..., ix["is_power"]] = pl.is_power[None]
        X[..., ix["qual_win_top10_elo"]] = f["qual_win_top10_elo"]
        X[..., ix["qual_win_11_25_elo"]] = f["qual_win_11_25_elo"]
        if j > 1:
            X[..., ix["elo_trend_later"]] = f["elo_trend"]
        if extra is not None:
            X[..., ix["ccg_won"]], X[..., ix["ccg_lost"]] = extra["won"], extra["lost"]
            X[..., ix["ccg_won_x_conf"]] = extra["won"] * (extra["conf_mean"] - M["conf_center"])
        sit, k = cm.situation(j, real)
        if sit == "real":
            in25, p_ = pts(real_arr[k])
            X[..., ix["prior_real_in25"]], X[..., ix["prior_real_pts"]] = in25[None], p_[None]
        elif sit == "A":
            X[..., ix["prior_A_pts"]] = pts(RANK[j - 1])[1]
        elif sit == "B":
            X[..., ix["prior_B_pts"]] = pts(RANK[j - 1])[1]
            if "anchor_pts" in ix:
                X[..., ix["anchor_pts"]] = pts(real_arr[k])[1][None]
        s = np.einsum("snf,sf->sn", X, BETA) + GUM[:, j - 1, :]
        RANK[j], SCORE[j] = _ranks(s)[0], s
        if debug:
            FEAT[j] = f

    for j in range(1, 6):
        f = _features(pl, P, snaps[j], WIN, CH, pl.OPP, pl.ADJ[None], pl.V[j][None])
        poll(j, f)

    # ---- 3. standings + tiebreakers -> the championship games ----------------------------------------------
    W13 = WIN.sum(2)
    FW = (WIN & pl.OPPFCS[None]).sum(2)
    extra_fcs = np.maximum(FW - 1, 0)                                       # at most one FCS win counts
    TW = W13 - extra_fcs
    OV = TW / np.maximum(pl.GP[None] - extra_fcs, 1)
    FWIN = WIN[:, np.arange(n), pl.LAST]
    cfp4 = real_arr.get(4)
    C1 = np.zeros((S, pl.n_conf), int)
    C2 = np.zeros((S, pl.n_conf), int)
    for ci, c in enumerate(pl.ccg):
        if fixed_matchups and c["name"] in fixed_matchups:
            C1[:, ci], C2[:, ci] = (pl.idx[t] for t in fixed_matchups[c["name"]])
            continue
        mi, members = c["midx"], c["members"]
        gq = [q for q, _, _ in c["games"]]
        hw_c = HW[:, gq]
        for s in range(S):
            games = [(ht, at, hw_c[s, k]) for k, (_, ht, at) in enumerate(c["games"])]
            rating = dict(zip(members, RANK[5][s, mi]))
            src = cfp4 if cfp4 is not None else RANK[4][s]
            cfp = {t: int(src[i]) for t, i in zip(members, mi) if src[i] <= 25}
            ctx = st._Ctx.from_records(c["name"], cs.SEASON, members, games,
                                       dict(zip(members, TW[s, mi])), dict(zip(members, OV[s, mi])),
                                       dict(zip(members, FWIN[s, mi])), rating, cfp, base.DIV)
            (a_, _), (b_, _) = st.seeds_for(ctx)
            C1[s, ci], C2[s, ci] = pl.idx[a_], pl.idx[b_]

    # ---- 4. championship games, Selection Day, bracket ------------------------------------------------------
    c_opp = np.full((S, n), FCS)
    c_win = np.zeros((S, n), bool)
    c_ch = np.zeros((S, n))
    c_adj = np.zeros((S, n))
    c_val = np.zeros((S, n), bool)
    CCGW = np.zeros((S, pl.n_conf), int)
    for ci, c in enumerate(pl.ccg):
        h, a = C1[:, ci], C2[:, ci]
        adj = 0.0 if c["neutral"] else P["HFA"]
        e = 1.0 / (1.0 + 10.0 ** ((R[ar, a] - (R[ar, h] + adj)) / 400.0))
        hw = Ucc[:, ci] < e
        w = pl.idx.get(ccg_picks.get(c["name"]), -1)
        hw = np.where(h == w, True, np.where(a == w, False, hw))
        d = K * (hw - e)                                                    # a conference game: no cross-conference boost
        R[ar, h] += d
        R[ar, a] -= d
        c_opp[ar, h], c_opp[ar, a] = a, h
        c_win[ar, h], c_win[ar, a] = hw, ~hw
        c_ch[ar, h], c_ch[ar, a] = d, -d
        c_adj[ar, h], c_adj[ar, a] = adj, -adj
        c_val[ar, h] = c_val[ar, a] = True
        CCGW[:, ci] = np.where(hw, h, a)
    E6 = R[:, :n].copy()
    f6 = _features(pl, P, E6,
                   np.concatenate([WIN, c_win[..., None]], axis=2), np.concatenate([CH, c_ch[..., None]], axis=2),
                   np.concatenate([np.broadcast_to(pl.OPP[None], (S, n, pl.G)), c_opp[..., None]], axis=2),
                   np.concatenate([np.broadcast_to(pl.ADJ[None], (S, n, pl.G)), c_adj[..., None]], axis=2),
                   np.concatenate([np.broadcast_to(pl.VALID[None], (S, n, pl.G)), c_val[..., None]], axis=2))
    cmean = (E6 @ pl.MEMB.T) / pl.MEMB.sum(1)[None]                        # conference Elo means
    conf_mean = np.where(pl.conf_id[None] >= 0, cmean[:, np.maximum(pl.conf_id, 0)], E6.mean(1)[:, None])
    f6["conf_elo_mean"] = conf_mean
    poll(6, f6, extra=dict(won=c_val & c_win, lost=c_val & ~c_win, conf_mean=conf_mean))

    SEEDS = np.zeros((S, 12), int)
    order6 = np.argsort(RANK[6], axis=1, kind="stable")
    for s in range(S):
        order = [pl.teams[i] for i in order6[s]]
        champs = {c["name"]: pl.teams[CCGW[s, ci]] for ci, c in enumerate(pl.ccg)}
        seeded = br.select_field(order, dict(zip(pl.teams, RANK[6][s])), pl.conf_of, champs)[0]
        SEEDS[s] = [pl.idx[t] for t in seeded]

    # ---- 5. after Selection Day: remaining regular games (Army-Navy), then the playoff -----------------------
    HWpost, _ = play_fixed(pl.post, Upost, R) if pl.post["n"] else (np.zeros((S, 0), bool), None)
    cid = np.concatenate([pl.conf_id, [-9]])
    conf_names = np.array(pl.conf + [None], dtype=object)

    def game(h, a, neutral, u, code):
        e = 1.0 / (1.0 + 10.0 ** ((R[ar, a] - (R[ar, h] + (0.0 if neutral else P["HFA"]))) / 400.0))
        w = pl.idx.get(po_picks.get(code), -1)
        hw = np.where(h == w, True, np.where(a == w, False, u < e))
        diff = np.array([bool(x) and bool(y) and x != y for x, y in zip(conf_names[h], conf_names[a])])
        d = K * np.where(diff, P["CROSS_CONF"], 1.0) * (hw - e)
        R[ar, h] += d
        R[ar, a] -= d
        return np.where(hw, h, a)

    PO = {}
    for i, (hs, as_) in enumerate(R1):
        PO[f"R1-{i + 1}"] = game(SEEDS[:, hs - 1], SEEDS[:, as_ - 1], False, Upo[:, i], f"R1-{i + 1}")
    for i, (bye, g_) in enumerate(br.QUARTERS):
        PO[f"QF-{i + 1}"] = game(SEEDS[:, bye - 1], PO[f"R1-{g_ + 1}"], True, Upo[:, 4 + i], f"QF-{i + 1}")
    for i, (a_, b_) in enumerate(br.SEMIS):
        PO[f"SF-{i + 1}"] = game(PO[f"QF-{a_ + 1}"], PO[f"QF-{b_ + 1}"], True, Upo[:, 8 + i], f"SF-{i + 1}")
    PO["FINAL"] = game(PO["SF-1"], PO["SF-2"], True, Upo[:, 10], "FINAL")

    out = dict(n_sims=S, RANK=RANK, SEEDS=SEEDS, C1=C1, C2=C2, CCGW=CCGW, PO=PO, CHAMP=PO["FINAL"], HW=HW,
               HWpost=HWpost, R=R[:, :n])
    if debug:
        out.update(FEAT=FEAT, SCORE=SCORE, BETA=BETA, GUM=GUM, snaps=snaps)
    return out


def summarize(pl, out):
    """One row per team: the forecast."""
    S, n = out["n_sims"], pl.n
    seeds = np.zeros((S, n), int)
    np.put_along_axis(seeds, out["SEEDS"], np.arange(1, 13)[None, :].repeat(S, 0), axis=1)
    made = seeds > 0
    conf_title = np.zeros((S, n), bool)
    np.put_along_axis(conf_title, out["CCGW"], True, axis=1)
    in_ccg = np.zeros((S, n), bool)
    np.put_along_axis(in_ccg, out["C1"], True, axis=1)
    np.put_along_axis(in_ccg, out["C2"], True, axis=1)
    fr = out["RANK"][6]

    def reached(codes):                                       # played in a game of this round
        m = np.zeros((S, n), bool)
        for c in codes:
            np.put_along_axis(m, out["PO"][c][:, None], True, axis=1)
        return m
    won_r1_or_bye = reached([f"R1-{i}" for i in range(1, 5)]) | ((seeds >= 1) & (seeds <= 4))
    t = pd.DataFrame({
        "team": pl.teams, "conference": pl.conf,
        "p_playoff": made.mean(0), "p_bye": ((seeds >= 1) & (seeds <= 4)).mean(0),
        "p_title_game": in_ccg.mean(0), "p_conf_champ": conf_title.mean(0),
        "p_quarterfinal": won_r1_or_bye.mean(0), "p_semifinal": reached([f"QF-{i}" for i in range(1, 5)]).mean(0),
        "p_final": reached(["SF-1", "SF-2"]).mean(0),
        "p_champion": np.bincount(out["CHAMP"], minlength=n) / S,
        "avg_seed": np.where(made.sum(0) > 0, seeds.sum(0) / np.maximum(made.sum(0), 1), np.nan),
        "final_rank_median": np.median(fr, axis=0), "final_rank_lo": np.percentile(fr, 10, axis=0),
        "final_rank_hi": np.percentile(fr, 90, axis=0), "p_final_top12": (fr <= 12).mean(0),
        "p_final_top25": (fr <= 25).mean(0)})
    return t.sort_values(["p_playoff", "p_champion", "final_rank_median"], ascending=[False, False, True]).reset_index(drop=True)
