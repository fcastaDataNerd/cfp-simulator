"""cfb_model - apply the trained committee model. No training happens here.

    import cfb_model as cm
    M = cm.load_model(MASTER_DIR)                         # written by model_ordinal (cell 10e)
    rel = cm.Release(j=1, is_first=True, is_sd=False, feat=features, pairs=pairs)
    out = cm.predict_season(M, [rel1, rel2, ...], real_polls={})
    sims = cm.simulate_season(M, [rel1, rel2, ...], real_polls={}, rng=...)

One release = the resume table from cfb_engine.features_at(...) at that release's cutoff.
The model gives every team a score; the predicted poll is the teams sorted by score.

WHICH PRIOR-POLL LINE IS USED  (MODEL_SPEC 11.2)
  first   release 1: no prior at all (always predicted cold)
  real    the previous release is a real CFP poll
  A       Era A - no real poll exists yet: prior = our own previous prediction
  B       Era B - a real poll exists but is 2+ releases old: prior = our previous prediction,
          plus the last real poll as an anchor
`predict_season` picks the situation for each release from `real_polls` automatically.

Scores are only comparable WITHIN a release (release 1 has no prior term, so every score
is lower there). Use ranks, gaps within a week, or the probabilities from simulate_season.
"""
import json

import numpy as np
import pandas as pd


# =============================================================================
# 1. The model
# =============================================================================
def load_model(master_dir):
    with open(master_dir / "model_params.json", encoding="utf-8") as fh:
        m = json.load(fh)
    m["beta"] = np.array(m["beta"], float)
    m["boot_betas"] = np.array(m["boot_betas"], float)
    m["idx"] = {f: i for i, f in enumerate(m["feats"])}
    return m


class Release:
    """One release's inputs. feat = cfb_engine.features_at(...) for that cutoff (one row per
    team); pairs = pair_evidence(...) or None (Stage 2 is skipped without it)."""

    def __init__(self, j, feat, is_first, is_sd, pairs=None, label=None):
        self.j, self.is_first, self.is_sd, self.pairs, self.label = j, bool(is_first), bool(is_sd), pairs, label
        self.feat = feat.reset_index(drop=True)
        self.teams = self.feat["team"].to_numpy()
        self._static = None


def _static_matrix(M, rel):
    """The part of the design that does not depend on the prior poll (cached per release)."""
    if rel._static is not None:
        return rel._static
    f = rel.feat
    cols = {f"loss_ge{k}": (f["losses"] >= k).astype(float) for k in range(1, 6)}
    cols["elo"] = f["elo"].astype(float)
    cols["sor_log"] = np.log(f["sor_elo"].astype(float) + M["sor_shift"])
    cols["is_power"] = f["is_power"].astype(float)
    cols["qual_win_top10_elo"] = f["qual_win_top10_elo"].astype(float)
    cols["qual_win_11_25_elo"] = f["qual_win_11_25_elo"].astype(float)
    cols["elo_trend_later"] = 0.0 if rel.is_first else f["elo_trend"].astype(float).fillna(0.0)
    won = (f["ccg_status"] == "won").astype(float) if rel.is_sd else 0.0 * f["elo"]
    lost = (f["ccg_status"] == "lost").astype(float) if rel.is_sd else 0.0 * f["elo"]
    cols["ccg_won"], cols["ccg_lost"] = won, lost
    cols["ccg_won_x_conf"] = won * (f["conf_elo_mean"].astype(float) - M["conf_center"])
    X = np.zeros((len(f), len(M["feats"])))
    for name, i in M["idx"].items():
        if name in cols:
            X[:, i] = np.asarray(cols[name], float)
    rel._static = X
    return X


def _pts(rank_arr):
    """Prior-poll features from an array of ranks (NaN / > 25 = unranked)."""
    r = np.nan_to_num(np.asarray(rank_arr, float), nan=99.0)
    in25 = (r <= 25).astype(float)
    return in25, np.where(in25 > 0, 26.0 - r, 0.0)


def _align(teams, rank_map):
    return np.array([rank_map.get(t, np.nan) for t in teams], float)


def design(M, rel, sit, prior=None, anchor=None):
    """The model's input matrix for a release: one row per team, one column per coefficient
    (M["feats"]), with the prior-poll columns filled in for the situation."""
    X = _static_matrix(M, rel).copy()
    ix = M["idx"]
    if sit != "first":
        in25, pts = _pts(prior)
        if sit == "real":
            X[:, ix["prior_real_in25"]], X[:, ix["prior_real_pts"]] = in25, pts
        elif sit == "A":
            X[:, ix["prior_A_pts"]] = pts
        elif sit == "B":
            X[:, ix["prior_B_pts"]] = pts
            if "anchor_pts" in ix:
                X[:, ix["anchor_pts"]] = _pts(anchor)[1]
        else:
            raise ValueError(sit)
    return X


def stage1_scores(M, rel, sit, prior=None, anchor=None, beta=None):
    """Stage 1 (Plackett-Luce) score for every team in the release."""
    return design(M, rel, sit, prior, anchor) @ (M["beta"] if beta is None else beta)


# =============================================================================
# 2. Stage 2 - head-to-head / common-opponent re-ranking to a fixed point
# =============================================================================
def pair_evidence(tg, season, cutoff, pool):
    """Head-to-head net wins and common-opponent record difference for every pair in `pool`,
    from games on or before `cutoff`. Same definition as build_cfb_resume's pair_week."""
    d = tg[(tg["season"] == season) & (tg["gdate"] <= cutoff) & tg["team"].isin(pool)]
    res = {}
    for t, o, w in zip(d["team"], d["opponent"], d["win"]):
        res.setdefault(t, {}).setdefault(o, []).append(bool(w))
    net = lambda R, o: sum(1 if w else -1 for w in R[o])
    out = []
    pool = sorted(pool)
    for i, A in enumerate(pool):
        ra = res.get(A, {})
        for B in pool[i + 1:]:
            rb = res.get(B, {})
            h2h = net(ra, B) if B in ra else 0
            common = (set(ra) & set(rb)) - {A, B}
            co = sum(net(ra, o) for o in common) - sum(net(rb, o) for o in common)
            if h2h or co:
                out.append((A, B, h2h, co))
    return out


def stage2_rerank(M, teams, s, pairs):
    """Sort by Stage 1, then flip ADJACENT teams whenever the pairwise evidence outweighs the
    score gap; repeat until no swap happens. Only the two teams of a pair ever move. The
    Stage 1 scores are then handed out in the new order."""
    if not M.get("stage2_on") or not pairs:
        return s
    p = M["stage2"]
    pos = {t: i for i, t in enumerate(teams)}
    z = {}
    for A, B, h2h, co in pairs:
        if A in pos and B in pos:
            v = p["g_h"] * h2h + p["g_c"] * co
            if v != 0:
                z[(pos[A], pos[B])], z[(pos[B], pos[A])] = v, -v
    order = list(np.argsort(-s, kind="stable"))
    for _ in range(200):
        swapped = False
        for q in range(len(order) - 1):
            i, j = order[q], order[q + 1]
            v = z.get((i, j))
            if v is None:
                continue
            gap = s[i] - s[j]
            if p["kappa"] * gap + v * np.exp(-abs(gap) / p["tau"]) < 0:
                order[q], order[q + 1] = j, i
                swapped = True
        if not swapped:
            break
    s2 = np.empty_like(s)
    s2[np.array(order)] = np.sort(s)[::-1]
    return s2


def ranks_of(s):
    o = np.argsort(-s, kind="stable")
    r = np.empty(len(s), int)
    r[o] = np.arange(1, len(s) + 1)
    return r


# =============================================================================
# 3. A season of releases
# =============================================================================
def situation(j, real_polls):
    """(sit, last_real) for release j given the real polls that exist (dict release -> ranks)."""
    if j == 1:
        return "first", None
    if (j - 1) in real_polls:
        return "real", j - 1
    earlier = [k for k in real_polls if k < j - 1]
    return ("B", max(earlier)) if earlier else ("A", None)


def predict_season(M, releases, real_polls=None, start=None):
    """Predict each release in order, chaining our own prediction as the prior wherever a
    real poll does not exist. real_polls: {release number: {team: rank}}.
    start: first release to predict (default: all). Returns {j: DataFrame}."""
    real_polls = real_polls or {}
    out, prev = {}, None
    for rel in sorted(releases, key=lambda r: r.j):
        if start is not None and rel.j < start:
            continue
        sit, k = situation(rel.j, real_polls)
        prior = anchor = None
        if sit == "real":
            prior = _align(rel.teams, real_polls[k])
        elif sit in ("A", "B"):
            if prev is None:
                raise ValueError(f"release {rel.j}: no previous prediction to chain from")
            prior = _align(rel.teams, prev)
            if sit == "B":
                anchor = _align(rel.teams, real_polls[k])
        X = design(M, rel, sit, prior, anchor)
        s1 = X @ M["beta"]
        s = stage2_rerank(M, rel.teams, s1, rel.pairs)
        rk, rk1 = ranks_of(s), ranks_of(s1)
        prev = {t: r for t, r in zip(rel.teams, rk) if r <= 25}
        d = rel.feat.copy()
        d["situation"], d["h"] = sit, (None if k is None else rel.j - k)
        d["prior_rank"] = prior if prior is not None else np.nan      # the prior poll actually used
        d["anchor_rank"] = anchor if anchor is not None else np.nan
        for f_, i_ in M["idx"].items():                               # every model input, as used
            d["x_" + f_] = X[:, i_]
        d["score_stage1"], d["score"], d["rank_stage1"], d["rank"] = s1, s, rk1, rk
        top = np.sort(s)[::-1]
        d["gap_to_first"] = top[0] - s
        d["gap_to_12th"] = s - top[min(11, len(top) - 1)]          # > 0 = inside the top 12
        out[rel.j] = d.sort_values("rank").reset_index(drop=True)
    return out


def simulate_season(M, releases, real_polls=None, start=None, m_draw=5, rng=None):
    """Rank distributions: for each of the bootstrap coefficient sets, draw `m_draw` committee
    polls (Plackett-Luce randomness = Gumbel noise on the scores). Where the prior is our own
    prediction, the SIMULATED poll is fed forward, so uncertainty compounds as it would for a
    user. Stage 2 is not applied (it moves < 1 spot on average).
    Returns {j: (teams, ranks array [n_sims, n_teams])}."""
    rng = np.random.default_rng(0) if rng is None else rng
    real_polls = real_polls or {}
    rels = [r for r in sorted(releases, key=lambda r: r.j) if start is None or r.j >= start]
    store = {r.j: [] for r in rels}
    for beta in M["boot_betas"]:
        for _ in range(m_draw):
            prev = None
            for rel in rels:
                sit, k = situation(rel.j, real_polls)
                prior = anchor = None
                if sit == "real":
                    prior = _align(rel.teams, real_polls[k])
                elif sit in ("A", "B"):
                    prior = _align(rel.teams, prev)
                    if sit == "B":
                        anchor = _align(rel.teams, real_polls[k])
                s = stage1_scores(M, rel, sit, prior, anchor, beta=beta) + rng.gumbel(size=len(rel.teams))
                rk = ranks_of(s)
                prev = {t: r for t, r in zip(rel.teams, rk) if r <= 25}
                store[rel.j].append(rk)
    return {r.j: (r.teams, np.vstack(store[r.j])) for r in rels}


def summarize_sims(teams, ranks):
    """Per team: median rank, 80% range, P(top 4), P(top 12), P(top 25)."""
    return pd.DataFrame({"team": teams, "median_rank": np.median(ranks, axis=0),
                         "range_lo": np.percentile(ranks, 10, axis=0), "range_hi": np.percentile(ranks, 90, axis=0),
                         "p_top4": (ranks <= 4).mean(0), "p_top12": (ranks <= 12).mean(0),
                         "p_top25": (ranks <= 25).mean(0)})
