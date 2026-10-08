"""cfb_bracket - turn the final ranking into the 12-team playoff field and bracket.

    import cfb_bracket as br
    champs = br.champions(tg, season)                      # {conference: title-game winner}
    b = br.build_bracket(ranking, champs, rules="2026")    # ranking: team, rank, conference
    b["field"]        seeds 1-12 with how each team got in
    b["bumped"]       teams ranked in the top 12 that are left out
    b["games"]        first round + the bracket slots that follow
    br.advance_odds(b, elo, P)                             # chance of reaching each round

THE 2026 RULES (agreed 2026-10-01, MODEL_SPEC section 14)
  1. Automatic: the ACC, Big 12, Big Ten and SEC championship-game winners - whatever
     their ranking.
  2. Automatic: the highest-ranked Group of Six team (American, Conference USA, MAC,
     Mountain West, Pac-12, Sun Belt). It does NOT have to be a conference champion.
  3. Notre Dame is in if it is ranked in the top 12 - it cannot be bumped.
  4. The remaining places go to the highest-ranked teams left.
  5. Straight seeding: the 12 teams are seeded in ranking order. An automatic-bid team
     ranked outside the top 12 therefore takes one of the LAST seeds (in ranking order),
     and the lowest-ranked unprotected at-large teams drop out.
  6. Seeds 1-4 get byes. First round at the higher seed's campus: 5 v 12, 6 v 11, 7 v 10,
     8 v 9. Quarterfinals on are at neutral sites with no reseeding:
     1 v (8/9), 4 v (5/12), 3 v (6/11), 2 v (7/10); semifinals (1/8/9 side) v (4/5/12 side)
     and (2/7/10 side) v (3/6/11 side).

Earlier seasons are supported only to test the code against the real fields:
  "2024"  five highest-ranked conference champions; the four highest-ranked CHAMPIONS get
          seeds 1-4 (byes); everyone else seeded by rank.
  "2025"  five highest-ranked conference champions; straight seeding.
"""
import numpy as np
import pandas as pd

P4 = ["ACC", "Big 12", "Big Ten", "SEC"]
G6 = ["American Athletic", "Conference USA", "Mid-American", "Mountain West", "Pac-12", "Sun Belt"]
FIRST_ROUND = [(8, 9), (5, 12), (6, 11), (7, 10)]           # (home seed, away seed)
# quarterfinal: bye seed v winner of first-round game index; semifinal pairs quarterfinals
QUARTERS = [(1, 0), (4, 1), (3, 2), (2, 3)]
SEMIS = [(0, 1), (2, 3)]


def champions(tg, season):
    """{conference: winner of its championship game} from the results/picks so far."""
    d = tg[(tg["season"] == season) & tg["is_ccg"] & tg["win"].astype(bool)]
    return dict(zip(d["conference"], d["team"]))


def select_field(order, rank, conf, champs, rules="2026"):
    """The rules themselves. order: teams best-first; rank / conf: dicts; champs: {conference:
    title-game winner}. Returns (seeded teams, automatic bids {team: reason}, protected set,
    the field there would be without the Notre Dame guarantee)."""
    pos = {t: i for i, t in enumerate(order)}
    auto, protected = {}, set()
    if rules == "2026":
        for c in P4:
            if c in champs:
                auto[champs[c]] = f"{c} champion"
        g6 = next((t for t in order if conf.get(t) in G6), None)
        if g6 is not None:
            auto.setdefault(g6, "top Group of Six team")
        if rank.get("Notre Dame", 99) <= 12:
            protected.add("Notre Dame")
    elif rules in ("2024", "2025"):
        ch = sorted(set(champs.values()), key=lambda t: pos[t])[:5]
        inv = {t: c for c, t in champs.items()}
        auto = {t: f"{inv[t]} champion" for t in ch}
    else:
        raise ValueError(rules)

    def fill(start):
        field = set(start)
        for t in order:                                      # at-large: highest ranked left
            if len(field) >= 12:
                break
            field.add(t)
        return field

    field = fill(set(auto) | protected)
    seeded = sorted(field, key=lambda t: pos[t])
    if rules == "2024":                                      # byes to the four best champions
        byes = sorted(auto, key=lambda t: pos[t])[:4]
        seeded = byes + [t for t in seeded if t not in byes]
    return seeded, auto, protected, fill(set(auto))


def build_bracket(ranking, champs, rules="2026"):
    """ranking: DataFrame with team, rank (1 = best) and conference. Teams without a rank
    (e.g. outside a real top 25) are treated as ranked after everyone that has one, in the
    order they appear - pass the model's full ranking to avoid that."""
    r = ranking.copy()
    r["_o"] = np.arange(len(r))
    r["_rk"] = pd.to_numeric(r["rank"], errors="coerce")
    r = r.sort_values(["_rk", "_o"], na_position="last").reset_index(drop=True)
    order = list(r["team"])
    rank = dict(zip(r["team"], r["_rk"]))
    conf = dict(zip(r["team"], r["conference"]))
    for c, t in champs.items():                              # a champion missing from the ranking
        if t not in rank:
            order.append(t)
            rank[t], conf[t] = np.nan, c
    pos = {t: i for i, t in enumerate(order)}                # full ordering, ranked teams first

    seeded, auto, protected, without_protection = select_field(order, rank, conf, champs, rules)
    field = set(seeded)

    def how(t):
        if t in auto:
            return auto[t]
        if t in protected and t not in without_protection:
            return "at-large (Notre Dame top-12 guarantee)"
        return "at-large"

    fld = pd.DataFrame({"seed": range(1, len(seeded) + 1), "team": seeded,
                        "rank": [rank[t] for t in seeded], "conference": [conf.get(t) for t in seeded],
                        "bid": [how(t) for t in seeded],
                        "bye": [s <= 4 for s in range(1, len(seeded) + 1)]})
    bumped = [t for t in order if rank[t] <= 12 and t not in field]
    replaced_by = [t for t in seeded if not (rank[t] <= 12)]
    s2t = dict(zip(fld["seed"], fld["team"]))
    games = [dict(game=f"R1-{i + 1}", round="first round", home=s2t[h], away=s2t[a], home_seed=h, away_seed=a,
                  site=f"at {s2t[h]}", neutral=False) for i, (h, a) in enumerate(FIRST_ROUND)]
    for i, (bye, g_) in enumerate(QUARTERS):
        h, a = FIRST_ROUND[g_]
        games.append(dict(game=f"QF-{i + 1}", round="quarterfinal", home=s2t[bye], away=f"winner R1-{g_ + 1}",
                          home_seed=bye, away_seed=f"{h}/{a}", site="neutral", neutral=True))
    for i, (a, b) in enumerate(SEMIS):
        games.append(dict(game=f"SF-{i + 1}", round="semifinal", home=f"winner QF-{a + 1}", away=f"winner QF-{b + 1}",
                          home_seed=None, away_seed=None, site="neutral", neutral=True))
    games.append(dict(game="FINAL", round="championship", home="winner SF-1", away="winner SF-2",
                      home_seed=None, away_seed=None, site="neutral", neutral=True))
    return dict(rules=rules, field=fld, bumped=bumped, replaced_by=replaced_by, games=pd.DataFrame(games))


def advance_odds(bracket, elo, p, results=None, wp=None):
    """Chance each team wins its first-round game, reaches the semifinal, the final, and wins
    the title (home field in the first round only). Exact. Win probabilities from `wp(a, b,
    neutral)` = P(a beats b), a at home when not neutral (the app passes the game model);
    without it, from Elo.
    results: {game code: winner} for playoff games already decided ("R1-1" .. "FINAL")."""
    results = results or {}
    f = bracket["field"]
    s2t = dict(zip(f["seed"], f["team"]))

    def elo_wp(a, b, neutral):
        adj = 0.0 if neutral else p["HFA"]
        return 1.0 / (1.0 + 10.0 ** ((elo[b] - (elo[a] + adj)) / 400.0))
    wp = wp or elo_wp

    def play(da, db, neutral=True):
        """Distributions {team: prob} for the two sides -> distribution of the winner."""
        out = {}
        for a, pa in da.items():
            for b, pb in db.items():
                w = wp(a, b, neutral)
                out[a] = out.get(a, 0.0) + pa * pb * w
                out[b] = out.get(b, 0.0) + pa * pb * (1 - w)
        return out

    def decided(code, dist):
        return {results[code]: 1.0} if code in results else dist

    r1 = [decided(f"R1-{i + 1}", play({s2t[h]: 1.0}, {s2t[a]: 1.0}, neutral=False))
          for i, (h, a) in enumerate(FIRST_ROUND)]
    qf = [decided(f"QF-{i + 1}", play({s2t[bye]: 1.0}, r1[g_])) for i, (bye, g_) in enumerate(QUARTERS)]
    sf = [decided(f"SF-{i + 1}", play(qf[a], qf[b])) for i, (a, b) in enumerate(SEMIS)]
    fin = decided("FINAL", play(sf[0], sf[1]))
    out = f[["seed", "team"]].copy()
    won_r1 = {t: v for d in r1 for t, v in d.items()}
    out["p_quarterfinal"] = [1.0 if s <= 4 else won_r1.get(t, 0.0) for s, t in zip(out["seed"], out["team"])]
    out["p_semifinal"] = [sum(d.get(t, 0.0) for d in qf) for t in out["team"]]
    out["p_final"] = [sum(d.get(t, 0.0) for d in sf) for t in out["team"]]
    out["p_champion"] = [fin.get(t, 0.0) for t in out["team"]]
    return out
