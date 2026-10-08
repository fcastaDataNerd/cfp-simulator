"""cfb_winprob - the game model's win probabilities, for the app.

The game model (winprob/, a LightGBM margin model) is run weekly by winprob/predict_winprob and
writes three CSVs to winprob/data/app/. This module only READS them:

    import cfb_winprob as cw
    wp = cw.load(BASE_DIR)                 # None if the files are missing -> callers use Elo
    wp.game(game_id)                       # (P(home wins), predicted home margin) or None
    wp.pair(home, away, neutral)           # same, for any two FBS teams (title games, playoff)
    wp.matrices(teams)                     # (P_home, P_neutral) n x n arrays for the forecast

Played games carry the prediction made before kickoff; games not yet played use every team's
inputs frozen as of the last update. FCS games are not in the files: they stay on Elo.
"""
from pathlib import Path

import numpy as np
import pandas as pd

APP_DIR = Path("winprob") / "data" / "app"


class WinProb:
    def __init__(self, games, pairs, rankings):
        self.games_df, self.rankings = games, rankings
        self._game = {int(i): (float(p), float(m)) for i, p, m in zip(games["gameId"], games["p_home"], games["home_margin"])}
        self._home = {(a, b): (float(p), float(m)) for a, b, p, m in
                      zip(pairs["team_a"], pairs["team_b"], pairs["p_a_home"], pairs["a_home_margin"])}
        self._neutral = {(a, b): (float(p), float(m)) for a, b, p, m in
                         zip(pairs["team_a"], pairs["team_b"], pairs["p_neutral"], pairs["neutral_margin"])}
        self.as_of = str(rankings["as_of"].iloc[0]) if "as_of" in rankings and len(rankings) else None

    def game(self, game_id):
        return self._game.get(int(game_id))

    def pair(self, home, away, neutral):
        return (self._neutral if neutral else self._home).get((home, away))

    def matrices(self, teams):
        n = len(teams)
        PH, PN = np.full((n, n), np.nan), np.full((n, n), np.nan)
        for i, a in enumerate(teams):
            for j, b in enumerate(teams):
                if i != j:
                    h, nt = self._home.get((a, b)), self._neutral.get((a, b))
                    if h:
                        PH[i, j] = h[0]
                    if nt:
                        PN[i, j] = nt[0]
        return PH, PN


def load(base_dir):
    d = Path(base_dir) / APP_DIR
    try:
        return WinProb(pd.read_csv(d / "game_probs_2026.csv"), pd.read_csv(d / "matchup_probs_2026.csv"),
                       pd.read_csv(d / "model_rankings_2026.csv"))
    except FileNotFoundError:
        return None
