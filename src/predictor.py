import requests
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional
import json
import lightgbm as lgb

class WinPredictor:
    def __init__(self, run_model_path: str, wicket_model_path: str, max_balls: int = 120):
        self.run_model = lgb.Booster(model_file=run_model_path)
        self.wicket_model = lgb.Booster(model_file=wicket_model_path)
        self.max_balls = max_balls

        meta = run_model_path.replace("_model.txt","_meta.json")
        with open(meta) as f:
            meta = json.load(f)
            self.alpha = meta.get("alpha")

    def predict_mu(self, rr, rrr, overs_rem, wih, is_second, score):
        features = np.column_stack([rr, rrr, wih, is_second, score, overs_rem])
        return self.run_model.predict(features)

    def predict_p_wicket(self, rr, rrr, overs_rem, wih, is_second, score):
        features = np.column_stack([rr, rrr, wih, is_second, score, overs_rem])
        return self.wicket_model.predict(features)

    def sample_runs(self, mu):
        r = 1 / self.alpha
        p = r / (r + mu)
        return np.random.negative_binomial(r, p)

    def featurize(self, scores, wickets, balls, target):
        legal_balls = np.maximum(balls, 1)
        rr = scores / legal_balls * 6
        wih = 10 - wickets
        overs_rem = (self.max_balls - balls) / 6
        
        if target is not None:
            runs_rem = np.maximum(target - scores, 0)
            balls_rem = np.maximum(self.max_balls - balls, 1)
            rrr = runs_rem / balls_rem * 6
            is_second = np.ones_like(scores)
        else:
            rrr = np.zeros_like(scores)
            is_second = np.zeros_like(scores)
        return rr, rrr, overs_rem, wih, is_second

    def simulate(self, initial_state, n_sims=10000):
        scores = np.full(n_sims, initial_state["score"], dtype=float)
        wickets = np.full(n_sims, initial_state["wickets"], dtype=int)
        balls = np.full(n_sims, initial_state["balls"], dtype=int)
        target = initial_state["target"]
        alive = np.ones(n_sims, dtype=bool)

        while np.any(alive):
            
            # This check is done first to ensure we do not predict if there is no wickets or balls left in hand
            dead_mask = (wickets[alive] >= 10) | (balls[alive] >= self.max_balls)
            if target is not None:
                dead_mask |= (scores[alive] >= target)
            
            # Update alive indices effectively
            alive_indices = np.where(alive)[0]
            alive[alive_indices[dead_mask]] = False

            if not np.any(alive):
                break
            
            rr, rrr, overs_rem, wih, is_second = self.featurize(scores[alive], wickets[alive], balls[alive], target)
            mu = self.predict_mu(rr, rrr, overs_rem, wih, is_second, scores[alive])
            p_w = self.predict_p_wicket(rr, rrr, overs_rem, wih, is_second, scores[alive])
            
            runs = self.sample_runs(mu)
            wkts = (np.random.rand(np.sum(alive)) < p_w).astype(int)
            
            scores[alive] += runs
            wickets[alive] += wkts
            balls[alive] += 6

        if target is None: 
            return np.mean(scores)
        return np.mean(scores >= target)

    def get_balls_bowled(self, overs_float):
        over_count = int(overs_float)
        ball_count = round((overs_float % 1) * 10)
        return (over_count * 6) + ball_count