import sys
import json
import time
import requests
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import redis
import math

HEADERS = {"User-Agent": "Mozilla/5.0"}
if getattr(sys, 'frozen', False):
    COEFF_ROOT = sys._MEIPASS
else:
    COEFF_ROOT = os.path.dirname(os.path.abspath(__file__))

class WinPredictor:
    def __init__(self):

        with open(f"{COEFF_ROOT}/run_model_coeffs.json") as f:
            self.run_coeffs = json.load(f)
        with open(f"{COEFF_ROOT}/wicket_model_coeffs.json") as f:
            self.wicket_coeffs = json.load(f)

    def predict_mu(self, rr, rrr, overs_rem, wih, is_second, score):
        z = (self.run_coeffs["const"] +
             self.run_coeffs["rr"] * rr +
             self.run_coeffs["required_run_rate"] * rrr +
             self.run_coeffs["overs_remaining"] * overs_rem +
             self.run_coeffs["wickets_in_hand"] * wih +
             self.run_coeffs["is_second_innings"] * is_second +
             self.run_coeffs["current_score"] * score)
        return np.exp(z)

    def predict_p_wicket(self, rr, rrr, overs_rem, wih, is_second, score):
        z = (self.wicket_coeffs["const"] +
             self.wicket_coeffs["rr"] * rr +
             self.wicket_coeffs["required_run_rate"] * rrr +
             self.wicket_coeffs["overs_remaining"] * overs_rem +
             self.wicket_coeffs["wickets_in_hand"] * wih +
             self.wicket_coeffs["is_second_innings"] * is_second +
             self.wicket_coeffs["current_score"] * score)
        return 1 / (1 + np.exp(-z))

    def sample_runs(self, mu):
        alpha = self.run_coeffs["alpha"]
        r = 1 / alpha
        p = r / (r + mu)
        return np.random.negative_binomial(r, p)

    def featurize(self, scores, wickets, balls, target):
        legal_balls = np.maximum(balls, 1)
        rr = scores / legal_balls * 6
        wih = 10 - wickets
        overs_rem = (120 - balls) / 6
        
        if target is not None:
            runs_rem = np.maximum(target - scores, 0)
            balls_rem = np.maximum(120 - balls, 1)
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
            rr, rrr, overs_rem, wih, is_second = self.featurize(scores, wickets, balls, target)
            mu = self.predict_mu(rr, rrr, overs_rem, wih, is_second, scores)
            p_w = self.predict_p_wicket(rr, rrr, overs_rem, wih, is_second, scores)
            
            runs = self.sample_runs(mu)
            wkts = (np.random.rand(n_sims) < p_w).astype(int)
            
            scores[alive] += runs[alive]
            wickets[alive] += wkts[alive]
            balls[alive] += 6

            dead = (wickets >= 10) | (balls >= 120)
            if target is not None:
                dead |= (scores >= target)
            alive = ~dead

        if target is None: 
            return np.mean(scores)
        return np.mean(scores >= target)

    def get_balls_bowled(self, overs_float):
        over_count = int(overs_float)
        ball_count = round((overs_float % 1) * 10)
        return (over_count * 6) + ball_count

@dataclass
class MatchState:
    match_id: str
    team1: str = ""
    team2: str = ""
    batting_team: str = ""
    bowling_team: str = ""
    status: str = ""
    score: int = 0
    wickets: int = 0
    overs: float = 0.0
    crr: float = 0.0
    target: Optional[int] = None
    last_updated: str = ""

    def update(self):
        try:
            r = requests.get(f"https://www.cricbuzz.com/api/mcenter/comm/{self.match_id}", headers=HEADERS, timeout=5)
            data = r.json()
        except: return
        header = data.get("matchHeader", {})
        mini = data.get("miniscore", {})
        if not mini: return
        t1, t2 = header.get("team1", {}), header.get("team2", {})
        bat_id = mini.get("batTeam", {}).get("teamId")
        self.team1, self.team2 = t1.get("name", ""), t2.get("name", "")
        self.batting_team = t1.get("name") if t1.get("id") == bat_id else t2.get("name")
        self.bowling_team = t2.get("name") if t1.get("id") == bat_id else t1.get("name")
        self.status = header.get("status", "")
        inn = mini.get("matchScoreDetails", {}).get("inningsScoreList", [])
        curr = inn[-1] if inn else {}
        self.score, self.wickets, self.overs = curr.get("score", 0), curr.get("wickets", 0), curr.get("overs", 0.0)
        self.crr, self.target = mini.get("currentRunRate", 0.0), mini.get("target")
        self.last_updated = time.strftime("%H:%M:%S")

@dataclass
class MarketTokenState:
    token_id: str
    outcome: str = ""
    best_bid: float = 0.0
    best_ask: float = 0.0

    def update(self, raw_json: str):
        try: payload = json.loads(raw_json)
        except: return
        
        items = payload if isinstance(payload, list) else [payload]
        for data in items:
            event = data.get("event_type")
            if event == "book":
                bids, asks = data.get("bids", []), data.get("asks", [])
                if bids: self.best_bid = float(max(bids, key=lambda x: float(x["price"]))["price"])
                if asks: self.best_ask = float(min(asks, key=lambda x: float(x["price"]))["price"])
            elif event == "price_change":
                if "best_bid" in data: self.best_bid = float(data["best_bid"])
                if "best_ask" in data: self.best_ask = float(data["best_ask"])

def arbitrage(match_id, redis_ip="localhost", slug=None):
    predictor = WinPredictor()
    match_state = MatchState(match_id)
    
    # Use provided slug or a default placeholder
    if not slug:
        slug = f"cricket-match-{match_id}"
        print(f"⚠️ No slug provided, using default: {slug}")
        
    r = redis.Redis(host=redis_ip, port=6379, decode_responses=True)
    
    try:
        token_ids = r.smembers(f"slug:assets:{slug}")
        if not token_ids:
            print(f"⚠️ No tokens found for slug: {slug}")
    except redis.ConnectionError:
        print(f"❌ Could not connect to Redis at {redis_ip}")
        return

    token_states = {}
    streams_map = {}
    for tid in token_ids:
        meta = r.hgetall(f"token:meta:{tid}")
        token_states[tid] = MarketTokenState(token_id=tid, outcome=meta.get("outcome", ""))
        streams_map[f"orderbook:stream:{tid}"] = "$"

    os.system('clear')
    try:
        while True:
            response = r.xread(streams_map, block=500)
            if response:
                match_state.update()
                state = {
                    "score": match_state.score,
                    "wickets": match_state.wickets,
                    "balls": predictor.get_balls_bowled(match_state.overs),
                    "target": match_state.target
                }
                if match_state.target:
                    sim_result = predictor.simulate(state, n_sims=5000)
                    sim_text = f"Win%: {round(sim_result*100, 1)}"
                else:
                    proj_score = predictor.simulate(state, n_sims=5000)
                    chase_state = {
                        "score": 0,
                        "wickets": 0,
                        "balls": 0,
                        "target": int(proj_score) + 1
                    }
                    chase_win_prob = predictor.simulate(chase_state, n_sims=5000)
                    team_win_prob = 1.0 - chase_win_prob
                    sim_text = f"Proj: {int(proj_score)} | Win%: {round(team_win_prob*100, 1)}"

                for stream_key, messages in response:
                    for msg_id, data in messages:
                        streams_map[stream_key] = msg_id
                        tid = stream_key.split("orderbook:stream:")[-1]
                        token_states[tid].update(data["data"])

                sys.stdout.write("\033[H")
                print("\033[94m" + "="*65 + "\033[0m")
                print(f" 🏏  \033[1;97mLIVE MATCH MONITOR\033[0m | Updated: \033[93m{match_state.last_updated}\033[0m")
                print("\033[94m" + "="*65 + "\033[0m")
                print(f" \033[1;92m{match_state.batting_team:20}\033[0m \033[1;91m{match_state.score}/{match_state.wickets}\033[0m ({match_state.overs} ov)")
                print(f" Opponent: {match_state.bowling_team:18} CRR: {match_state.crr}")
                if match_state.target:
                    print(f" Target:   \033[93m{match_state.target}\033[0m")
                print("\033[94m" + "-"*65 + "\033[0m")
                print(f" \033[1;95m{sim_text}\033[0m")
                print("\033[94m" + "-"*65 + "\033[0m")
                print(" \033[1;97mPOLYMARKET PRICES:\033[0m")
                for tid, ts in token_states.items():
                    outcome_padded = (ts.outcome[:20] + '..') if len(ts.outcome) > 20 else ts.outcome.ljust(22)
                    print(f"  • {outcome_padded} | \033[92mBid: {ts.best_bid:.3f}\033[0m | \033[91mAsk: {ts.best_ask:.3f}\033[0m")
                print("\033[94m" + "="*65 + "\033[0m")
                sys.stdout.write("\033[J")
                sys.stdout.flush()
    except KeyboardInterrupt: pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python minimal_monitor.py <match_id> <redis_ip> [slug]")
        sys.exit(1)
    
    m_id = sys.argv[1]
    r_ip = sys.argv[2] if len(sys.argv) > 2 else "localhost"
    m_slug = sys.argv[3] if len(sys.argv) > 3 else None
    
    arbitrage(m_id, r_ip, m_slug)