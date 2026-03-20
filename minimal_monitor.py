import sys
import json
import time
import requests
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

HEADERS = {"User-Agent": "Mozilla/5.0"}
if getattr(sys, 'frozen', False):
    COEFF_ROOT = sys._MEIPASS
else:
    COEFF_ROOT = os.path.dirname(os.path.abspath(__file__))

class WinPredictor:
    def __init__(self):
        with open(f"{COEFF_ROOT}/outputs/t20_int_run_model_coeffs.json") as f:
            self.run_coeffs = json.load(f)
        with open(f"{COEFF_ROOT}/outputs/t20_int_wicket_model_coeffs.json") as f:
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
            rr, rrr, overs_rem, wih, is_second = self.featurize(scores[alive], wickets[alive], balls[alive], target)
            mu = self.predict_mu(rr, rrr, overs_rem, wih, is_second, scores[alive])
            p_w = self.predict_p_wicket(rr, rrr, overs_rem, wih, is_second, scores[alive])
            
            runs = self.sample_runs(mu)
            wkts = (np.random.rand(np.sum(alive)) < p_w).astype(int)
            
            scores[alive] += runs
            wickets[alive] += wkts
            balls[alive] += 6

            dead_mask = (wickets[alive] >= 10) | (balls[alive] >= 120)
            if target is not None:
                dead_mask |= (scores[alive] >= target)
            
            # Update alive indices effectively
            alive_indices = np.where(alive)[0]
            alive[alive_indices[dead_mask]] = False

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

def monitor_match(match_id):
    predictor = WinPredictor()
    match_state = MatchState(match_id)
    
    os.system('clear')
    try:
        while True:
            match_state.update()
            if not match_state.batting_team:
                sys.stdout.write("\033[H")
                print(f"Waiting for match {match_id} data... (Last attempt: {time.strftime('%H:%M:%S')})")
                time.sleep(10)
                continue

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
            print("\033[94m" + "="*65 + "\033[0m")
            sys.stdout.write("\033[J")
            sys.stdout.flush()
            
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python minimal_monitor.py <match_id>")
        sys.exit(1)
    
    monitor_match(sys.argv[1])