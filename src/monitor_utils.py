import sys
import json
import time
import requests
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

HEADERS = {"User-Agent": "Mozilla/5.0"}
if getattr(sys, 'frozen', False):
    COEFF_ROOT = sys._MEIPASS
else:
    COEFF_ROOT = Path(__file__).resolve().parent.parent

RUN_MODEL_IPL = f"{COEFF_ROOT}/outputs/t20_ipl_run_model_coeffs.json"
WICKET_MODEL_IPL = f"{COEFF_ROOT}/outputs/t20_ipl_wicket_model_coeffs.json"

RUN_MODEL_T20 = f"{COEFF_ROOT}/outputs/t20_int_run_model_coeffs.json"
WICKET_MODEL_T20 = f"{COEFF_ROOT}/outputs/t20_int_wicket_model_coeffs.json"

RUN_MODEL_ODI = f"{COEFF_ROOT}/outputs/odi_int_run_model_coeffs.json"
WICKET_MODEL_ODI = f"{COEFF_ROOT}/outputs/odi_int_wicket_model_coeffs.json"

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