"""
This module has transformations to convert raw json to proper ball by ball dataset. This however does not have any feature sets
"""
import os
import zipfile
import json
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from dagster import asset, MaterializeResult
from dagster_duckdb import DuckDBResource

class CricketDataCurator:
    def __init__(self, input_dir: str, total_balls: int = 120):
        self.input_dir = Path(input_dir)
        self.total_balls = total_balls

    def create_team_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'team' not in df.columns:
            return df
        unique_teams = sorted(df['team'].unique())
        team_mapping = {team: i for i, team in enumerate(unique_teams)}
        df['team_id'] = df['team'].map(team_mapping)
        df['opponent_team_id'] = df['opponent'].map(team_mapping)
        return df

    def _get_match_metadata(self, info: Dict) -> Dict:
        season = str(info.get('season', 'unknown'))
        return {
            'season': season,
            'venue': info.get('venue', 'Unknown Venue'),
            'city': info.get('city', 'Unknown City'),
            'toss_winner': info.get('toss', {}).get('winner', 'Unknown'),
            'toss_decision': info.get('toss', {}).get('decision', 'Unknown'),
            'teams': info.get('teams', []),
            'date': info.get('dates', ['Unknown'])[0]
        }

    def _calculate_target_score(self, innings_data: List) -> int:
        if not innings_data:
            return 0
        first_innings_total = sum(d['runs']['total'] for o in innings_data[0].get('overs', []) for d in o['deliveries'])
        return first_innings_total + 1

    def build_dataset_from_match(self, match_data: Dict) -> List[Dict]:
        dataset = []
        info = match_data.get('info', {})
        meta = self._get_match_metadata(info)
        
        unique_str = f"{meta['teams']}_{meta['date']}_{meta['venue']}"
        match_id = hashlib.sha256(unique_str.encode()).hexdigest()[:15]
        team1 = meta['teams'][0] if len(meta['teams']) > 0 else 'Unknown'
        team2 = meta['teams'][1] if len(meta['teams']) > 1 else 'Unknown'
        
        target_score = self._calculate_target_score(match_data.get('innings', []))

        for idx, innings in enumerate(match_data.get('innings', [])):
            team = innings['team']
            opponent = team2 if team == team1 else team1
            
            final_total = sum(d['runs']['total'] for o in innings.get('overs', []) for d in o['deliveries'])
            is_second_innings = (idx == 1)
            chased_successfully = is_second_innings and (final_total >= target_score)
            
            wickets_lost = 0
            current_runs = 0
            legal_balls = 0
            
            batter_stats = defaultdict(lambda: {'runs': 0, 'balls': 0})
            bowler_stats = defaultdict(lambda: {'runs': 0, 'wickets': 0, 'balls': 0})

            for over in innings.get('overs', []):
                over_num = over['over']
                
                for delivery in over['deliveries']:
                    batter = delivery.get('batter', 'Unknown')
                    bowler = delivery.get('bowler', 'Unknown')
                    
                    runs_batter = delivery['runs']['batter']
                    runs_total = delivery['runs']['total']
                    
                    extras = delivery.get('extras', {})
                    is_wide = 'wides' in extras
                    is_noball = 'noballs' in extras
                    is_legal = not (is_wide or is_noball)
                    
                    wickets = delivery.get('wickets', [])
                    is_wicket = len(wickets) > 0
                    
                    row = {
                        'match_id': match_id,
                        'season': meta['season'],
                        'venue': meta['venue'],
                        'city': meta['city'],
                        'date': meta['date'],
                        'innings': idx + 1,
                        'team': team,
                        'opponent': opponent,
                        'batter': batter,
                        'bowler': bowler,
                        'toss_winner': meta['toss_winner'],
                        'toss_decision': meta['toss_decision'],
                        'toss_win_match_team': (meta['toss_winner'] == team),
                        'over_number': over_num,
                        'ball_in_over': (legal_balls % 6) + 1,
                        'balls_remaining': max(0, self.total_balls - legal_balls),
                        'legal_balls_bowled': int(legal_balls),
                        'wickets_in_hand': int(10 - wickets_lost),
                        'current_score': int(current_runs),
                        'runs_remaining_target': (target_score - current_runs) if is_second_innings else None,
                        'batter_score': batter_stats[batter]['runs'],
                        'batter_balls_faced': batter_stats[batter]['balls'],
                        'bowler_wickets_in_match': bowler_stats[bowler]['wickets'],
                        'runs_off_ball': int(runs_batter),
                        'total_runs_ball': int(runs_total),
                        'is_dot': 1 if runs_total == 0 else 0,
                        'is_six': 1 if runs_batter >= 6 else 0,
                        'is_boundary': 1 if runs_batter >= 4 else 0,
                        'is_wicket': 1 if is_wicket else 0,
                        'phase': 'powerplay' if over_num < 6 else ('middle' if over_num < 16 else 'death'),
                        'censored': chased_successfully,
                        'target_total': target_score if is_second_innings else None,
                        'final_total': final_total
                    }
                    dataset.append(row)

                    current_runs += runs_total
                    if is_wicket: wickets_lost += len(wickets)
                    if is_legal: legal_balls += 1
                    if is_legal and not is_wide: batter_stats[batter]['balls'] += 1
                    batter_stats[batter]['runs'] += runs_batter
                    
                    if is_legal or is_wide:
                        bowler_stats[bowler]['runs'] += runs_total
                        if is_legal: bowler_stats[bowler]['balls'] += 1
                    if is_wicket: bowler_stats[bowler]['wickets'] += len(wickets)

        return dataset

    def process_file(self, file_path: str) -> Optional[List[Dict]]:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'innings' not in data or len(data['innings']) < 2:
                return None
            return self.build_dataset_from_match(data)
        except Exception:
            return None

    def curate(self, n_jobs: int = -1) -> pd.DataFrame:
        files = [str(f) for f in self.input_dir.glob("*.json")]
        if not files:
            return pd.DataFrame()

        results = Parallel(n_jobs=n_jobs)(
            delayed(self.process_file)(fp) for fp in tqdm(files, desc="Curating matches")
        )

        all_data = [row for match in results if match for row in match]
        df = pd.DataFrame(all_data)
        df = self.create_team_mapping(df)
        return df

def run_curation_pipeline(context, duckdb: DuckDBResource, zip_path: str, table_name: str, total_balls: int):
    with tempfile.TemporaryDirectory() as temp_dir:
        context.log.info(f"Extracting zip to {temp_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        curator = CricketDataCurator(temp_dir, total_balls=total_balls)
        df = curator.curate(n_jobs=-1)
        df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
        latest_idx = df['date_dt'].idxmax()
        latest_match = df.loc[latest_idx]
        
        context.log.info(f"Generated {len(df)} ball-by-ball records. Saving to DuckDB...")
        
        with duckdb.get_connection() as conn:
            conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
            
    return MaterializeResult(
        metadata={
            "total_rows": len(df),
            "latest_match": f"{latest_match['team']} vs {latest_match['opponent']}",
            "latest_match_date": str(latest_match['date']),
            "venue": latest_match['venue'],
            "table_name": table_name
        }
    )

@asset(deps=['raw_data_t20'], group_name="silver", compute_kind="python")
def curate_t20_dataset(context, duckdb: DuckDBResource):
    return run_curation_pipeline(
        context, duckdb, "data/raw/t20s_male_json.zip", "t20_ball_by_ball", 120
    )

@asset(deps=['raw_data_odi'], group_name="silver", compute_kind="python")
def curate_odi_dataset(context, duckdb: DuckDBResource):
    return run_curation_pipeline(
        context, duckdb, "data/raw/odis_json.zip", "odi_ball_by_ball", 300
    )

@asset(deps=['raw_data_ipl'], group_name='silver', compute_kind="python")
def curate_ipl_dataset(context, duckdb: DuckDBResource):
    return run_curation_pipeline(
        context, duckdb, "data/raw/ipl_json.zip", "ipl_ball_by_ball", 120
    )