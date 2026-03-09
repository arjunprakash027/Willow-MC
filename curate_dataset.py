import argparse
import json
import os
import uuid
import warnings
import zipfile
import hashlib
import requests
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class CricketDataCurator:
    """
    Handles the curation of cricket match JSON data into structured formats and 
    provides visualization tools for deep analytics.
    """

    def __init__(self, input_dir: str, output_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        
        # Set visualization styles
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except:
            plt.style.use('ggplot')
        sns.set_context("notebook", font_scale=1.1)
        sns.set_palette("tab10")

    def download_and_extract(self, url: str = "https://cricsheet.org/downloads/t20s_male_json.zip"):
        """
        Downloads and extracts the latest data from Cricsheet.
        """
        print(f"\n🌐 FETCHING LATEST DATA FROM CRICSHEET")
        print(f"=======================================")
        
        zip_path = self.input_dir / "temp_data.zip"
        
        print(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, "wb") as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)

        print(f"Extracting to: {self.input_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.input_dir)
        
        os.remove(zip_path)
        print(f"✅ Download and extraction complete.")

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
        
        # Deterministic match_id based on match metadata
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
                    
                    # Store state BEFORE the delivery
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
                        'balls_remaining': max(0, 120 - legal_balls),
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
                        'is_six': 1 if runs_batter == 6 else 0,
                        'is_boundary': 1 if runs_batter >= 4 else 0,
                        'is_wicket': 1 if is_wicket else 0,
                        'phase': 'powerplay' if over_num < 6 else ('middle' if over_num < 16 else 'death'),
                        'censored': chased_successfully,
                        'target_total': target_score if is_second_innings else None,
                        'final_total': final_total
                    }
                    dataset.append(row)

                    # Update state AFTER delivery
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
            print(f"❌ No JSON files found in {self.input_dir}. Use --download if needed.")
            return pd.DataFrame()

        results = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
            delayed(self.process_file)(fp) for fp in tqdm(files, desc="Curating matches")
        )

        all_data = [row for match in results if match for row in match]
        df = pd.DataFrame(all_data)
        df = self.create_team_mapping(df)
        
        output_path = self.output_dir / "dls_dataset.parquet"
        df.to_parquet(output_path, compression='zstd', index=False)
        
        print(f"\n✅ Curation Complete:")
        print(f"• Processed {len([r for r in results if r])} matches")
        print(f"• Generated {len(df):,} data points")
        
        # Recent Match Stats
        if not df.empty:
            # Drop duplicates to get unique matches with their metadata
            metadata_cols = ['match_id', 'date', 'team', 'opponent', 'venue', 'city']
            match_meta = df[metadata_cols].drop_duplicates('match_id')
            match_meta['date_dt'] = pd.to_datetime(match_meta['date'], errors='coerce')
            latest_match = match_meta.sort_values('date_dt', ascending=False).iloc[0]
            
            location = f" at {latest_match['venue']}"
            if latest_match['city'] and latest_match['city'] != 'Unknown City':
                location += f", {latest_match['city']}"
                
            print(f"• Most Recent Match: {latest_match['team']} vs {latest_match['opponent']} ({latest_match['date']}){location}")

        print(f"• Saved to: {output_path}")
        
        return df

def main():
    parser = argparse.ArgumentParser(description="Clean and Visualize Cricket Match Data")
    parser.add_argument("--download", action="store_true", help="Download latest T20 data from Cricsheet")
    parser.add_argument("--input", type=str, 
                        default="data/raw",
                        help="Input JSON directory")
    parser.add_argument("--output", type=str, 
                        default="data/processed",
                        help="Output directory")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel jobs")
    
    args = parser.parse_args()
    
    curator = CricketDataCurator(args.input, args.output)
    
    if args.download:
        curator.download_and_extract()
        
    curator.curate(n_jobs=args.jobs)

if __name__ == "__main__":
    main()