import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import duckdb
from typing import Dict, List
from joblib import Parallel, delayed
from tqdm import tqdm

# Ensure project root is in path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.predictor import WinPredictor  # noqa: E402

def select_matches_for_backtest(db_path: str, dataset: str, n_matches: int = 100, random_state: int = 42) -> List[str]:
    """
    Select matches from DuckDB database that have exactly 2 innings.
    """
    conn = duckdb.connect(db_path, read_only=True)
    try:
        query = f"""
            SELECT match_id 
            FROM {dataset}
            GROUP BY match_id 
            HAVING MAX(innings) = 2
        """
        matches = conn.execute(query).fetchdf()
    finally:
        conn.close()
    
    if len(matches) == 0:
        return []
        
    if len(matches) < n_matches:
        n_matches = len(matches)
        
    return matches['match_id'].sample(n=n_matches, random_state=random_state).tolist()

def single_match_prediction(df: pd.DataFrame, predictor: WinPredictor) -> Dict:
    """
    Simulate ball-by-ball predictions for a single match and compute squared errors per phase.
    """
    if df.empty:
        return {}

    in2_final = df[df['innings'] == 2]['final_total'].iloc[0]
    target = df[df['innings'] == 2]['target_total'].iloc[0]
    team2_won = 1 if in2_final >= target else 0
    team1_won = 1 - team2_won

    errors_by_stage = {
        'First Innings powerplay': [],
        'First Innings middle': [],
        'First Innings death': [],
        'Second Innings powerplay': [],
        'Second Innings middle': [],
        'Second Innings death': []
    }

    for _, balls in df.iterrows():
        innings = balls['innings']
        phase = balls['phase']

        state = {
            "score": balls["current_score"],
            "wickets": 10 - balls["wickets_in_hand"],
            "balls": balls["legal_balls_bowled"],
            "target": balls["target_total"] if balls['innings'] == 2 else None
        }
        
        innings_name = "First Innings" if innings == 1 else "Second Innings"
        stage_key = f"{innings_name} {phase}"
        
        if stage_key not in errors_by_stage:
            continue
            
        if innings == 1:
            score = predictor.simulate(state, n_sims=100)
            default_state = {
                "score": 0,
                "wickets": 0,
                "balls": 0,
                "target": score + 1
            }
            chase_win_prob = predictor.simulate(default_state, n_sims=100)
            prob = 1 - chase_win_prob
            truth = team1_won
        else:
            prob = predictor.simulate(state, n_sims=100)
            truth = team2_won
        
        error = (prob - truth) ** 2
        errors_by_stage[stage_key].append(error)

    return {stage: np.mean(errors) if errors else None for stage, errors in errors_by_stage.items()}

def evaluate_model(db_path: str, run_model_path: str, wicket_model_path: str, dataset: str, n_matches: int = 100, output_path: str = None, random_state: int = 42) -> Dict:
    """
    Run backtesting evaluation on selected matches and compute overall average errors.
    """
    predictor = WinPredictor(run_model_path=run_model_path, wicket_model_path=wicket_model_path)
    matches = select_matches_for_backtest(db_path, dataset=dataset, n_matches=n_matches, random_state=random_state)
    
    if not matches:
        raise ValueError(f"No valid matches found in table '{dataset}' to run backtest.")
        
    # Query all data for the selected matches in a single call to remove connection overhead inside loop
    conn = duckdb.connect(db_path, read_only=True)
    try:
        placeholders = ",".join(["?"] * len(matches))
        query = f"SELECT * FROM {dataset} WHERE match_id IN ({placeholders})"
        all_data_df = conn.execute(query, matches).fetchdf()
    finally:
        conn.close()
        
    if all_data_df.empty:
        raise ValueError("No data returned from database for selected matches.")
        
    # Group by match_id to prepare separate dataframes for each match prediction
    match_dfs = [group for _, group in all_data_df.groupby("match_id")]
    
    # Run process-based parallelism to bypass the GIL and utilize multiple cores
    results = Parallel(n_jobs=-1, prefer='processes')(
        delayed(single_match_prediction)(match_df, predictor) 
        for match_df in tqdm(match_dfs, desc=f"Backtesting {dataset}")
    )
    
    results = [r for r in results if r]
    
    if not results:
        raise ValueError("All match simulations returned empty results.")
        
    avg_errors = {
        stage: float(np.mean([r[stage] for r in results if r.get(stage) is not None])) 
        for stage in results[0].keys()
    }
    
    summary = {
        "dataset": dataset,
        "total_matches_tested": len(results),
        "run_model": run_model_path,
        "wicket_model": wicket_model_path,
        "average_errors": avg_errors
    }
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=4)
            
    return summary

def main():
    parser = argparse.ArgumentParser(description="Standalone Model Backtesting & Evaluation Script")
    parser.add_argument("--database", required=True, help="Path to DuckDB database (e.g. data/willow.db)")
    parser.add_argument("--run-model", required=True, help="Path to LightGBM run model (.txt)")
    parser.add_argument("--wicket-model", required=True, help="Path to LightGBM wicket model (.txt)")
    parser.add_argument("--dataset", required=True, help="Database table/dataset to test (e.g. t20_ball_by_ball)")
    parser.add_argument("--n-matches", type=int, default=100, help="Number of matches to backtest")
    parser.add_argument("--output-json", help="Path to write evaluation metrics JSON summary file")
    parser.add_argument("--random-state", type=int, default=42, help="Seed used for match selection sampling")
    
    args = parser.parse_args()
    
    try:
        print(f"Starting evaluation of models for dataset '{args.dataset}' using database '{args.database}'...")
        summary = evaluate_model(
            db_path=args.database,
            run_model_path=args.run_model,
            wicket_model_path=args.wicket_model,
            dataset=args.dataset,
            n_matches=args.n_matches,
            output_path=args.output_json,
            random_state=args.random_state
        )
        
        print("\nEvaluation Completed Successfully!")
        print(f"Total Matches Tested: {summary['total_matches_tested']}")
        print("Average Squared Errors by Phase:")
        for phase, err in summary['average_errors'].items():
            print(f"  - {phase}: {err:.5f}")
            
        if args.output_json:
            print(f"Results logged to: {args.output_json}")
            
    except Exception as e:
        print(f"Error occurred during evaluation: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
