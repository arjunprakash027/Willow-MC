import os
import sys
import numpy as np
from typing import Dict
from src.predictor import WinPredictor

from joblib import Parallel, delayed
from tqdm import tqdm
from dagster import asset, MaterializeResult
from dagster_duckdb import DuckDBResource

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


def single_match_prediction(duckdb: DuckDBResource ,match_id: str, predictor: WinPredictor, dataset: str) -> Dict:

    with duckdb.get_connection() as conn:
        df = conn.execute(f"SELECT * FROM {dataset} WHERE match_id = ?", [match_id]).fetchdf()
    
    in1_final = df[df['innings'] == 1]['final_total'].iloc[0]
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

def select_matches_for_backtest(duckdb: DuckDBResource, n_matches: int = 1000, dataset: str = None):
    with duckdb.get_connection() as conn:
        # Get matches that have exactly 2 innings (no super overs, not washed out in 1st innings)
        query = f"""
            SELECT match_id 
            FROM {dataset}
            GROUP BY match_id 
            HAVING MAX(innings) = 2
        """
        matches = conn.execute(query).fetchdf()
    
    return matches['match_id'].sample(n=n_matches).tolist()

def backtest_models(context, duckdb: DuckDBResource, run_model_path: str, wicket_model_path: str, dataset: str):

    predictor = WinPredictor(run_model_path=run_model_path, wicket_model_path=wicket_model_path)
    matches = select_matches_for_backtest(duckdb, n_matches=100, dataset=dataset)
    
    results = Parallel(n_jobs=4, prefer='threads')(delayed(single_match_prediction)(duckdb, match_id, predictor, dataset) for match_id in tqdm(matches, desc=f"Backtesting {dataset}"))
    
    avg_errors = {stage: np.mean([r[stage] for r in results if r[stage] is not None]) for stage in results[0].keys()}
    
    return MaterializeResult(
        metadata={
            "average_errors": avg_errors,
            "total_matches": len(matches)
        }
    )

@asset(deps=['t20_balls_model', 't20_wickets_model'], group_name="backtesting", compute_kind="python")
def backtest_t20_model(context, duckdb: DuckDBResource):

    return backtest_models(
        context,
        duckdb,
        "outputs/t20_int_run_model.txt",
        "outputs/t20_int_wicket_model.txt",
        "t20_ball_by_ball"
    )
    
@asset(deps=['ipl_balls_model', 'ipl_wickets_model'], group_name="backtesting", compute_kind="python")
def backtest_ipl_model(context, duckdb: DuckDBResource):

    return backtest_models(
        context,
        duckdb,
        "outputs/t20_ipl_run_model.txt",
        "outputs/t20_ipl_wicket_model.txt",
        "ipl_ball_by_ball"
    )