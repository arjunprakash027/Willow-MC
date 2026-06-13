import os
import sys
from dagster import asset, MaterializeResult
from dagster_duckdb import DuckDBResource

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.evaluation import evaluate_model  # noqa: E402

def backtest_models(context, duckdb: DuckDBResource, run_model_path: str, wicket_model_path: str, dataset: str):
    # Retrieve the database path configured in the DuckDB resource
    db_path = duckdb.database
    
    # Run the modular evaluation
    output_path = f"outputs/{dataset}_backtest_results.json"
    results = evaluate_model(
        db_path=db_path,
        run_model_path=run_model_path,
        wicket_model_path=wicket_model_path,
        dataset=dataset,
        n_matches=100,
        output_path=output_path
    )
    
    return MaterializeResult(
        metadata={
            "average_errors": results["average_errors"],
            "total_matches": results["total_matches_tested"],
            "output_path": output_path
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