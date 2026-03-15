import duckdb
from dagster import asset, MaterializeResult
from dagster_duckdb import DuckDBResource
import pandas as pd

def compute_next_n_balls_features(df: pd.DataFrame, total_balls: int, n_balls: int = 6):
    rows = []
    df = df.sort_values(["match_id", "innings", "over_number", "ball_in_over"])
    
    for (match_id, innings), g in df.groupby(["match_id", "innings"]):
        g = g.reset_index(drop=True)
        is_second_innings = 1 if innings == 2 else 0

        for i in range(len(g) - n_balls):
            cur = g.loc[i]
            nxt = g.loc[i+1 : i+n_balls]

            runs_next_n = nxt["total_runs_ball"].sum()
            wickets_next_n = nxt["is_wicket"].sum()
            wicket_event = int(wickets_next_n > 0)

            legal_balls = max(cur["legal_balls_bowled"], 1)
            rr = cur["current_score"] / legal_balls * 6
            overs_remaining = (total_balls - legal_balls) / 6
            target = cur["target_total"] if is_second_innings else 0

            if is_second_innings and target > 0:
                runs_remaining = max(target - cur["current_score"], 0)
                rem_balls = max(total_balls - legal_balls, 1)
                rrr = runs_remaining / rem_balls * 6
            else:
                rrr = 0

            rows.append({
                "match_id": match_id,
                "innings": innings,
                "is_second_innings": is_second_innings,
                "rr": rr,
                "required_run_rate": rrr,
                "overs_remaining": overs_remaining,
                "wickets_in_hand": cur["wickets_in_hand"],
                "current_score": cur["current_score"],
                "target_total": target,
                "target_runs_next_n_balls": runs_next_n,
                "wickets_next_n_balls": wickets_next_n,
                "wicket_event": wicket_event
            })
            
    return pd.DataFrame(rows)

@asset(deps=['curate_dataset'], group_name="gold", compute_kind="python")
def next_n_balls_features_t20(context, duckdb: DuckDBResource):
    with duckdb.get_connection() as conn:
        df = conn.execute("SELECT * FROM ball_by_ball").fetchdf()

    train_df = compute_next_n_balls_features(df, total_balls=120)

    context.log.info(f"Generated {len(train_df)} features for T20...")
    
    with duckdb.get_connection() as conn:
        conn.execute("CREATE OR REPLACE TABLE next_n_balls_features_t20 AS SELECT * FROM train_df")

    return MaterializeResult(
        metadata={
            "total_rows": len(train_df),
            "table_name": "next_n_balls_features_t20",
            "n_balls": 6
        }
    )
