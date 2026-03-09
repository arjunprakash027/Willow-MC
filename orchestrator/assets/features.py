"""
Features for ML models will be created here
"""

import duckdb
from dagster import asset, MaterializeResult
from dagster_duckdb import DuckDBResource
import pandas as pd

@asset(deps=['curate_dataset'], group_name="gold", compute_kind="python")
def next_n_balls_features(context, duckdb: DuckDBResource):
    """
    Creates features for ML models using the next n balls.
    """
    N_BALLS = 6
    with duckdb.get_connection() as conn:

        df = conn.execute("SELECT * FROM ball_by_ball").fetchdf()

        rows = []

        df = df.sort_values(
            ["match_id", "innings", "over_number", "ball_in_over"]
        )

        df["ball_index"] = (
            df.groupby(["match_id", "innings"])
            .cumcount()
        )

        for (match_id, innings), g in df.groupby(["match_id", "innings"]):

            g = g.reset_index(drop=True)

            is_second_innings = 1 if innings == 2 else 0

            # Ensure we don't go out of bounds for the last 6 balls of an inning
            for i in range(len(g) - N_BALLS):

                cur = g.loc[i]
                
                # Grab the next 6 balls (loc is inclusive on both ends, so i+1 to i+6 gets exactly 6 balls)
                nxt = g.loc[i+1:i+N_BALLS]

                # Calculate aggregates over the next 6 balls
                runs_next_6_balls = nxt["total_runs_ball"].sum()
                wickets_next_6_balls = nxt["is_wicket"].sum()
                wicket_event = int(wickets_next_6_balls > 0)

                legal_balls = max(cur["legal_balls_bowled"], 1)

                rr = cur["current_score"] / legal_balls * 6

                overs_remaining = (120 - legal_balls) / 6

                target = cur["target_total"] if is_second_innings else 0

                if is_second_innings and target > 0:
                    runs_remaining = max(target - cur["current_score"], 0)
                    balls_remaining = max(120 - legal_balls, 1)
                    rrr = runs_remaining / balls_remaining * 6
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
                    "target_runs_next_6_balls": runs_next_6_balls,
                    "wickets_next_6_balls": wickets_next_6_balls,
                    "wicket_event": wicket_event
                })

        train_df = pd.DataFrame(rows)

        context.log.info(f"Generated {len(train_df)} next {N_BALLS} ball features...")
        
        with duckdb.get_connection() as conn:
            conn.execute("CREATE OR REPLACE TABLE next_n_balls_features AS SELECT * FROM train_df")

        return MaterializeResult(
            metadata={
                "total_rows": len(train_df),
                "table_name": "next_n_balls_features",
                "n_balls":N_BALLS
            }
    )
        

        
