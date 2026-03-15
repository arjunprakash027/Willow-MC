import pandas as pd
from dagster import asset, MaterializeResult, MetadataValue
from dagster_duckdb import DuckDBResource
import statsmodels.api as sm
from statsmodels.api import Logit
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import json

FEATURES = ["rr", "required_run_rate", "wickets_in_hand", "is_second_innings", "current_score", "overs_remaining"]

@asset(deps=['next_n_balls_features_t20'], group_name="models", compute_kind="python")
def t20_balls_model(context, duckdb: DuckDBResource):

    with duckdb.get_connection() as conn:
        train_df = conn.execute("SELECT * FROM next_n_balls_features_t20").fetchdf()
        X = train_df[FEATURES]
        X = sm.add_constant(X)

        y_runs = train_df["target_runs_next_6_balls"]

        model_runs = sm.NegativeBinomial(y_runs, X).fit()
        
        run_coeffs = model_runs.params.to_dict()

        with open("outputs/t20_int_run_model_coeffs.json", "w") as f:
            json.dump(run_coeffs, f)

        return MaterializeResult(
            metadata={
                "aic":float(model_runs.aic),
                "log_likelihood":float(model_runs.llf),
                "sample_size":len(train_df),
                "coefficients":MetadataValue.json(model_runs.params.to_dict())
            }
        )

@asset(deps=['next_n_balls_features_t20'], group_name="models", compute_kind="python")
def t20_wickets_model(context, duckdb: DuckDBResource):
    
    with duckdb.get_connection() as conn:
        train_df = conn.execute("SELECT * FROM next_n_balls_features_t20").fetchdf()
        X = train_df[FEATURES]
        y = train_df["wicket_event"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        wicket_model = Logit(y_train, X_train).fit()

        y_pred_prob = wicket_model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        wicket_coeffs = wicket_model.params.to_dict()

        with open("outputs/t20_int_wicket_model_coeffs.json", "w") as f:
            json.dump(wicket_coeffs, f)

        return MaterializeResult(
            metadata={
                "precision":float(precision),
                "recall":float(recall),
                "sample_size":len(train_df),
                "coefficients":MetadataValue.json(wicket_model.params.to_dict())
            }
        )
        
        