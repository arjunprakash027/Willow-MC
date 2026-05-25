import pandas as pd
from dagster import asset, MaterializeResult, MetadataValue
from dagster_duckdb import DuckDBResource
import statsmodels.api as sm
from statsmodels.api import Logit
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from lightgbm import LGBMClassifier
import json
import os

FEATURES = ["rr", "required_run_rate", "wickets_in_hand", "is_second_innings", "current_score", "overs_remaining"]

def train_runs_model(duckdb: DuckDBResource, table_name: str, file_prefix: str):
    with duckdb.get_connection() as conn:
        train_df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        
    X = train_df[FEATURES]
    X = sm.add_constant(X)
    y_runs = train_df["target_runs_next_n_balls"]

    model_runs = sm.NegativeBinomial(y_runs, X).fit(disp=0)
    
    run_coeffs = model_runs.params.to_dict()
    os.makedirs("outputs", exist_ok=True)
    with open(f"outputs/{file_prefix}_run_model_coeffs.json", "w") as f:
        json.dump(run_coeffs, f)

    return MaterializeResult(
        metadata={
            "aic": float(model_runs.aic),
            "log_likelihood": float(model_runs.llf),
            "sample_size": len(train_df),
            "coefficients": MetadataValue.json(run_coeffs)
        }
    )

def train_wickets_model(duckdb: DuckDBResource, table_name: str, file_prefix: str):
    with duckdb.get_connection() as conn:
        train_df = conn.execute(f"SELECT * FROM {table_name}").fetchdf()
        
    X = train_df[FEATURES]
    y = train_df["wicket_event"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {'max_depth': 8,
        'num_leaves': 56,
        'min_child_samples': 118,
        'subsample': 0.6827821817802229,
        'colsample_bytree': 0.9634968128793103,
        'n_estimators': 1921,
        'learning_rate': 0.046621616348413115,
        'min_gain_to_split': 0.2593050460541686,
        'reg_alpha': 0.003326750254522071,
        'reg_lambda': 1.812693851694228e-07,
        'extra_trees': False
        }

    wicket_model = LGBMClassifier(**params)

    wicket_model.fit(X_train, y_train)

    y_pred_prob = wicket_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    os.makedirs("outputs", exist_ok=True)
    model_path = f"outputs/{file_prefix}_wicket_model.txt"
    wicket_model.booster_.save_model(model_path)

    return MaterializeResult(
        metadata={
            "precision": float(precision),
            "recall": float(recall),
            "sample_size": len(train_df),
            "model_path": model_path
        }
    )

@asset(deps=['next_n_balls_features_t20'], group_name="models", compute_kind="python")
def t20_balls_model(context, duckdb: DuckDBResource):
    return train_runs_model(duckdb, "next_n_balls_features_t20_ball_by_ball", "t20_int")

@asset(deps=['next_n_balls_features_t20'], group_name="models", compute_kind="python")
def t20_wickets_model(context, duckdb: DuckDBResource):
    return train_wickets_model(duckdb, "next_n_balls_features_t20_ball_by_ball", "t20_int")

@asset(deps=['next_n_balls_features_odi'], group_name="models", compute_kind="python")
def odi_balls_model(context, duckdb: DuckDBResource):
    return train_runs_model(duckdb, "next_n_balls_features_odi_ball_by_ball", "odi_int")

@asset(deps=['next_n_balls_features_odi'], group_name="models", compute_kind="python")
def odi_wickets_model(context, duckdb: DuckDBResource):
    return train_wickets_model(duckdb, "next_n_balls_features_odi_ball_by_ball", "odi_int")

@asset(deps=['next_n_balls_features_ipl'], group_name="models", compute_kind="python")
def ipl_balls_model(context, duckdb: DuckDBResource):
    return train_runs_model(duckdb, "next_n_balls_features_ipl_ball_by_ball", "t20_ipl")

@asset(deps=['next_n_balls_features_ipl'], group_name="models", compute_kind="python")
def ipl_wickets_model(context, duckdb: DuckDBResource):
    return train_wickets_model(duckdb, "next_n_balls_features_ipl_ball_by_ball", "t20_ipl")