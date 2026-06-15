# Willow-MC 🏏

A high-fidelity, stochastic cricket forecasting engine, live match probability monitor, and comprehensive data pipeline powered by Dagster, DuckDB, and Monte Carlo methods.

---

## 📖 Overview

Willow-MC moves beyond static, rule-based cricket forecasting by simulating the remaining balls of a match thousands of times using dynamically parameterized statistical models. It natively incorporates match state (wickets in hand, required run rates, overs remaining) to forecast outcomes. 

The system consists of three main components:
1. **The Orchestration Pipeline**: Ingests historical ball-by-ball data, engineers predictive features, trains regression models, and executes parallel backtesting.
2. **The Monte Carlo Engine (`src/predictor.py`)**: A pure, stateless class that simulates game states using the modeled distributions.
3. **The Live Terminal Monitor**: A CLI tool that polls live data from Cricbuzz and passes it into the engine to calculate real-time, second-by-second win probabilities.

---

## ➗ The Mathematics

The Monte Carlo engine relies on passing the current match state (Runs, Wickets, legal balls, target) into two core components that project individual ball outcomes:

- **Run Generation (Negative Binomial)**: Generating the number of runs scored off an expected delivery is modeled as a Negative Binomial distribution. Features like Current Run Rate (CRR), Required Run Rate (RRR), and Wickets in Hand influence the `mu` (mean) and `alpha` dispersion parameters.
- **Wicket Probability (LightGBM)**: The probability of a wicket falling on any given ball is modeled using a LightGBM (`LGBMClassifier`) model. Features like Current Run Rate (CRR), Required Run Rate (RRR), Wickets in Hand, Current Score, and Innings are passed into a gradient-boosted decision tree classifier to capture non-linear relationships and thresholds, scaling aggressively during high RRR chases or "death" over scenarios where teams take immense risks.

By sampling these distributions thousands of times (`n_sims=5000`), the engine produces a dense, converged probability of victory or projected 1st innings score.

---

## 🏗 System Architecture

- **Orchestration**: [Dagster](https://dagster.io/) is used for defining and visualizing the data pipeline assets.
- **Data Warehousing**: [DuckDB](https://duckdb.org/) is leveraged as a ridiculously fast, embedded analytical database storing millions of historical records.
- **Concurrency**: Parallelization via `joblib` allows for massive, concurrent back-testing of historical games. 

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install the dependencies using `uv` (or `pip`):
```bash
uv pip install -r requirements.txt
```

### 2. Run the Dagster Pipeline
Boot the pipeline orchestration UI to manage historical data and train networks:
```bash
dagster dev -m orchestrator
```
*(Navigates to http://localhost:3000)*

Materialize the assets in the following order:
1. **Bronze (`raw_data`)**: Pulls the master ball-by-ball datasets from Cricsheet.
2. **Silver (`curate_dataset`)**: Parses chaotic JSONs and dumps them structured into DuckDB tables.
3. **Gold (`next_n_balls_features_t20`)**: Computes contextual window metrics (features) per ball.
4. **Modeling (`t20_balls_model` & `t20_wickets_model`)**: Fits the data. The runs model is trained via Negative Binomial regression (`statsmodels`) and saved as `.json` coefficient files; the wickets model is trained via LightGBM (`LGBMClassifier`) and saved as `.txt` booster files in the `outputs/` directory.

### 3. Backtesting
You can execute the backtests in two ways:
* **Standalone CLI (Fast - Optimized):** Runs the backtesting simulation loop externally bypassing the GIL and using in-memory pre-fetching. Run:
  ```bash
  bash scripts/backtest.sh
  ```
  This will execute the evaluation for both IPL and T20 datasets and dump results to `outputs/`.
* **Dagster Pipeline:** You can execute the `backtesting` asset group directly from the Dagster webserver, which delegates to the same optimized evaluation core.

It computes the **Mean Squared Error (MSE)** dynamically across the 6 major phases of a match:
- **First Innings**: Powerplay (1-6), Middle (7-15), Death (16-20)
- **Second Innings**: Powerplay (1-6), Middle (7-15), Death (16-20)

### 4. Real-time Live Monitor
Once your models are built and generated in the `outputs/` folder, run the live probability monitor against an ongoing match. You will need the Cricbuzz match ID (found in their URL):
```bash
python minimal_monitor.py 100001
```

---

## 📂 Project Structure

```text
Willow-MC/
├── src/
│   ├── predictor.py         # The core Monte Carlo class `WinPredictor`. Decoupled from any live API.
│   └── evaluation.py        # Optimized, standalone model backtesting core and CLI parser.
├── orchestrator/
│   ├── assets/              # Dagster modular assets for modeling, features, and backtesting.
│   └── __init__.py          # Definition of Dagster jobs/schedules.
├── scripts/
│   ├── run_pipeline.sh      # Shell wrapper to materialize the pipeline once via CLI.
│   └── backtest.sh          # Shell wrapper to run standalone optimized backtests.
├── notebooks/               # Jupyter workbooks for exploratory analysis and ad-hoc queries.
├── data/                    # Local embedded DB files (willow.db via DuckDB).
├── outputs/                 # Exported JSON/txt model files and backtest evaluation results.
├── minimal_monitor.py       # Live match tracking CLI. Polling Cricbuzz state -> `WinPredictor`.
├── requirements.txt         # Project runtime dependencies (includes ruff for linting).
└── README.md
```

---

## 📝 Changelog

### v1.3.0 (2026-06-15)
* **Dynamic Live Match Selector**: Implemented dynamic live match scraping from Cricbuzz, allowing selection of ongoing IPL, T20, and ODI matches directly from the Streamlit sidebar.
* **Cached Session State**: Integrated match fetching with Streamlit session state and added a manual refresh button to prevent redundant network requests and avoid IP blocks.

### v1.2.0 (2026-06-13)
* **Modular Evaluation Script (`src/evaluation.py`)**: Separated backtesting math and execution loops from Dagster assets. Created a CLI interface to allow running evaluations standalone.
* **Performance Optimization (~3x Speedup)**:
  * Switched from thread-based to process-based parallelism (`prefer='processes'`, `n_jobs=-1`) to bypass Python's GIL.
  * Implemented in-memory data pre-fetching in the parent process, eliminating database connection overhead inside parallel processes.
* **Execution Wrappers**: Added helper scripts `scripts/run_pipeline.sh` and `scripts/backtest.sh` for fast CLI invocation.
* **Linting & Code Quality**: Added `ruff` to project requirements and ran code-quality fixes across all modified python assets.

### v1.1.0 (2026-05-25)
* **Wickets Model Upgrade**: Upgraded the modeling of wickets falling from a simple Logistic Regression (Logit link) model to a LightGBM Classifier (`LGBMClassifier`).
* **Non-linear Interactions**: The LightGBM classifier captures complex, non-linear relationships and interactions between match state variables (such as wickets in hand, required run rate, current score, overs remaining, and innings).
* **Export Format Update**: The trained wickets models (`t20`, `ipl`, and `odi`) are now exported as booster text files (`outputs/<prefix>_wicket_model.txt`) instead of JSON files.
* **Simulator Integration**: Integrated `lightgbm.Booster` prediction capabilities into the stateless Monte Carlo simulator (`src/predictor.py`), enabling high-accuracy simulation runs.

