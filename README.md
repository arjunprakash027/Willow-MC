# Willow-MC 🏏

A stochastic cricket forecasting pipeline and real-time arbitrage monitor.

## 🏗 Architecture

- **Orchestration**: [Dagster](https://dagster.io/) pipelines for data ingestion and transformation.
- **Data Storage**: [DuckDB](https://duckdb.org/) for high-performance, in-process analytical queries.
- **Modeling**: Negative Binomial (runs) and Logistic Regression (wickets).
- **Inference**: Monte Carlo simulation engine for ball-by-ball win probabilities.
- **Live Trading**: Real-time Cricbuzz ingestion and Polymarket orderbook monitoring via Redis.

## 🚀 Quickstart

**1. Install Dependencies**
```bash
uv pip install -r requirements.txt
```

**2. Data Pipeline (Dagster)**
Launch the orchestrator to download Cricsheet data, parse the matches, and populate DuckDB.
```bash
dagster dev -m orchestrator
```
Open `http://localhost:3000` to materialize the assets:
*   `raw_data` (Bronze): Downloads zip archives.
*   `curate_dataset` (Silver): Transforms JSONs into a ball-by-ball DuckDB table.
*   `next_n_balls_features` (Gold): Generates N-ball state features for ML.

**3. Live Monitor**
Run the real-time inference engine against live market prices. *(Requires Redis)*
```bash
# Usage: python minimal_monitor.py <match_id> [redis_ip] [slug]
python minimal_monitor.py 100001 localhost crint-ind-nzl-2026-03-08
```

## 📂 Structure
*   `orchestrator/`: Dagster definitions, resources, and DuckDB-backed assets.
*   `notebooks/`: Research, Monte Carlo simulations, and model training.
*   `minimal_monitor.py`: The live terminal UI for inference vs. markets.
*   `data/willow.db`: DuckDB persistent storage (generated automatically).
