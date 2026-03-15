# Willow-MC 🏏

A stochastic cricket forecasting pipeline and real-time arbitrage monitor.

## 🏗 Architecture

- **Orchestration**: [Dagster](https://dagster.io/) for modular ingestion, transformation, and modeling.
- **Data Storage**: [DuckDB](https://duckdb.org/) analytical database for ball-by-ball records and feature storage.
- **Modeling**: Negative Binomial (runs) and Logistic Regression (wickets), persisted as JSON coefficients.
- **Inference**: High-fidelity Monte Carlo engine for simulating match outcomes.
- **Trading**: Real-time arbitrage monitoring across Cricbuzz and Polymarket.

## 🚀 Quickstart

**1. Install Dependencies**
```bash
uv pip install -r requirements.txt
```

**2. Data Pipeline (Dagster)**
```bash
# Start the dev server and open http://localhost:3000
dagster dev -m orchestrator
```
Materialize the assets in order:
*   `raw_data` (Bronze): Downloads latest Cricsheet ball-by-ball data.
*   `curate_dataset` (Silver): Parses match JSONs into structured DuckDB tables.
*   `next_n_balls_features_t20` (Gold): Parameterized feature engineering for T20 format.
*   `t20_balls_model` & `t20_wickets_model`: Statistical training with statsmodels, exporting to `outputs/`.

**3. Live Monitor**
```bash
# Usage: python minimal_monitor.py <match_id> [redis_ip] [slug]
python minimal_monitor.py 100001 localhost crint-ind-nzl-2026-03-08
```

## 📂 Structure
*   `orchestrator/assets/`: Modular assets for `ingestion`, `transformations`, `features`, and `models`.
*   `outputs/`: Generated model coefficients (.json) used for live inference.
*   `minimal_monitor.py`: Real-time terminal UI for market vs. model edge.
*   `data/willow.db`: Persistent DuckDB storage for the entire history of cricket.
