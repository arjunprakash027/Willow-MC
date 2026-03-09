# Willow-MC 🏏

An end-to-end stochastic cricket forecasting pipeline. Features high-frequency ingestion, situational state-transition modeling, and a Monte Carlo inference engine for real-time win probability simulation.

## 🛠 Features
- **Data Curation**: Automated download and processing of Cricsheet T20 data.
- **Modeling**: Negative Binomial (runs) and Logistic Regression (wickets) models.
- **Simulation**: Monte Carlo inference engine for ball-by-ball win probability.
- **Monitoring**: Live match monitoring via Cricbuzz API with real-time Polymarket price comparison.

## 🚀 Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Redis Setup**:
   The live monitor requires a Redis server for Polymarket orderbook streams.
   ```bash
   # Example: Start redis locally
   redis-server
   ```

## 📊 Usage

### 1. Data Curation
Download and transform raw JSON match data into a structured Parquet dataset:
```bash
python curate_dataset.py --download --input data/raw --output data/processed
```

### 2. Live Monitoring & Inference
Monitor a live match and compare predicted win probabilities with market prices:
```bash
# Usage: python minimal_monitor.py <CRICBUZZ_MATCH_ID> <REDIS_IP>
python minimal_monitor.py 100001 localhost
```

## 📂 Project Structure
- `curate_dataset.py`: ETL pipeline for match data.
- `minimal_monitor.py`: Real-time inference and market monitoring.
- `notebooks/`: Research and model training (Monte Carlo, Poisson/Negative Binomial).
- `run_model_coeffs.json`: Pre-trained coefficients for run prediction.
- `wicket_model_coeffs.json`: Pre-trained coefficients for wicket prediction.
- `requirements.txt`: Project dependencies.
