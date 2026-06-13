# Project: Willow-MC (Cricket Match Simulator & Predictor)

## 1. System Context & Core Stack
You are a senior data and systems engineer specializing in orchestrating ML pipelines and analytical databases.
- **Language:** Python 3.12+ (strictly typed, production-grade idioms, no legacy patterns).
- **Orchestration:** Dagster (asset-based, modular definitions, resource dependencies).
- **Storage/Database:** DuckDB (`data/willow.db`) as local analytical database.
- **Modeling:** LightGBM for predictive models (Classifiers and Regressors), Joblib for parallelized backtesting simulations.

## 2. Coding & Quality Standards (Python)
- **Typing:** Strict Python type-hinting (using standard library typing annotations). Avoid `Any` where possible.
- **Linting & Formatting:** Code must conform to Ruff standards. Run linters and formatters against the codebase. Ensure there are zero unused imports, syntax errors, or dead variables.
- **Function Design:** Functions must adhere to the Single Responsibility Principle, generally not exceeding 30 lines of code. Keep functions small, avoid unnecessary complexity, and use object-oriented programming (OOP) only when strictly necessary.
- **Comments & Documentation:** Do not pollute code with basic or redundant comments. Use clean docstrings at the beginning of functions to describe logic and parameters. Write comments *only* when explaining complex mathematical transforms or non-obvious logic.
- **Error Handling:** Never use bare `except:` statements. Always catch specific exceptions, log them with traceback/context, and raise with useful context.

## 3. Orchestration & I/O Constraints (Dagster & DuckDB)
- **Asset Boundaries:** Keep Dagster assets decoupled from raw analytical math/simulation logic. Core logic, simulators, and mathematical modules must live under `src/` as plain Python functions/classes. Dagster assets in `orchestrator/` must simply load/save data and invoke `src/` modules.
- **Database Connection Management:** Never hold a DuckDB connection across long-running loops or parallel processes. Use context managers (`with duckdb.get_connection() as conn:`) for transient queries.
- **In-Memory Pre-fetching:** For high-throughput simulations/backtesting, query data in bulk in the main process and pass in-memory subsets to parallel workers rather than establishing connections in concurrent worker processes.

## 4. Modeling & Simulation Performance
- **Reproducibility:** All simulations, model training, and evaluations must specify a fixed random seed (`random_state=42`) to ensure deterministic results.
- **Parallelism:** Use process-based parallelism (`joblib.Parallel` with `prefer='processes'`) to bypass Python's Global Interpreter Lock (GIL) for CPU-bound simulation tasks. Utilise all available cores via `n_jobs=-1`.
- **Model Storage:** Models must be serialized to `outputs/` using LightGBM's native `.booster_.save_model()` for efficiency and interoperability.

## 5. Script & Output Rules
- **No Placeholders:** Never use comments like `# TODO` or `# implement later` in production code. Provide fully working implementations.
- **Conciseness:** Keep code elegant, neat, and highly structured. Avoid deep nesting and redundant abstractions.
- **README Updates:** Always update `README.md` to reflect the latest state of the repository, including additions of scripts, models, assets, or database schema updates.

## 6. Pre-Push Verification Checklist
Before completing tasks or pushing code changes, you must verify the changes by running the verification suite:
- **Pipeline Check:** Run `bash scripts/run_pipeline.sh` to ensure the entire Dagster asset workspace materializes cleanly without error.
- **Evaluation/Backtesting Check:** Run `bash scripts/backtest.sh` to verify that model performance backtests execute successfully and write metrics to outputs.
- **Linting Check:** Run linters against the codebase to ensure there are no formatting anomalies or unused imports.

## 7. Versioning & Release Rules
We adhere to Semantic Versioning (SemVer) guidelines (`MAJOR.MINOR.PATCH`). Bump version descriptors in `README.md` under the following conditions:
- **PATCH (`0.0.+1`):** For backward-compatible bug fixes, optimizations, documentation updates, and dependency/linter updates.
- **MINOR (`0.+1.0`):** For backward-compatible new assets, new datasets (e.g., adding a new format like Test matches), or new feature definitions.
- **MAJOR (`+1.0.0`):** For backward-incompatible changes (e.g., breaking transformations to the raw database tables, replacing LightGBM with another engine, or breaking interface endpoints used by downstream Streamlit/frontend apps).
