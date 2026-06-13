#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running Dagster pipeline in $PROJECT_ROOT..."
cd "$PROJECT_ROOT"

PYTHONPATH=. .venv/bin/dagster asset materialize -m orchestrator.definitions --select "*"
