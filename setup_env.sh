#!/usr/bin/env bash
# setup_env.sh — Create a clean venv and install all dependencies

set -e

PYTHON=${PYTHON:-python3.11}
VENV_DIR=".venv"

echo "Creating virtual environment in $VENV_DIR …"
$PYTHON -m venv "$VENV_DIR"

echo "Activating venv and installing dependencies …"
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r requirements.txt

echo ""
echo "✅ Environment ready."
echo ""
echo "To activate:  source $VENV_DIR/bin/activate"
echo ""
echo "Then run the pipeline:"
echo "  python run_pipeline.py"
echo ""
echo "Then start the API:"
echo "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
