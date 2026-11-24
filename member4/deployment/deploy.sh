#!/usr/bin/env bash
# ==============================================================================
# SarcasTone One-Click Startup Script
# ==============================================================================
set -euo pipefail

echo "========================================================"
echo " Starting SarcasTone Multimodal Sarcasm Detection System"
echo "========================================================"

export PYTHONPATH="${PYTHONPATH:-.}:."

# Check for Python 3.11
PYTHON_BIN="python3.11"
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    PYTHON_BIN="python3"
fi

echo "Using Python: $($PYTHON_BIN --version)"

# 1. Verify synchronization
echo "[1/3] Verifying 3-Way Multimodal Synchronization..."
$PYTHON_BIN verify_all_members_sync.py

# 2. Start Backend in Background
echo "[2/3] Starting FastAPI Backend on http://0.0.0.0:8000..."
$PYTHON_BIN -m uvicorn member4.backend.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

trap "echo 'Shutting down services...'; kill -TERM $BACKEND_PID 2>/dev/null || true" EXIT

sleep 2

# 3. Start Frontend Dashboard
echo "[3/3] Starting Streamlit Dashboard on http://localhost:8501..."
streamlit run member4/frontend/app.py --server.port 8501 --server.address 0.0.0.0
