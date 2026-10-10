#!/usr/bin/env bash
# ==============================================================================
# ReValue AI — Unified Production Service Starter
# ==============================================================================
# Launches both the FastAPI REST microservice and Streamlit Web Application.
# Handles graceful termination (Ctrl+C kills all background processes cleanly).

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "⚡ Starting ReValue AI Electronics Valuation Platform"
echo "=========================================================="

# 1. Activate Virtual Environment
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "📦 Activating virtual environment (../venv)..."
    source ../venv/bin/activate
else
    echo "❌ Error: Virtual environment 'venv' not found."
    echo "Please create one using: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# 2. Check and clean up existing ports if already running
cleanup_stale() {
    local port=$1
    local pids=$(lsof -ti :$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "⚠️  Port $port is in use by PID(s): $pids. Stopping existing process..."
        kill -9 $pids 2>/dev/null || true
        sleep 1
    fi
}

echo "🧹 Checking ports 8000 and 8501..."
cleanup_stale 8000
cleanup_stale 8501

# 3. Setup cleanup trap for graceful exit on Ctrl+C or kill
cleanup() {
    echo ""
    echo "🛑 Shutting down ReValue AI services..."
    if [ -n "$FASTAPI_PID" ] && kill -0 "$FASTAPI_PID" 2>/dev/null; then
        echo "   Killing FastAPI (PID: $FASTAPI_PID)..."
        kill "$FASTAPI_PID" 2>/dev/null || true
    fi
    if [ -n "$STREAMLIT_PID" ] && kill -0 "$STREAMLIT_PID" 2>/dev/null; then
        echo "   Killing Streamlit (PID: $STREAMLIT_PID)..."
        kill "$STREAMLIT_PID" 2>/dev/null || true
    fi
    echo "✅ All services stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 4. Start FastAPI Backend in background
echo "🚀 Launching FastAPI microservice (port 8000)..."
uvicorn src.api:app --host 127.0.0.1 --port 8000 --log-level info &
FASTAPI_PID=$!

# Wait for FastAPI to become ready
echo "⏳ Waiting for FastAPI health endpoint..."
MAX_RETRIES=20
COUNT=0
until curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; do
    sleep 0.5
    COUNT=$((COUNT + 1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo "❌ FastAPI failed to start within timeout."
        exit 1
    fi
done
echo "✅ FastAPI is healthy and serving predictions!"

# 5. Print URLs
echo "=========================================================="
echo "🌐 Services are LIVE and operational:"
echo "   • Streamlit App:     http://localhost:8501"
echo "   • FastAPI Docs:      http://localhost:8000/docs"
echo "   • Health Check:      http://localhost:8000/health"
echo "   • Drift Telemetry:   http://localhost:8000/monitor/drift"
echo "=========================================================="
echo "Press Ctrl+C to stop both services."
echo ""

# 6. Launch Streamlit Frontend in foreground
streamlit run src/dashboard.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false
