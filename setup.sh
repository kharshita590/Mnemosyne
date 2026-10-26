#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== Setting up Mnemosyne ==="

if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    echo "1. Starting stack via Docker Compose..."
    docker compose up -d --build
    echo "2. Waiting for healthcheck..."
    sleep 5
    curl -s http://localhost:8080/health || echo "Starting up..."
    echo ""
    echo "Mnemosyne running at http://localhost:8080"
    exit 0
fi

echo "Docker not found. Falling back to local virtualenv setup."

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

uv sync
source .venv/bin/activate
alembic upgrade head
python mcp_local/server.py
