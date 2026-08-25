#!/usr/bin/env bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Ensure virtualenv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install --upgrade pip
    .venv/bin/pip install -r requirements_app.txt
fi

echo "======================================================="
echo "🥔 Potato Leaf Disease AI — Web Platform"
echo "======================================================="
echo "🚀 Starting server at http://localhost:8080 ..."
echo "Press Ctrl+C to stop."
echo "======================================================="

PORT=${PORT:-8080}
.venv/bin/python3 server.py
