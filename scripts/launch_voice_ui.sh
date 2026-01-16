#!/bin/bash

# Launch script for Voice UI
# Starts both the Flask voice server and Next.js frontend

echo "Starting EchoSee Voice UI..."

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Function to cleanup background processes
cleanup() {
    echo "Stopping servers..."
    kill $VOICE_SERVER_PID 2>/dev/null
    kill $NEXT_SERVER_PID 2>/dev/null
    exit 0
}

trap cleanup EXIT INT TERM

# Start voice server
echo "Starting voice API server on port 9004..."
cd "$PROJECT_ROOT"
python -m api.voice_server &
VOICE_SERVER_PID=$!

# Wait a bit for server to start
sleep 2

# Start Next.js dev server
echo "Starting Next.js frontend on port 9003..."
cd "$PROJECT_ROOT/echonoti"
npm run dev &
NEXT_SERVER_PID=$!

echo ""
echo "Voice UI is running!"
echo "  - Voice API: http://localhost:9004"
echo "  - Web UI: http://localhost:9003/voice"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for both processes
wait
