#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for Cygnus Pyramid (dev)
# Usage:
#   ./launch.sh start        # start backend + frontend (default)
#   START_LLAMAS=1 ./launch.sh start   # also start llama servers via ./start_servers.sh
#   OPEN_BROWSER=1 ./launch.sh start   # attempt to open http://localhost:5173/ (desktop only)
#   ./launch.sh stop         # stop services started by this script
#   ./launch.sh status       # show status

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

BACKEND_LOG="$LOG_DIR/backend.log"
FRONTEND_LOG="$LOG_DIR/frontend.log"
START_SERVERS_LOG="$LOG_DIR/start_servers.log"

BACKEND_PID_FILE="$LOG_DIR/backend.pid"
FRONTEND_PID_FILE="$LOG_DIR/frontend.pid"
START_SERVERS_PID_FILE="$LOG_DIR/start_servers.pid"

python_cmd() {
    # Prefer project virtualenv Python if present, then fallback to system python
    if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
        echo "$PROJECT_DIR/.venv/bin/python"
        return
    fi
    if [ -x "$PROJECT_DIR/venv/bin/python" ]; then
        echo "$PROJECT_DIR/venv/bin/python"
        return
    fi
    command -v python3 >/dev/null 2>&1 && echo python3 || echo python
}

start_backend() {
    if [ -f "$BACKEND_PID_FILE" ] && kill -0 "$(cat "$BACKEND_PID_FILE")" 2>/dev/null; then
        echo "Backend already running (pid $(cat $BACKEND_PID_FILE))."
        return
    fi
    echo "Starting backend... (logs: $BACKEND_LOG)"
    # Ensure the project root is on PYTHONPATH so `python -m backend.main` resolves local package
    nohup env PYTHONPATH="$PROJECT_DIR" $(python_cmd) -m backend.main > "$BACKEND_LOG" 2>&1 &
    echo $! > "$BACKEND_PID_FILE"
    sleep 1
    echo "Backend started (pid $(cat $BACKEND_PID_FILE))."
}

start_frontend() {
    # Prefer nested frontend (frontend/frontend) if present (matches CI/workflow behavior)
    if [ -d "$PROJECT_DIR/frontend/frontend" ] && [ -f "$PROJECT_DIR/frontend/frontend/package.json" ]; then
        FRONTEND_ROOT="$PROJECT_DIR/frontend/frontend"
        echo "Detected nested frontend at $FRONTEND_ROOT; using that."
    else
        if [ ! -d "$PROJECT_DIR/frontend" ]; then
            echo "No frontend directory found; skipping frontend start."
            return
        fi
        FRONTEND_ROOT="$PROJECT_DIR/frontend"
    fi

    if [ -f "$FRONTEND_PID_FILE" ] && kill -0 "$(cat "$FRONTEND_PID_FILE")" 2>/dev/null; then
        echo "Frontend already running (pid $(cat $FRONTEND_PID_FILE))."
        return
    fi
    echo "Ensuring frontend deps in $FRONTEND_ROOT..."
    if [ ! -d "$FRONTEND_ROOT/node_modules" ]; then
        (cd "$FRONTEND_ROOT" && npm install --no-audit --no-fund)
    fi
    echo "Starting frontend (Vite) from $FRONTEND_ROOT... (logs: $FRONTEND_LOG)"
    # Use bash -lc so variables expand correctly inside the command string
    nohup bash -lc "cd \"$FRONTEND_ROOT\" && npm run dev --silent" > "$FRONTEND_LOG" 2>&1 &
    echo $! > "$FRONTEND_PID_FILE"
    sleep 1
    echo "Frontend started (pid $(cat $FRONTEND_PID_FILE))."
}

start_llama_servers() {
    if [ ! -x "$PROJECT_DIR/start_servers.sh" ]; then
        echo "No executable start_servers.sh found in project root. To start local LLM servers, run: ./start_servers.sh (ensure a GGUF model exists at the expected path or edit the script)."
        return
    fi
    echo "Starting llama servers (logs: $START_SERVERS_LOG). Use FORCE_KILL_PORTS=1 to override port owners if desired."
    nohup env FORCE_KILL_PORTS="${FORCE_KILL_PORTS:-0}" "$PROJECT_DIR/start_servers.sh" > "$START_SERVERS_LOG" 2>&1 &
    echo $! > "$START_SERVERS_PID_FILE"
    sleep 1
    echo "start_servers.sh invoked (pid $(cat $START_SERVERS_PID_FILE)). Check $START_SERVERS_LOG for details."
}

open_browser() {
    URL="http://localhost:5173/"
    if [ "${OPEN_BROWSER:-0}" != "1" ]; then
        return
    fi
    if command -v xdg-open >/dev/null 2>&1 && ( [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ] ); then
        echo "Opening browser to $URL"
        xdg-open "$URL" >/dev/null 2>&1 || true
    else
        echo "Skipping browser open (no desktop DISPLAY or xdg-open)." 
    fi
}

stop_service() {
    pidfile="$1"
    name="$2"
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (pid $pid)..."
            kill "$pid" || true
            sleep 1
            if kill -0 "$pid" 2>/dev/null; then
                echo "$name did not exit; killing..."
                kill -9 "$pid" || true
            fi
        else
            echo "$name PID file exists but process $pid not running. Cleaning up."
        fi
        rm -f "$pidfile"
    else
        echo "No pid file for $name at $pidfile"
    fi
}

status() {
    echo "== Status =="
    for f in "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE" "$START_SERVERS_PID_FILE"; do
        if [ -f "$f" ]; then
            pid=$(cat "$f")
            if kill -0 "$pid" 2>/dev/null; then
                echo "$(basename $f): running (pid $pid)"
            else
                echo "$(basename $f): stale pidfile (pid $pid)"
            fi
        else
            echo "$(basename $f): not running"
        fi
    done
    echo "Vite URL: http://localhost:5173/"
    echo "Backend status endpoint: http://localhost:8001/api/status"
}

case "${1:-start}" in
    start)
        start_backend
        start_frontend
        if [ "${START_LLAMAS:-0}" = "1" ]; then
            start_llama_servers
        else
            echo "LLM servers not started. To start them set START_LLAMAS=1 environment variable."
        fi
        open_browser
        ;;
    stop)
        stop_service "$BACKEND_PID_FILE" "backend"
        stop_service "$FRONTEND_PID_FILE" "frontend (vite)"
        stop_service "$START_SERVERS_PID_FILE" "start_servers.sh"
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {start|stop|status}"
        exit 2
        ;;
esac
