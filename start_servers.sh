#!/bin/bash
# CYGNUS PYRAMID - Start llama.cpp servers

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MODEL_PATH="$PROJECT_DIR/models/Falcon-H1-1.5B-Deep-Instruct-Q5_K.gguf"
LLAMA_SERVER="$PROJECT_DIR/llama.cpp/build/bin/llama-server"

# Check model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Build llama.cpp if needed
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "Building llama.cpp..."
    
    if [ ! -d "llama.cpp" ]; then
        git clone https://github.com/ggerganov/llama.cpp.git
    fi
    
    cd llama.cpp
    cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j$(nproc)
    cd ..
    
    LLAMA_SERVER="$PROJECT_DIR/llama.cpp/build/bin/llama-server"
fi

mkdir -p logs

# Preflight port check: ensure ports are free unless FORCE_KILL_PORTS=1
for port in 8081 8082 8083; do
    if ss -ltnp 2>/dev/null | grep -q ":$port "; then
        if [ "$FORCE_KILL_PORTS" = "1" ]; then
            echo "Port $port in use — force-killing processes (FORCE_KILL_PORTS=1)."
            fuser -k $port/tcp 2>/dev/null || true
        else
            echo "ERROR: port $port is in use:" 
            ss -ltnp | grep ":$port" || true
            echo "If this is Ollama, run: sudo snap stop ollama && sudo snap remove ollama"
            echo "To override and force-kill processes on these ports, re-run with: FORCE_KILL_PORTS=1 ./start_servers.sh"
            exit 1
        fi
    fi
done

echo "Starting CYGNUS on port 8081..."
$LLAMA_SERVER -m "$MODEL_PATH" --port 8081 -c 4096 -ngl 99 --host 0.0.0.0 > logs/cygnus.log 2>&1 &
echo $! > logs/cygnus.pid

sleep 2

echo "Starting KATZ on port 8082..."
$LLAMA_SERVER -m "$MODEL_PATH" --port 8082 -c 4096 -ngl 99 --host 0.0.0.0 > logs/katz.log 2>&1 &
echo $! > logs/katz.pid

sleep 2

echo "Starting DOGZ on port 8083..."
$LLAMA_SERVER -m "$MODEL_PATH" --port 8083 -c 4096 -ngl 99 --host 0.0.0.0 > logs/dogz.log 2>&1 &
echo $! > logs/dogz.pid

echo ""
echo "Waiting for servers..."
sleep 5

echo "Checking health..."
curl -s http://localhost:8081/health && echo " CYGNUS OK" || echo " CYGNUS starting..."
curl -s http://localhost:8082/health && echo " KATZ OK" || echo " KATZ starting..."
curl -s http://localhost:8083/health && echo " DOGZ OK" || echo " DOGZ starting..."

echo ""
echo "Servers started! Logs in ./logs/"
