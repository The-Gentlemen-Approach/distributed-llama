#!/bin/bash

# H-Pipe End-to-End Test Script

set -e

# Ensure we are running from the project root
cd "$(dirname "$0")/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== H-Pipe End-to-End Test ===${NC}\n"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up...${NC}"
    pkill -f hpipe-worker || true
    sleep 1
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Clean old logs
rm -f worker*.log root.log

# Configuration
MODEL_PATH="models/llama3_2_1b_instruct_q40/dllama_model_llama3_2_1b_instruct_q40.m"
TOKENIZER_PATH="models/llama3_2_1b_instruct_q40/dllama_tokenizer_llama3_2_1b_instruct_q40.t"
PROMPT="Hello my name is Gyo. And I will tell you how to do gentlemen approach."
NWORKERS=${1:-2}
START_PORT=9999
WORKER_PORTS=()
for i in $(seq 0 $((NWORKERS-1))); do
    WORKER_PORTS+=($((START_PORT + i)))
done
NTHREADS=4
STEPS=20
CHUNK_SIZE=4

echo -e "${GREEN}📋 Configuration:${NC}"
echo "  Model: $MODEL_PATH"
echo "  Prompt: \"$PROMPT\""
echo "  Workers: $NWORKERS"
echo "  Chunk size: $CHUNK_SIZE"
echo "  Steps: $STEPS"
echo ""

# Start workers
echo -e "${GREEN}🚀 Starting $NWORKERS workers...${NC}"
for i in $(seq 0 $((NWORKERS-1))); do
    PORT=${WORKER_PORTS[$i]}
    echo "  Starting worker $i on port $PORT"
    ./hpipe-worker --port $PORT --nthreads $NTHREADS > worker${i}.log 2>&1 &
    WORKER_PIDS[$i]=$!
done

# Wait for workers to start
echo -e "${YELLOW}⏳ Waiting for workers to initialize...${NC}"
sleep 3

# Check if workers are running
for i in $(seq 0 $((NWORKERS-1))); do
    if ! kill -0 ${WORKER_PIDS[$i]} 2>/dev/null; then
        echo -e "${RED}❌ Worker $i failed to start!${NC}"
        echo "Worker $i log:"
        cat worker${i}.log
        exit 1
    fi
    echo -e "${GREEN}✓ Worker $i is running (PID: ${WORKER_PIDS[$i]})${NC}"
done

echo ""

# Build worker addresses
WORKER_ADDRS=""
for i in $(seq 0 $((NWORKERS-1))); do
    WORKER_ADDRS="$WORKER_ADDRS 127.0.0.1:${WORKER_PORTS[$i]}"
done

# Run root
echo -e "${GREEN}🎯 Starting root coordinator...${NC}\n"
echo "=========================================="
./hpipe-root \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --prompt "$PROMPT" \
    --workers $WORKER_ADDRS \
    --steps $STEPS \
    --chunk-size $CHUNK_SIZE \
    --nthreads $NTHREADS 2>&1 | tee root.log

ROOT_EXIT_CODE=$?
echo "=========================================="
echo ""

# Show worker logs
echo -e "${BLUE}📊 Worker Logs:${NC}\n"
for i in $(seq 0 $((NWORKERS-1))); do
    echo -e "${YELLOW}--- Worker $i (last 20 lines) ---${NC}"
    tail -20 worker${i}.log
    echo ""
done

# Check exit code
if [ $ROOT_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Test completed successfully!${NC}"
else
    echo -e "${RED}❌ Test failed with exit code $ROOT_EXIT_CODE${NC}"
    exit $ROOT_EXIT_CODE
fi
