#!/bin/bash

# H-Pipe 파이프라인 오버랩 테스트

NUM_WORKERS=${1:-2}

echo "========================================="
echo "H-Pipe Pipeline Overlap Test"
echo "Workers: $NUM_WORKERS"
echo "========================================="
echo ""

# 시작 포트
BASE_PORT=9999

# 기존 로그 파일 삭제
rm -f root-pipeline.log worker*-pipeline.log

# 워커 PID 배열
declare -a WORKER_PIDS

# 워커 주소 목록 생성
WORKER_ADDRS=""
for ((i=0; i<NUM_WORKERS; i++)); do
    PORT=$((BASE_PORT + i))
    if [ $i -eq 0 ]; then
        WORKER_ADDRS="127.0.0.1:$PORT"
    else
        WORKER_ADDRS="$WORKER_ADDRS 127.0.0.1:$PORT"
    fi
done

# 워커들 시작
echo "Starting $NUM_WORKERS workers..."
for ((i=0; i<NUM_WORKERS; i++)); do
    PORT=$((BASE_PORT + i))
    WORKER_NUM=$((i + 1))
    LOG_FILE="worker${WORKER_NUM}-pipeline.log"

    echo "  Worker $WORKER_NUM on port $PORT..."
    ./hpipe-pipeline-test worker $PORT $i > $LOG_FILE 2>&1 &
    WORKER_PIDS[$i]=$!
    sleep 0.3
done

echo ""
echo "Waiting for workers to initialize..."
sleep 1

# Root 노드 시작
echo "Starting Root node with pipeline test..."
./hpipe-pipeline-test root $WORKER_ADDRS > root-pipeline.log 2>&1
ROOT_EXIT=$?

# 잠시 대기
sleep 0.5

# 워커 프로세스 정리
echo ""
echo "Cleaning up worker processes..."
for pid in "${WORKER_PIDS[@]}"; do
    kill $pid 2>/dev/null
done

echo ""
echo "========================================="
echo "Test Results"
echo "========================================="
echo ""

# Root 로그 출력
echo "ROOT LOG:"
echo "----------------------------------------"
cat root-pipeline.log
echo ""

# 워커 로그 출력
for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_NUM=$((i + 1))
    LOG_FILE="worker${WORKER_NUM}-pipeline.log"

    echo "WORKER $WORKER_NUM LOG:"
    echo "----------------------------------------"
    cat $LOG_FILE
    echo ""
done

# 결과 요약
echo "========================================="
echo "Summary"
echo "========================================="
if [ $ROOT_EXIT -eq 0 ]; then
    echo "✅ Pipeline test PASSED with $NUM_WORKERS workers"
else
    echo "❌ Pipeline test FAILED"
fi

exit $ROOT_EXIT
