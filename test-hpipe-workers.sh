#!/bin/bash

# H-Pipe 네트워크 테스트 - 가변 워커 수 지원
# 사용법: ./test-hpipe-workers.sh [워커 수]
# 예시: ./test-hpipe-workers.sh 3

# 워커 수 파라미터 (기본값: 2)
NUM_WORKERS=${1:-2}

# 워커 수 검증
if ! [[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || [ "$NUM_WORKERS" -lt 1 ]; then
    echo "Error: Invalid number of workers. Must be a positive integer."
    echo "Usage: $0 [number_of_workers]"
    exit 1
fi

echo "========================================="
echo "H-Pipe Network Test with $NUM_WORKERS workers"
echo "========================================="
echo ""

# 시작 포트
BASE_PORT=9999

# 기존 로그 파일 삭제
rm -f root.log worker*.log

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
    LOG_FILE="worker${WORKER_NUM}.log"

    echo "  Worker $WORKER_NUM on port $PORT..."
    ./hpipe-network-test worker $PORT $i > $LOG_FILE 2>&1 &
    WORKER_PIDS[$i]=$!
    sleep 0.3
done

echo ""
echo "Waiting for workers to initialize..."
sleep 1

# Root 노드 시작
echo "Starting Root node..."
./hpipe-network-test root $WORKER_ADDRS > root.log 2>&1
ROOT_EXIT=$?

# 잠시 대기 (워커들이 정리되도록)
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
echo "========================================="
echo "ROOT LOG:"
echo "========================================="
cat root.log
echo ""

# 워커 로그 출력
for ((i=0; i<NUM_WORKERS; i++)); do
    WORKER_NUM=$((i + 1))
    LOG_FILE="worker${WORKER_NUM}.log"

    echo "========================================="
    echo "WORKER $WORKER_NUM LOG:"
    echo "========================================="
    cat $LOG_FILE
    echo ""
done

# 결과 요약
echo "========================================="
echo "Summary"
echo "========================================="
if [ $ROOT_EXIT -eq 0 ]; then
    echo "✅ Test PASSED with $NUM_WORKERS workers"
else
    echo "❌ Test FAILED with exit code $ROOT_EXIT"
fi

exit $ROOT_EXIT
