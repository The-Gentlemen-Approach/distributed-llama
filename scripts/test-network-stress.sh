#!/bin/bash

# Ensure we are running from the project root
cd "$(dirname "$0")/.."

# H-Pipe 네트워크 종합 테스트 (1~10 워커)

echo "========================================="
echo "H-Pipe Network Comprehensive Test"
echo "Testing with 1 to 10 workers"
echo "========================================="
echo ""

PASSED=0
FAILED=0

for NUM_WORKERS in {1..10}; do
    echo "----------------------------------------"
    echo "Testing with $NUM_WORKERS worker(s)..."
    echo "----------------------------------------"

    ./scripts/test-network-basic.sh $NUM_WORKERS > /dev/null 2>&1

    if [ $? -eq 0 ]; then
        echo "✅ PASSED with $NUM_WORKERS worker(s)"
        ((PASSED++))
    else
        echo "❌ FAILED with $NUM_WORKERS worker(s)"
        ((FAILED++))

        # 실패한 경우 로그 출력
        echo ""
        echo "  Root log (last 10 lines):"
        tail -10 root.log | sed 's/^/    /'
        echo ""
    fi

    # 프로세스 정리
    pkill -f hpipe-network-test 2>/dev/null
    sleep 0.5
done

echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total tests: $((PASSED + FAILED))"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "🎉 All tests PASSED!"
    exit 0
else
    echo "⚠️  Some tests failed"
    exit 1
fi
