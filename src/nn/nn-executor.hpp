/**
 * nn-executor.hpp: 신경망 실행 엔진
 *
 * 주요 구성요소:
 * - NnDevice: 디바이스 추상화 (CPU, GPU)
 * - NnDeviceSegment: 세그먼트별 연산 실행 단위
 * - NnExecutor: 멀티스레드 실행 조율자
 * - NnNodeSynchronizer: 노드 간 동기화 인터페이스
 *
 * 실행 모델:
 * 1. 네트워크를 세그먼트로 분할 (레이어 그룹)
 * 2. 각 세그먼트는 디바이스(CPU/GPU)에서 실행
 * 3. 멀티스레드로 병렬 처리
 * 4. 세그먼트 간 노드 동기화
 */
#ifndef NN_EXECUTOR_H
#define NN_EXECUTOR_H

#include "nn-core.hpp"
#include <atomic>
#include <vector>
#include <stdexcept>
#include "pthread.h"

/**
 * NnDeviceSegment: 디바이스별 세그먼트 실행 단위
 *
 * 세그먼트는 여러 연산(Op)의 모음으로, 하나의 디바이스에서 실행됨
 * 각 디바이스(CPU, GPU)는 이 인터페이스를 구현
 */
class NnDeviceSegment {
public:
    virtual ~NnDeviceSegment() {};

    /**
     * loadWeight: 연산에 사용될 가중치 로드
     * @param opIndex: 연산 인덱스
     * @param offset: 가중치 내 오프셋
     * @param nBytes: 로드할 바이트 수
     * @param weight: 가중치 데이터
     */
    virtual void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) = 0;

    /**
     * forward: 특정 연산 실행
     * @param opIndex: 실행할 연산 인덱스
     * @param nThreads: 총 스레드 수
     * @param threadIndex: 현재 스레드 인덱스
     * @param batchSize: 배치 크기
     */
    virtual void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) = 0;
};

/**
 * NnDevice: 디바이스 추상화 인터페이스
 *
 * CPU, GPU 등 다양한 하드웨어를 추상화
 * 각 디바이스는 여러 세그먼트를 생성할 수 있음
 */
class NnDevice {
public:
    virtual NnUint maxNThreads() = 0;  // 이 디바이스가 지원하는 최대 스레드 수
    virtual ~NnDevice() {}
    virtual NnDeviceSegment *createSegment(NnUint segmentIndex) = 0;
};

/**
 * NnNodeSynchronizer: 노드 간 동기화 인터페이스
 *
 * 분산 추론에서 노드들 간 데이터 교환을 담당
 */
class NnNodeSynchronizer {
public:
    virtual ~NnNodeSynchronizer() {};

    /**
     * sync: 세그먼트 실행 후 노드 간 동기화
     * @param segmentIndex: 세그먼트 인덱스
     * @param nThreads: 총 스레드 수
     * @param threadIndex: 현재 스레드 인덱스
     */
    virtual void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) = 0;
};

/**
 * NnFakeNodeSynchronizer: 단일 노드용 더미 동기화기
 * 동기화가 필요 없는 경우 사용 (아무 작업도 하지 않음)
 */
class NnFakeNodeSynchronizer : public NnNodeSynchronizer {
public:
    ~NnFakeNodeSynchronizer() override {};
    void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) override;
};

/**
 * NnNetExecution: 네트워크 실행 컨텍스트
 *
 * 파이프 메모리를 관리하고 배치 크기를 추적
 * 모든 연산은 이 파이프들을 통해 데이터를 주고받음
 */
class NnNetExecution {
public:
    NnUint nThreads;     // 사용할 스레드 수
    NnUint nPipes;       // 파이프 개수
    NnByte **pipes;      // 파이프 배열 (연산 간 데이터 전달)
    NnUint batchSize;    // 현재 배치 크기
    NnUint nBatches;     // 최대 배치 크기

    NnNetExecution(NnUint nThreads, NnNetConfig *netConfig);
    ~NnNetExecution();
    void setBatchSize(NnUint batchSize);
};

/**
 * NnExecutorStepType: 실행 단계 타입
 */
enum NnExecutorStepType {
    STEP_EXECUTE_OP,     // 연산 실행
    STEP_SYNC_NODES,     // 노드 간 동기화
};

#define N_STEP_TYPES STEP_SYNC_NODES + 1

/**
 * NnExecutorDevice: 디바이스 및 담당 세그먼트 범위
 *
 * 하나의 디바이스가 처리할 세그먼트 범위를 지정
 * 예: GPU가 레이어 0~10, CPU가 레이어 11~32 처리
 */
class NnExecutorDevice {
public:
    std::unique_ptr<NnDevice> device;  // 디바이스 (CPU, GPU 등)
    int segmentFrom;                    // 시작 세그먼트 (-1이면 모든 세그먼트)
    int segmentTo;                      // 종료 세그먼트 (-1이면 모든 세그먼트)

    NnExecutorDevice(NnDevice *device, int segmentFrom, int segmentTo);
};

/**
 * NnExecutorStep: 실행 단계
 *
 * Executor는 스텝들의 시퀀스를 순차적으로 실행
 * 각 스텝은 연산 실행 또는 동기화를 나타냄
 */
typedef struct {
    NnExecutorStepType type;     // 스텝 타입 (연산 or 동기화)
    NnDeviceSegment *segment;    // 실행할 세그먼트 (연산인 경우)
    NnUint arg0;                 // 추가 인자 (opIndex 또는 segmentIndex)
    NnOpConfig *opConfig;        // 연산 설정 (연산인 경우)
} NnExecutorStep;

/**
 * NnExecutorContext: 실행 컨텍스트 (스레드 간 공유)
 *
 * 모든 워커 스레드가 공유하는 실행 상태
 * 원자적 연산으로 스레드 안전성 보장
 */
typedef struct {
    NnUint nThreads;                      // 워커 스레드 수
    NnUint nSteps;                        // 총 스텝 수
    NnExecutorStep *steps;                // 스텝 배열
    NnNodeSynchronizer *synchronizer;     // 노드 동기화기

    // 스레드 동기화용 원자적 변수
    std::atomic_uint currentStepIndex;    // 현재 실행 중인 스텝 인덱스
    std::atomic_uint doneThreadCount;     // 완료된 스레드 수
    std::atomic_bool isAlive;             // 스레드 풀 활성 상태

    NnUint batchSize;                     // 현재 배치 크기
    Timer *timer;                         // 성능 측정용 타이머
    NnUint totalTime[N_STEP_TYPES];       // 스텝 타입별 누적 시간
} NnExecutorContext;

/**
 * NnExecutorThread: 워커 스레드 정보
 */
typedef struct {
    NnUint threadIndex;           // 스레드 인덱스
    NnExecutorContext *context;   // 공유 컨텍스트
    PthreadHandler handler;       // pthread 핸들러
} NnExecutorThread;

/**
 * NnExecutorException: 실행 중 발생하는 예외
 */
class NnExecutorException : public std::runtime_error {
public:
    NnExecutorException(const std::string message);
};

/**
 * NnExecutor: 신경망 실행 조율자
 *
 * 역할:
 * 1. 세그먼트를 디바이스에 할당
 * 2. 가중치 로딩 조율
 * 3. 멀티스레드 실행 관리
 * 4. 노드 간 동기화 조율
 *
 * 실행 흐름:
 * - forward() 호출 시 모든 스텝을 순차 실행
 * - 각 스텝은 여러 스레드가 병렬 처리
 * - 스텝 간에는 배리어로 동기화
 */
class NnExecutor {
private:
    NnNetExecution *netExecution;
    NnNodeConfig *nodeConfig;
    std::vector<std::unique_ptr<NnDeviceSegment>> segments;
    std::vector<NnExecutorStep> steps;
    NnExecutorThread *threads;
    NnExecutorContext context;
public:
    NnExecutor(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, std::vector<NnExecutorDevice> *device, NnNetExecution *netExecution, NnNodeSynchronizer *synchronizer, bool benchmark);
    ~NnExecutor();
    void loadWeight(const char *name, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight);
    void forward();
    NnUint getTotalTime(NnExecutorStepType type);
};

#endif