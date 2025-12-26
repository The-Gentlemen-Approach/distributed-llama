/**
 * nn-cpu.hpp: CPU 디바이스 구현
 *
 * NnDevice 인터페이스의 CPU 구현
 * - 멀티스레드 CPU 연산 실행
 * - ARM NEON, x86 AVX2 최적화 지원
 * - 버퍼 및 가중치 메모리 관리
 *
 * 구조:
 * - NnCpuDevice: CPU 디바이스 (버퍼 관리, 세그먼트 생성)
 * - NnCpuDeviceSegment: CPU 세그먼트 (연산 실행)
 * - nn-cpu-ops.hpp: 실제 연산 커널 구현
 */
#ifndef NN_CPU_H
#define NN_CPU_H

#include <vector>
#include "nn-executor.hpp"
#include "nn-cpu-ops.hpp"

#define DEBUG_USE_MMAP_FOR_WEIGHTS false  // 가중치를 mmap으로 로드 (디버그용)

/**
 * NnCpuDevice: CPU 디바이스 구현
 *
 * 역할:
 * - 버퍼 메모리 할당 및 관리 (가중치, KV 캐시 등)
 * - 파이프와 버퍼 포인터 해석
 * - CPU 세그먼트 생성
 */
class NnCpuDevice : public NnDevice {
public:
    NnByte **buffers;  // 버퍼 배열 (가중치, KV 캐시, 임시 버퍼 등)

private:
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
    NnNetExecution *netExecution;
    NnUint nBuffers;        // 버퍼 개수
    NnByte *bufferFlags;    // 버퍼 플래그 (할당 상태 등)

public:
    NnCpuDevice(NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution);
    ~NnCpuDevice() override;
    NnUint maxNThreads() override;  // CPU 코어 수 반환
    NnDeviceSegment *createSegment(NnUint segmentIndex) override;

    /**
     * resolvePointer: 포인터 설정을 실제 메모리 주소로 해석
     * @param pntrSize: 출력 크기 정보
     * @param pointerConfig: 포인터 설정 (파이프 또는 버퍼 참조)
     * @return 실제 메모리 주소 배열 (배치별)
     */
    std::vector<NnByte *> resolvePointer(NnSize3D *pntrSize, NnPointerConfig *pointerConfig);
};

/**
 * NnCpuDeviceSegment: CPU 세그먼트 구현
 *
 * 하나의 세그먼트 내 모든 연산을 CPU에서 실행
 * 각 연산은 멀티스레드로 병렬 처리
 */
class NnCpuDeviceSegment : public NnDeviceSegment {
public:
    NnUint nOps;                  // 이 세그먼트의 연산 수
    NnCpuOpForward *opForward;    // 연산 함수 포인터 배열
    NnCpuOpContext *opContexts;   // 연산별 컨텍스트 (입출력 포인터, 가중치 등)

    NnCpuDeviceSegment(NnCpuOpForward *opForward, NnCpuOpContext *opContexts, NnUint nOps)
        : opForward(opForward), opContexts(opContexts), nOps(nOps) {}
    ~NnCpuDeviceSegment() override;

    void loadWeight(NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight) override;
    void forward(NnUint opIndex, NnUint nThreads, NnUint threadIndex, NnUint batchSize) override;
};

#endif