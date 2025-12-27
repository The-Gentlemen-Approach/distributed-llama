#ifndef HPIPE_CORE_HPP
#define HPIPE_CORE_HPP

#include <vector>
#include <string>
#include <cstdint>
#include "simple-llm.hpp" 

// ==================================================================================
// 1. 데이터 구조체 (Data Structures)
// ==================================================================================

// 워커에게 할당된 레이어 범위 [start, end] (포함)
struct LayerRange {
    int start;
    int end;
};

// 단일 파이프라인 청크를 위한 작업 단위
struct ChunkTask {
    int seq_id;             // 청크 고유 ID
    int start_pos;          // 전체 시퀀스 내 시작 위치
    std::vector<int> tokens; // 이 청크에 포함된 토큰들
};

// ==================================================================================
// 2. 네트워크 프로토콜 (Network Protocols)
// ==================================================================================

// 초기 설정 패킷
struct HPipeConfig {
    int worker_id;      // 워커 ID
    int total_workers;  // 전체 워커 수
    
    // 레이어 할당 정보
    LayerRange layer_range;
    
    // 순방향 (데이터) 경로 정보
    char next_host[256];
    int next_port;
    
    // 역방향 (제어/ACK) 경로 정보
    char prev_host[256];
    int prev_port;
    
    // 모델 정보
    char model_path[512]; // 모델 파일 경로
    SimpleLlmHeader model_header; // 모델 메타데이터
};

// 런타임 데이터 패킷 헤더
struct HPipeHeader {
    uint32_t magic;       // 검증용 매직 넘버 (0x48504950)
    uint32_t seq_id;      // 청크 시퀀스 ID
    uint32_t step;        // 0: Prefill, 1+: Decoding
    uint32_t n_tokens;    // 토큰 수
    uint32_t data_size;   // 텐서 데이터 크기 (바이트)
};

// 런타임 제어(ACK) 패킷
struct HPipeControl {
    uint32_t magic;       // 검증용 매직 넘버 (0x41434B33)
    uint32_t seq_id;      // 처리 완료된 시퀀스 ID
};

// ==================================================================================
// 3. 추상화 인터페이스 (Abstraction Interfaces)
// ==================================================================================

// 레이어 분할 정책 인터페이스
class IPartitioningPolicy {
public:
    virtual ~IPartitioningPolicy() {}
    
    // 모델 헤더와 워커 수를 기반으로 각 워커의 레이어 범위를 결정합니다.
    virtual std::vector<LayerRange> assignLayers(
        const SimpleLlmHeader& header, 
        int n_workers
    ) = 0;
};

// 파이프라인 스케줄러 인터페이스
class IPipelineScheduler {
public:
    virtual ~IPipelineScheduler() {}
    
    // 프롬프트 토큰들을 작은 청크(Task)들로 분할합니다.
    virtual std::vector<ChunkTask> schedule(
        const std::vector<int>& prompt_tokens,
        int max_seq_len
    ) = 0;
};

#endif // HPIPE_CORE_HPP