/**
 * nn-core.hpp: 신경망 연산의 핵심 자료구조 및 설정
 *
 * 주요 구성요소:
 * - NnSize3D: 3차원 텐서 크기 정의
 * - Slice 구조체들: 텐서 병렬화를 위한 슬라이싱 정보
 * - NnOpCode: 지원하는 신경망 연산 타입
 * - Config 구조체들: 네트워크, 노드, 연산, 파이프, 버퍼 설정
 */
#ifndef NN_CORE_H
#define NN_CORE_H

#include <chrono>
#include <list>
#include <memory>
#include <cstdint>
#include "nn-quants.hpp"

// ========== Primitives ==========

/**
 * NnSize3D: 3차원 텐서 크기를 나타내는 구조체
 *
 * 텐서의 shape은 (z, y, x)로 표현되며, z는 batch 차원인 경우가 많음
 * 메모리 레이아웃은 row-major (C order)
 */
typedef struct {
    NnFloatType floatType;  // 데이터 타입 (f32, q40, q80 등)
    NnUint z;               // 첫 번째 차원 (보통 batch 또는 레이어 수)
    NnUint y;               // 두 번째 차원 (행)
    NnUint x;               // 세 번째 차원 (열)
    NnSize length;          // 총 요소 수 (z * y * x)
    NnSize nBytes;          // 총 바이트 수
    NnSize nBytesXY;        // 한 z-slice의 바이트 수 (y * x의 바이트)
} NnSize3D;

// ========== Slices (텐서 병렬화) ==========
// 분산 추론에서 각 노드가 처리할 텐서의 일부분을 정의

/**
 * NnKvCacheSlice: KV 캐시 슬라이싱 정보
 * KV 캐시는 어텐션 연산에서 Key와 Value를 저장하는 메모리
 */
typedef struct {
    NnUint kvDim0;          // 전체 KV 차원
    NnSize3D keySize;       // Key 캐시 크기
    NnSize3D valueSize;     // Value 캐시 크기
} NnKvCacheSlice;

/**
 * NnRowMatmulSlice: 행렬곱의 행 방향 슬라이싱
 *
 * 행렬 W (d x n)에서 행을 나누어 분산
 * 각 노드는 W의 일부 행 (d0 x n)을 처리
 */
typedef struct {
    NnFloatType type;       // 가중치 데이터 타입
    NnUint nNodes;          // 노드 수
    NnUint d0;              // 각 노드가 처리할 행 수 (d / nNodes)
    NnUint n;               // 열 수
    NnSize3D size;          // 전체 행렬 크기
    NnSize3D sliceSize;     // 각 노드의 슬라이스 크기
} NnRowMatmulSlice;

/**
 * NnColMatmulSlice: 행렬곱의 열 방향 슬라이싱
 *
 * 행렬 W (n x d)에서 열을 나누어 분산
 * 각 노드는 W의 일부 열 (n x d0)을 처리
 */
typedef struct {
    NnFloatType type;       // 가중치 데이터 타입
    NnUint nNodes;          // 노드 수
    NnUint n;               // 행 수
    NnUint n0;              // 각 노드의 시작 행 인덱스
    NnUint d;               // 전체 열 수
    NnSize3D size;          // 전체 행렬 크기
    NnSize3D sliceSize;     // 각 노드의 슬라이스 크기
} NnColMatmulSlice;

typedef struct {
    NnUint qDim0;
    NnUint qDimStart;
    NnUint qDimEnd;
    NnUint qShift;
    NnUint kvDim;
    NnUint kvDim0;
    NnUint kvDimStart;
    NnUint sliceDim;
    NnUint seqLen;
    NnUint headDim;
    NnUint nKvHeads;
    float ropeTheta;
    NnSize3D cacheSize;
} NnRopeSlice;

typedef struct {
    NnUint nHeads;
    NnUint nHeads0;
    NnSize3D attSize;
} NnMultiHeadAttSlice;

// ========== Base Enums ==========

/**
 * NnOpCode: 지원하는 신경망 연산 타입
 *
 * 모든 신경망 연산은 이 opcode 중 하나로 표현됨
 * 각 디바이스(CPU, GPU)는 이 연산들을 구현해야 함
 */
enum NnOpCode {
    OP_MERGE_ADD,        // 여러 텐서를 더함
    OP_MERGE_SUM,        // 여러 텐서를 합산
    OP_EMBEDDING,        // 토큰 임베딩 lookup
    OP_INV_RMS,          // RMS의 역수 계산 (정규화 준비)
    OP_RMS_NORM,         // RMS 정규화 적용
    OP_MATMUL,           // 행렬 곱셈
    OP_ROPE,             // Rotary Position Embedding
    OP_MULTIHEAD_ATT,    // Multi-Head Attention
    OP_GELU,             // GELU 활성화 함수
    OP_SILU,             // SILU(Swish) 활성화 함수
    OP_MUL,              // 요소별 곱셈
    OP_SCALE,            // 스칼라 곱
    OP_CAST,             // 타입 변환 (f32 <-> q80 등)
    OP_REPEAT_Z,         // Z 차원으로 복사
    OP_SHIFT,            // 시퀀스 shift (KV 캐시 업데이트용)
    OP_SOFTMAX,          // Softmax 활성화
    OP_MOE_GATE,         // MoE 게이팅 (어떤 전문가 사용할지 선택)
};

/**
 * NnOpQuantType: 연산의 양자화 조합 타입
 *
 * 형식: <입력>_<가중치>_<출력>
 * 예: F32_Q40_F32 = f32 입력, q40 가중치, f32 출력
 */
enum NnOpQuantType {
    F32_F32_F32,    // 모두 f32
    F32_Q40_F32,    // q40 가중치 사용 (일반적)
    F32_Q40_Q80,    // q40 가중치, q80 출력 (동기화 전)
    F32_F32_Q80,    // f32 가중치, q80 출력
    Q80_Q80_Q80,    // 모두 q80
    Q80_Q80_F32,    // q80 입력, f32 출력
    Q80_Q40_F32,    // q80 입력, q40 가중치, f32 출력
    Q80_F32_F32,    // q80 입력, f32 가중치
};

#define N_OP_CODES (OP_SHIFT + 1)
#define N_OP_QUANTS (Q80_F32_F32 + 1)

enum NnPointerSource {
    SRC_PIPE,
    SRC_BUFFER,
};

enum NnPointerType {
    PNTR_RAW,
    PNTR_BATCH,
    PNTR_BATCHED_SLICE
};

/**
 * NnSyncType: 노드 간 동기화 방식
 */
enum NnSyncType {
    SYNC_WITH_ROOT,                    // 루트가 전체 파이프를 모든 노드에 브로드캐스트
    SYNC_NODE_SLICES,                  // 각 노드가 자신의 슬라이스를 모든 노드에 전송
    SYNC_NODE_SLICES_EXCEPT_ROOT,      // 워커들만 루트에게 슬라이스 전송 (all-gather)
};

/**
 * NnRopeType: RoPE (Rotary Position Embedding) 변형 타입
 */
enum NnRopeType {
    ROPE_LLAMA = 0,      // 표준 LLaMA RoPE
    ROPE_FALCON = 1,     // Falcon 스타일 RoPE
    ROPE_LLAMA3_1 = 2,   // Llama 3.1 RoPE (스케일링 지원)
};

// ========== Base Configs ==========

/**
 * NnPipeConfig: 파이프 설정
 * 파이프는 연산 간 데이터를 전달하는 버퍼 (레지스터처럼 작동)
 */
typedef struct {
    char *name;         // 디버깅용 이름
    NnSize3D size;      // 파이프 크기
} NnPipeConfig;

/**
 * NnBufferConfig: 버퍼 설정
 * 버퍼는 가중치나 중간 결과를 저장하는 메모리
 */
typedef struct {
    char *name;         // 디버깅용 이름
    NnSize3D size;      // 버퍼 크기
} NnBufferConfig;

typedef struct {
    NnPointerSource source;
    NnUint pointerIndex;
    NnPointerType type;
} NnPointerConfig;

typedef struct {
    NnOpCode code;
    char *name;
    NnUint index;
    NnPointerConfig input;
    NnPointerConfig output;
    NnSize3D weightSize;
    NnByte *config;
    NnUint configSize;
} NnOpConfig;

typedef struct {
    NnUint pipeIndex;
} NnPreSyncConfig;

typedef struct {
    NnUint pipeIndex;
    NnSyncType syncType;
} NnSyncConfig;

typedef struct  {
    NnUint nOps;
    NnOpConfig *ops;
    NnUint nSyncs;
    NnSyncConfig *syncs;
} NnSegmentConfig;

typedef struct {
    NnUint nBatches;
    NnUint nNodes;
    NnUint nPipes;
    NnPipeConfig *pipes;
    NnUint nPreSyncs;
    NnPreSyncConfig *preSyncs;
} NnNetConfig;

typedef struct {
    NnUint nodeIndex;
    NnUint nBuffers;
    NnBufferConfig *buffers;
    NnUint nSegments;
    NnSegmentConfig *segments;
} NnNodeConfig;

// op configs

typedef struct {
    // empty
} NnEmbeddingOpConfig;

typedef struct {
    float epsilon;
    NnUint nColumns;
} NnInvRmsOpConfig;

typedef struct {
    NnUint invRmsBufferIndex;
    NnUint nColumns;
} NnRmsNormOpConfig;

typedef struct {
    NnUint nExperts;
    NnUint nActiveExperts;
    NnUint activeExpertIndexesBufferIndex;
} NnMatmulOpConfig;

typedef struct {
    NnRopeType type;
    NnUint isQ; // Cannot use `bool` here due to GPU memory alignment
    NnUint positionPipeIndex;
    NnUint ropeCacheBufferIndex;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactor;
    NnUint ropeScalingOrigMaxSeqLen;
    NnRopeSlice slice;
} NnRopeOpConfig;

typedef struct {
    NnUint nHeads;
    NnUint nHeads0;
    NnUint nKvHeads;
    NnUint headDim;
    NnUint seqLen;
    NnUint qSliceD0;
    NnUint kvDim0;
    NnUint positionPipeIndex;
    NnUint queryBufferIndex;
    NnUint keyCacheBufferIndex;
    NnUint valueCacheBufferIndex;
    NnUint attBufferIndex;
} NnMultiHeadAttOpConfig;

typedef struct {
    // empty
} NnMergeAddOpCodeConfig;

typedef struct {
    // empty
} NnMergeSumOpCodeConfig;

typedef struct {
    // empty
} NnSiluOpCodeConfig;

typedef struct {
    NnUint multiplierBufferIndex;
} NnMulOpCodeConfig;

typedef struct {
    NnUint scaleBufferIndex;
} NnScaleOpCodeConfig;

typedef struct {
    // empty
} NnCastOpCodeConfig;

typedef struct {
    // empty
} NnRepeatZOpCodeConfig;

typedef struct {
    NnUint indexPipeIndex;
} NnShiftOpCodeConfig;

typedef struct {
    // empty
} NnSoftmaxOpCodeConfig;

typedef struct {
    NnUint k;
    NnUint normTopk;
    NnUint indexesBufferIndex;
} NnMoeGateOpCodeConfig;

// utility functions

const char *opCodeToString(NnOpCode code);
const char *opQuantTypeToString(NnOpQuantType type);

NnSize getBytes(NnFloatType floatType, NnSize n);
NnSize getBlockSize(NnFloatType floatType);
NnOpQuantType getOpQuantType(NnFloatType input, NnFloatType weight, NnFloatType output);
NnSize3D size0();
NnSize3D size1D(NnFloatType floatType, NnUint x);
NnSize3D size2D(NnFloatType floatType, NnUint y, NnUint x);
NnSize3D size3D(NnFloatType floatType, NnUint z, NnUint y, NnUint x);
NnPointerConfig pointerBatchConfig(NnPointerSource source, NnUint index);
NnPointerConfig pointerBatchedSliceConfig(NnPointerSource source, NnUint index);
NnPointerConfig pointerRawConfig(NnPointerSource source, NnUint index);
bool hasPointerContinuousMemory(NnPointerConfig *config);

void releaseNetConfig(NnNetConfig *netConfig);
void releaseNodeConfig(NnNodeConfig *nodeConfig);

void printNodeRequiredMemory(NnNetConfig *netConfig, NnNodeConfig *nodeConfig);

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
public:
    Timer();
    void reset();
    NnUint elapsedMiliseconds();
    NnUint elapsedMicroseconds();
};

// slicers

NnKvCacheSlice sliceKvCache(NnUint kvDim, NnUint seqLen, NnUint nNodes);
NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);
NnColMatmulSlice sliceColMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);
NnRopeSlice sliceRope(NnRopeType type, NnUint qDim, NnUint kvDim, NnUint nKvHeads, NnUint nNodes, NnUint seqLen, NnUint headDim, float ropeTheta, NnUint nodeIndex);
NnMultiHeadAttSlice sliceMultiHeadAtt(NnUint nHeads, NnUint seqLen, NnUint nNodes, NnUint nBatches);

// splitters

NnUint splitRowMatmulWeight(NnRowMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0);
NnUint splitColMatmulWeight(NnColMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0);

// rope

void fullfillRopeCache(const NnRopeOpConfig *config, float *cache);

#endif
