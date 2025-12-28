#ifndef COMMON_LLM_TYPES_HPP
#define COMMON_LLM_TYPES_HPP

#include "../nn/nn-core.hpp"

// ==================================================================================
// Enums & Structs
// ==================================================================================

enum LlmHeaderKey {
    LLM_VERSION = 0,
    LLM_ARCH_TYPE = 1,
    LLM_DIM = 2,
    LLM_HIDDEN_DIM = 3,
    LLM_N_LAYERS = 4,
    LLM_N_HEADS = 5,
    LLM_N_KV_HEADS = 6,
    LLM_N_EXPERTS = 7,
    LLM_N_ACTIVE_EXPERTS = 8,
    LLM_VOCAB_SIZE = 9,
    LLM_SEQ_LEN = 10,
    LLM_HIDDEN_ACT = 11,
    LLM_ROPE_THETA = 12,
    LLM_WEIGHT_FLOAT_TYPE = 13,
    LLM_ROPE_SCALING_FACTOR = 14,
    LLM_ROPE_SCALING_LOW_FREQ_FACTOR = 15,
    LLM_ROPE_SCALING_HIGH_FREQ_FACTORY = 16,
    LLM_ROPE_SCALING_ORIG_MAX_SEQ_LEN = 17,
    LLM_ROPE_TYPE = 18,
    LLM_HEAD_DIM = 19,
    LLM_NORM_EPSILON = 20,
    LLM_MOE_HIDDEN_DIM = 21,
};

enum LlmHiddenAct {
    LLM_HIDDEN_ACT_GELU,
    LLM_HIDDEN_ACT_SILU,
};

enum LlmArchType {
    LLM_LLAMA = 0xABCD00,
    LLM_QWEN3 = 0xABCD01,
    LLM_QWEN3_MOE = 0xABCD02,
};

typedef struct {
    NnSize headerSize;
    NnSize fileSize;
    int version;
    LlmArchType archType;

    NnUint dim;
    NnUint nLayers;
    NnUint nHeads;
    NnUint headDim;
    NnUint nKvHeads;
    NnUint nExperts;
    NnUint nActiveExperts;

    NnUint origSeqLen;
    NnUint seqLen;

    NnUint hiddenDim;
    NnUint moeHiddenDim;
    LlmHiddenAct hiddenAct;

    NnUint qDim;
    NnUint kvDim;

    NnUint vocabSize;

    float ropeTheta;
    NnRopeType ropeType;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactory;
    NnUint ropeScalingOrigMaxSeqLen;

    float normEpsilon;

    NnFloatType weightType;
    NnFloatType syncType;
} LlmHeader;

typedef struct {
    LlmHeader *header;
    NnNetConfig netConfig;
    NnNodeConfig nodeConfig;  // Single-node: no array needed

    NnRowMatmulSlice qSlice;
    NnRowMatmulSlice kSlice;
    NnRowMatmulSlice vSlice;
    NnColMatmulSlice woSlice;

    NnRowMatmulSlice w1Slice;
    NnColMatmulSlice w2Slice;
    NnRowMatmulSlice w3Slice;

    NnRowMatmulSlice wclsSlice;

    NnUint positionPipeIndex;
    NnUint tokenPipeIndex;
    NnUint xPipeIndex;
    NnUint logitsPipeIndex;

    NnSize3D tokenEmbeddingSize;
    NnSize3D rmsNormSize;
    NnSize3D qkRmsNormSize;
    NnSize3D moeGateSize;
} LlmNet;

// ==================================================================================
// Functions
// ==================================================================================

/**
 * Loads the model header.
 */
LlmHeader loadLlmHeader(const char* path, const unsigned int maxSeqLen, NnFloatType syncType);

/**
 * Prints the model header.
 */
void printLlmHeader(LlmHeader *header);

#endif
