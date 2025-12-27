#ifndef SIMPLE_LLM_HPP
#define SIMPLE_LLM_HPP

#include "nn/nn-core.hpp"
#include "nn/nn-executor.hpp"
#include "simple-llm-utils.hpp"

// ==================================================================================
// Enums & Structs (Copied & Renamed from llm.hpp to avoid dependency)
// ==================================================================================

enum SimpleLlmHeaderKey {
    SIMPLE_VERSION = 0,
    SIMPLE_ARCH_TYPE = 1,
    SIMPLE_DIM = 2,
    SIMPLE_HIDDEN_DIM = 3,
    SIMPLE_N_LAYERS = 4,
    SIMPLE_N_HEADS = 5,
    SIMPLE_N_KV_HEADS = 6,
    SIMPLE_N_EXPERTS = 7,
    SIMPLE_N_ACTIVE_EXPERTS = 8,
    SIMPLE_VOCAB_SIZE = 9,
    SIMPLE_SEQ_LEN = 10,
    SIMPLE_HIDDEN_ACT = 11,
    SIMPLE_ROPE_THETA = 12,
    SIMPLE_WEIGHT_FLOAT_TYPE = 13,
    SIMPLE_ROPE_SCALING_FACTOR = 14,
    SIMPLE_ROPE_SCALING_LOW_FREQ_FACTOR = 15,
    SIMPLE_ROPE_SCALING_HIGH_FREQ_FACTORY = 16,
    SIMPLE_ROPE_SCALING_ORIG_MAX_SEQ_LEN = 17,
    SIMPLE_ROPE_TYPE = 18,
    SIMPLE_HEAD_DIM = 19,
    SIMPLE_NORM_EPSILON = 20,
    SIMPLE_MOE_HIDDEN_DIM = 21,
};

enum SimpleLlmHiddenAct {
    SIMPLE_HIDDEN_ACT_GELU,
    SIMPLE_HIDDEN_ACT_SILU,
};

enum SimpleLlmArchType {
    SIMPLE_LLAMA = 0xABCD00,
    SIMPLE_QWEN3 = 0xABCD01,
    SIMPLE_QWEN3_MOE = 0xABCD02,
};

typedef struct {
    NnSize headerSize;
    NnSize fileSize;
    int version;
    SimpleLlmArchType archType;

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
    SimpleLlmHiddenAct hiddenAct;

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
} SimpleLlmHeader;

typedef struct {
    SimpleLlmHeader *header;
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
} SimpleLlmNet;

// ==================================================================================
// Classes & Functions
// ==================================================================================

/**
 * Loads the model header.
 */
SimpleLlmHeader loadSimpleLlmHeader(const char* path, const unsigned int maxSeqLen, NnFloatType syncType);

/**
 * Prints the model header.
 */
void printSimpleLlmHeader(SimpleLlmHeader *header);

/**
 * Builds the network for single-node execution.
 */
SimpleLlmNet buildSimpleLlmNet(SimpleLlmHeader *h, NnUint nBatches);

/**
 * Releases the network resources.
 */
void releaseSimpleLlmNet(SimpleLlmNet *net);

/**
 * Loads weights into the network.
 */
void loadSimpleLlmNetWeight(const char* path, SimpleLlmNet *net, NnExecutor *executor);

/**
 * Simple Inference Control (Simplified RootLlmInference).
 * Assumes local execution or simplified network control.
 */
class SimpleLlmInference {
public:
    float *logitsPipe;

private:
    float *tokenPipe;
    float *positionPipe;
    SimpleLlmHeader *header;
    NnNetExecution *execution;
    NnExecutor *executor;
    
    // Simplified control packet structure locally if needed, 
    // but for single node we just set pipes.
    // If we want to support distributed in simple-llm, we need networking.
    // Assuming simple-llm is standalone and might not need full distributed complexity,
    // but the user wants "simple-dllama" which implies distributed llama simplified.
    // However, simple-dllama.cpp sets nNodes=1.
    
public:
    SimpleLlmInference(SimpleLlmNet *net, NnNetExecution *execution, NnExecutor *executor);
    void setBatchSize(NnUint batchSize);
    void setPosition(NnUint position);
    void setToken(NnUint batchIndex, NnUint token);
    void forward();
};

#endif
