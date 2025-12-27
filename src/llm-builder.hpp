#ifndef LLM_BUILDER_HPP
#define LLM_BUILDER_HPP

#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "simple-llm.hpp"
/**
 * Segment builder functions for LLM network construction.
 * These functions build individual segments that can be used by both
 * simple-llm (single-node) and hpipe (distributed pipeline).
 */

// ==================================================================================
// Buffer Indices Structure
// ==================================================================================

/**
 * All buffer indices allocated for the network.
 * Corresponds to simple-llm.cpp lines 250-288.
 */
typedef struct {
    // Shared buffers (reused across all layers)
    NnUint xBufferIndex;
    NnUint yBufferIndex;
    NnUint yqBufferIndex;
    NnUint zBufferIndex;
    NnUint zqSliceBufferIndex;
    NnUint qBufferIndex;
    NnUint kTempBufferIndex;
    NnUint vTempBufferIndex;
    NnUint invRmsBufferIndex;
    NnUint ropeCacheBufferIndex;
    NnUint attBufferIndex;
    NnUint logitsSliceBufferIndex;
    NnUint dBufferIndex;
    NnUint dqBufferIndex;
    NnUint lBufferIndex;

    // MoE buffers
    NnUint moeGtBufferIndex;
    NnUint moeExpertIndexesBufferIndex;
    NnUint moeYBufferIndex;
    NnUint moeYqBufferIndex;
    NnUint moeDBufferIndex;
    NnUint moeDQBufferIndex;
    NnUint moeLBufferIndex;
    NnUint moeSBufferIndex;
} LlmBufferIndices;

// ==================================================================================
// Network Initialization Structure
// ==================================================================================

/**
 * Configuration needed for building LLM segments.
 * Returned by initializeLlmNetwork().
 */
typedef struct {
    NnKvCacheSlice kvCacheSlice;
    NnMultiHeadAttSlice multiHeadAttSlice;
    NnRopeSlice ropeSlice;
    NnUint zqPipeIndex;
    NnUint nQNormColumns;
    NnUint nKNormColumns;
    NnUint nInvBufferColumns;
} LlmNetworkConfig;

// ==================================================================================
// Network Initialization Function
// ==================================================================================

/**
 * Initialize SimpleLlmNet with slices, pipes, and configuration.
 * Corresponds to simple-llm.cpp lines 167-246.
 *
 * @param net SimpleLlmNet structure to initialize
 * @param header Model header
 * @param nBatches Number of batches
 * @return LlmNetworkConfig with slices and configuration
 */
LlmNetworkConfig initializeLlmNetwork(SimpleLlmNet *net, SimpleLlmHeader *header, NnUint nBatches);

// ==================================================================================
// Buffer Allocation Function
// ==================================================================================

/**
 * Allocate all buffers for the network.
 * Corresponds to simple-llm.cpp lines 250-288.
 */
LlmBufferIndices allocateLlmBuffers(
    NnNodeConfigBuilder *nodeBuilder,
    SimpleLlmHeader *header,
    SimpleLlmNet *net,
    NnUint nBatches,
    NnUint nInvBufferColumns,
    const NnRopeSlice &ropeSlice,
    const NnMultiHeadAttSlice &multiHeadAttSlice
);

// ==================================================================================
// Segment Builder Functions
// ==================================================================================

/**
 * Build embedding segment.
 */
void buildEmbeddingSegment(
    NnNodeConfigBuilder *nodeBuilder,
    NnUint tokenPipeIndex,
    NnUint xPipeIndex,
    NnSize3D tokenEmbeddingSize
);

/**
 * Build attention segment for a transformer layer.
 */
void buildAttentionSegment(
    NnNodeConfigBuilder *nodeBuilder,
    const LlmBufferIndices *buffers,
    SimpleLlmNet *net,
    NnUint layerIndex,
    NnUint kBufferIndex,
    NnUint vBufferIndex,
    NnUint zqPipeIndex,
    const NnRopeSlice &ropeSlice,
    const NnKvCacheSlice &kvCacheSlice,
    const NnMultiHeadAttSlice &multiHeadAttSlice,
    NnUint nQNormColumns,
    NnUint nKNormColumns
);

/**
 * Build FFN segment (non-MoE).
 */
void buildFFNSegment(
    NnNodeConfigBuilder *nodeBuilder,
    const LlmBufferIndices *buffers,
    SimpleLlmNet *net,
    NnUint layerIndex,
    NnUint zqPipeIndex
);

/**
 * Build MoE FFN segment.
 */
void buildMoEFFNSegment(
    NnNodeConfigBuilder *nodeBuilder,
    const LlmBufferIndices *buffers,
    SimpleLlmNet *net,
    NnUint layerIndex,
    NnUint zqPipeIndex
);

/**
 * Build classifier segment (final layer).
 */
void buildClassifierSegment(
    NnNodeConfigBuilder *nodeBuilder,
    const LlmBufferIndices *buffers,
    SimpleLlmNet *net,
    NnUint zqPipeIndex
);

#endif // LLM_BUILDER_HPP
