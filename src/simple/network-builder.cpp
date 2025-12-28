#include "simple/network-builder.hpp"
#include "common/llm-builder.hpp"
#include "nn/nn-config-builder.hpp"

// ==================================================================================
// Build Network
// ==================================================================================

SimpleLlmNet buildSimpleLlmNet(SimpleLlmHeader *h, NnUint nBatches) {
    SimpleLlmNet n;

    // ==================================================================================
    // Initialize network (slices, pipes, config)
    // ==================================================================================
    LlmNetworkConfig config = initializeLlmNetwork(&n, h, nBatches);

    // ==================================================================================
    // Build node configuration
    // ==================================================================================
    NnNodeConfigBuilder nodeBuilder(0);

    // Allocate buffers
    LlmBufferIndices buffers = allocateLlmBuffers(
        &nodeBuilder, h, &n, nBatches,
        config.nInvBufferColumns, config.ropeSlice, config.multiHeadAttSlice
    );

    // Build embedding segment
    buildEmbeddingSegment(&nodeBuilder, n.tokenPipeIndex, n.xPipeIndex, n.tokenEmbeddingSize);

    // Build transformer layers
    for (NnUint layerIndex = 0; layerIndex < h->nLayers; layerIndex++) {
        const NnUint kBufferIndex = nodeBuilder.addBuffer("k", config.kvCacheSlice.keySize);
        const NnUint vBufferIndex = nodeBuilder.addBuffer("v", config.kvCacheSlice.valueSize);

        // Attention segment
        buildAttentionSegment(
            &nodeBuilder, &buffers, &n, layerIndex,
            kBufferIndex, vBufferIndex, config.zqPipeIndex,
            config.ropeSlice, config.kvCacheSlice, config.multiHeadAttSlice,
            config.nQNormColumns, config.nKNormColumns
        );

        // FFN segment (MoE or non-MoE)
        if (h->archType == SIMPLE_QWEN3_MOE) {
            buildMoEFFNSegment(&nodeBuilder, &buffers, &n, layerIndex, config.zqPipeIndex);
        } else {
            buildFFNSegment(&nodeBuilder, &buffers, &n, layerIndex, config.zqPipeIndex);
        }
    }

    // Build classifier segment
    buildClassifierSegment(&nodeBuilder, &buffers, &n, config.zqPipeIndex);

    n.nodeConfig = nodeBuilder.build();

    return n;
}
