#include "hpipe/llm/network-builder.hpp"
#include "common/llm-builder.hpp"
#include "nn/nn-config-builder.hpp"
#include <vector>

LlmNet buildHpipeLlmNet(LlmHeader *h, NnUint nBatches, int startSegment, int endSegment) {
    LlmNet n;

    int maxSegment = 2 * h->nLayers + 1; // 0 (Emb) + 2*N (Layers) + 1 (Class)
    if (endSegment == -1) endSegment = maxSegment;

    // ==================================================================================
    // CRITICAL: For H-Pipe, force syncType to F32 to prevent quantization loss
    // accumulation across multiple workers. Model weights remain Q40, but
    // intermediate activations transferred between workers must be F32.
    // ==================================================================================
    NnFloatType originalSyncType = h->syncType;
    h->syncType = F_32;

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

    // Loop through assigned segments
    for (int seg = startSegment; seg <= endSegment; seg++) {
        if (seg == 0) {
            // Segment 0: Embedding
            buildEmbeddingSegment(&nodeBuilder, n.tokenPipeIndex, n.xPipeIndex, n.tokenEmbeddingSize);
        } else if (seg == maxSegment) {
            // Last Segment: Classifier
            buildClassifierSegment(&nodeBuilder, &buffers, &n, config.zqPipeIndex);
        } else {
            // Layer Segments (1 to 2*N)
            // odd: Attention, even: FFN
            // Layer index k corresponds to segments 2*k+1 and 2*k+2
            // k = (seg - 1) / 2

            NnUint layerIndex = (seg - 1) / 2;
            bool isAttention = (seg % 2 != 0);

            if (isAttention) {
                // Attention Segment
                const NnUint kBufferIndex = nodeBuilder.addBuffer("k", config.kvCacheSlice.keySize);
                const NnUint vBufferIndex = nodeBuilder.addBuffer("v", config.kvCacheSlice.valueSize);

                // First segment for worker (if not embedding/classifier) should use CAST
                // This applies when worker receives data from previous worker
                bool isFirstForWorker = (seg == startSegment) && (startSegment > 0);

                // Last segment for worker should output accumulated residual
                bool isLastForWorker = (seg == endSegment);

                buildAttentionSegment(
                    &nodeBuilder, &buffers, &n, layerIndex,
                    kBufferIndex, vBufferIndex, config.zqPipeIndex,
                    config.ropeSlice, config.kvCacheSlice, config.multiHeadAttSlice,
                    config.nQNormColumns, config.nKNormColumns,
                    isFirstForWorker, isLastForWorker
                );
            } else {
                // FFN Segment
                bool isLastForWorker = (seg == endSegment);
                bool isFirstForWorker = (seg == startSegment);

                if (h->archType == LLM_QWEN3_MOE) {
                    buildMoEFFNSegment(&nodeBuilder, &buffers, &n, layerIndex, config.zqPipeIndex, isLastForWorker, isFirstForWorker);
                } else {
                    buildFFNSegment(&nodeBuilder, &buffers, &n, layerIndex, config.zqPipeIndex, isLastForWorker, isFirstForWorker);
                }
            }
        }
    }

    n.nodeConfig = nodeBuilder.build();

    // Restore original syncType
    h->syncType = originalSyncType;

    return n;
}
