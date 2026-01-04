#include "common/llm-builder.hpp"
#include "simple/utils.hpp"
#include <stdexcept>

// ==================================================================================
// Network Initialization
// ==================================================================================

LlmNetworkConfig initializeLlmNetwork(LlmNet *n, LlmHeader *h, NnUint nBatches) {
    // Calculate dimensions
    NnUint nExpertsOr1 = std::max(h->nExperts, 1u);
    NnUint nActiveExpertsOr1 = std::max(h->nActiveExperts, 1u);
    NnUint ffDim = h->hiddenDim;

    if (h->archType == LLM_QWEN3_MOE)
        ffDim = h->moeHiddenDim;

    // Initialize LlmNet sizes
    n->tokenEmbeddingSize = size2D(F_32, h->vocabSize, h->dim);
    n->rmsNormSize = size1D(F_32, h->dim);
    n->qkRmsNormSize = size1D(F_32, h->headDim);
    n->moeGateSize = size2D(F_32, h->dim, h->nExperts);

    // Initialize weight slices (single-node: no slicing)
    n->qSlice = initSingleNodeRowMatmulSlice(h->weightType, h->dim, h->qDim);
    n->kSlice = initSingleNodeRowMatmulSlice(h->weightType, h->dim, h->kvDim);
    n->vSlice = initSingleNodeRowMatmulSlice(h->weightType, h->dim, h->kvDim);
    n->woSlice = initSingleNodeColMatmulSlice(h->weightType, h->qDim, h->dim);
    n->w1Slice = initSingleNodeRowMatmulSlice(h->weightType, h->dim, ffDim);
    n->w2Slice = initSingleNodeColMatmulSlice(h->weightType, ffDim, h->dim);
    n->w3Slice = initSingleNodeRowMatmulSlice(h->weightType, h->dim, ffDim);
    n->wclsSlice = initSingleNodeRowMatmulSlice(h->weightType, h->dim, h->vocabSize);

    // Calculate norm columns (Qwen3-specific)
    LlmNetworkConfig config;
    config.nQNormColumns = 1;
    config.nKNormColumns = 1;
    config.nInvBufferColumns = 1;
    if (h->archType == LLM_QWEN3 || h->archType == LLM_QWEN3_MOE) {
        config.nQNormColumns = n->qSlice.d0 / h->headDim;
        config.nKNormColumns = n->kSlice.d0 / h->headDim;
        config.nInvBufferColumns = std::max(config.nQNormColumns, config.nKNormColumns);
    }

    // Build network config (pipes)
    NnNetConfigBuilder netBuilder(1, nBatches);  // Single-node

    n->positionPipeIndex = netBuilder.addPipe("POS", size2D(F_32, nBatches, 1));
    n->tokenPipeIndex = netBuilder.addPipe("TOK", size2D(F_32, nBatches, 1));
    n->xPipeIndex = netBuilder.addPipe("X", size2D(F_32, nBatches, h->dim));
    n->logitsPipeIndex = netBuilder.addPipe("LG", size2D(F_32, nBatches, h->vocabSize));
    config.zqPipeIndex = netBuilder.addPipe("ZQ", size2D(h->syncType, nBatches, h->dim));

    n->header = h;
    n->netConfig = netBuilder.build();

    // Initialize KV cache slice
    config.kvCacheSlice.kvDim0 = h->kvDim;
    config.kvCacheSlice.keySize = size2D(F_32, h->seqLen, h->kvDim);
    config.kvCacheSlice.valueSize = size2D(F_32, h->seqLen, h->kvDim);

    // Initialize multi-head attention slice
    config.multiHeadAttSlice.nHeads = h->nHeads;
    config.multiHeadAttSlice.nHeads0 = h->nHeads;
    config.multiHeadAttSlice.attSize = size3D(F_32, nBatches, h->nHeads, h->seqLen);

    // Initialize RoPE slice
    config.ropeSlice.kvDim = h->kvDim;
    config.ropeSlice.nKvHeads = h->nKvHeads;
    config.ropeSlice.seqLen = h->seqLen;
    config.ropeSlice.headDim = h->headDim;
    config.ropeSlice.ropeTheta = h->ropeTheta;
    config.ropeSlice.qDim0 = h->qDim;
    config.ropeSlice.kvDim0 = h->kvDim;

    if (h->ropeType == ROPE_LLAMA || h->ropeType == ROPE_LLAMA3_1) {
        config.ropeSlice.kvDimStart = 0;
        config.ropeSlice.qDimStart = 0;
        config.ropeSlice.qDimEnd = h->qDim;
        config.ropeSlice.qShift = 0;
        config.ropeSlice.sliceDim = h->qDim;
        config.ropeSlice.cacheSize = size2D(F_32, h->seqLen, config.ropeSlice.sliceDim);
    } else if (h->ropeType == ROPE_FALCON) {
        config.ropeSlice.cacheSize = size2D(F_32, h->seqLen, h->headDim);
    } else {
        throw std::runtime_error("Unsupported rope type");
    }

    return config;
}

// ==================================================================================
// Buffer Allocation
// ==================================================================================

LlmBufferIndices allocateLlmBuffers(
    NnNodeConfigBuilder *nodeBuilder,
    LlmHeader *h,
    LlmNet *n,
    NnUint nBatches,
    NnUint nInvBufferColumns,
    const NnRopeSlice &ropeSlice,
    const NnMultiHeadAttSlice &multiHeadAttSlice
) {
    NnUint nExpertsOr1 = std::max(h->nExperts, 1u);
    NnUint nActiveExpertsOr1 = std::max(h->nActiveExperts, 1u);

    LlmBufferIndices buffers;

    buffers.xBufferIndex = nodeBuilder->addBuffer("x", size2D(F_32, nBatches, h->dim));
    buffers.yBufferIndex = nodeBuilder->addBuffer("y", size2D(F_32, nBatches, h->dim));
    buffers.yqBufferIndex = h->syncType == F_32
        ? buffers.yBufferIndex
        : nodeBuilder->addBuffer("q_y", size2D(h->syncType, nBatches, h->dim));

    buffers.zBufferIndex = nodeBuilder->addBuffer("z", size2D(F_32, nBatches, h->qDim));
    buffers.zqSliceBufferIndex = nodeBuilder->addBuffer("q_z_slice", size2D(h->syncType, nBatches, h->qDim));

    buffers.qBufferIndex = nodeBuilder->addBuffer("q", size2D(F_32, nBatches, n->qSlice.d0));
    buffers.kTempBufferIndex = nodeBuilder->addBuffer("k_temp", size2D(F_32, nBatches, n->kSlice.d0));
    buffers.vTempBufferIndex = nodeBuilder->addBuffer("v_temp", size2D(F_32, nBatches, n->vSlice.d0));

    buffers.invRmsBufferIndex = nodeBuilder->addBuffer("inv_rms", size2D(F_32, nBatches, nInvBufferColumns));

    buffers.ropeCacheBufferIndex = nodeBuilder->addBuffer("rope_cache", ropeSlice.cacheSize);
    buffers.attBufferIndex = nodeBuilder->addBuffer("att", multiHeadAttSlice.attSize);
    buffers.logitsSliceBufferIndex = nodeBuilder->addBuffer("lg", size2D(F_32, nBatches, h->vocabSize));

    buffers.dBufferIndex = nodeBuilder->addBuffer("d", size2D(F_32, nBatches, n->w1Slice.d0));
    buffers.dqBufferIndex = h->syncType == F_32
        ? buffers.dBufferIndex
        : nodeBuilder->addBuffer("q_d", size2D(h->syncType, nBatches, n->w1Slice.d0));
    buffers.lBufferIndex = nodeBuilder->addBuffer("l", size2D(F_32, nBatches, n->w3Slice.d0));

    buffers.moeGtBufferIndex = nodeBuilder->addBuffer("gt", size2D(F_32, nBatches, nExpertsOr1));
    buffers.moeExpertIndexesBufferIndex = nodeBuilder->addBuffer("act_exp_ix", size2D(F_32, nBatches, nActiveExpertsOr1));
    buffers.moeYBufferIndex = nodeBuilder->addBuffer("moe_y", size3D(F_32, nActiveExpertsOr1, nBatches, h->dim));
    buffers.moeYqBufferIndex = h->syncType == F_32
        ? buffers.moeYBufferIndex
        : nodeBuilder->addBuffer("q_moe_y", size3D(h->syncType, nActiveExpertsOr1, nBatches, h->dim));
    buffers.moeDBufferIndex = nodeBuilder->addBuffer("moe_d", size3D(F_32, nActiveExpertsOr1, nBatches, n->w1Slice.d0));
    buffers.moeDQBufferIndex = h->syncType == F_32
        ? buffers.moeDBufferIndex
        : nodeBuilder->addBuffer("q_moe_d", size3D(h->syncType, nActiveExpertsOr1, nBatches, n->w1Slice.d0));
    buffers.moeLBufferIndex = nodeBuilder->addBuffer("moe_l", size3D(F_32, nActiveExpertsOr1, nBatches, n->w3Slice.d0));
    buffers.moeSBufferIndex = nodeBuilder->addBuffer("moe_s", size3D(F_32, nActiveExpertsOr1, nBatches, 1));

    return buffers;
}

// ==================================================================================
// Embedding Segment
// ==================================================================================

void buildEmbeddingSegment(
    NnNodeConfigBuilder *nodeBuilder,
    NnUint tokenPipeIndex,
    NnUint xPipeIndex,
    NnSize3D tokenEmbeddingSize
) {
    NnSegmentConfigBuilder start;
    start.addOp(
        OP_EMBEDDING, "embedding", 0,
        pointerBatchConfig(SRC_PIPE, tokenPipeIndex),
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        tokenEmbeddingSize,
        NnEmbeddingOpConfig{});
    nodeBuilder->addSegment(start.build());
}

// ==================================================================================
// Attention Segment
// ==================================================================================

void buildAttentionSegment(
    NnNodeConfigBuilder *nodeBuilder,
    const LlmBufferIndices *buf,
    LlmNet *net,
    NnUint layerIndex,
    NnUint kBufferIndex,
    NnUint vBufferIndex,
    NnUint zqPipeIndex,
    const NnRopeSlice &ropeSlice,
    const NnKvCacheSlice &kvCacheSlice,
    const NnMultiHeadAttSlice &multiHeadAttSlice,
    NnUint nQNormColumns,
    NnUint nKNormColumns,
    bool isFirstSegmentForWorker,
    bool isLastSegmentForWorker
) {
    LlmHeader *h = net->header;
    bool isFirstLayer = (layerIndex == 0) || isFirstSegmentForWorker;
    NnUint moeExpertIndexesBufferIndex = 0;

    NnSegmentConfigBuilder att;

    // Input handling: first layer vs other layers
    if (isFirstLayer) {
        NnUint srcPipeIndex = (layerIndex == 0) ? net->xPipeIndex : zqPipeIndex;
        att.addOp(
            OP_CAST, "block_cast_x", layerIndex,
            pointerBatchConfig(SRC_PIPE, srcPipeIndex),
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            size0(),
            NnCastOpCodeConfig{});
    } else {
        att.addOp(
            OP_MERGE_ADD, "block_merge_add", layerIndex,
            pointerBatchConfig(SRC_PIPE, zqPipeIndex),
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            size0(),
            NnMergeAddOpCodeConfig{});
    }

    // RMS Norm
    att.addOp(
        OP_INV_RMS, "block_norm_pre_0", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->invRmsBufferIndex),
        size0(),
        NnInvRmsOpConfig{h->normEpsilon, 1});
    att.addOp(
        OP_RMS_NORM, "block_norm_0", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
        net->rmsNormSize,
        NnRmsNormOpConfig{buf->invRmsBufferIndex, 1});
    if (buf->yBufferIndex != buf->yqBufferIndex) {
        att.addOp(
            OP_CAST, "block_cast_y", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->yqBufferIndex),
            size0(),
            NnCastOpCodeConfig{});
    }

    // QKV projections
    att.addOp(
        OP_MATMUL, "block_matmul_q", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->yqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->qBufferIndex),
        size2D(h->weightType, net->qSlice.n, net->qSlice.d0),
        NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
    att.addOp(
        OP_MATMUL, "block_matmul_k", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->yqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->kTempBufferIndex),
        size2D(h->weightType, net->kSlice.n, net->kSlice.d0),
        NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
    att.addOp(
        OP_MATMUL, "block_matmul_v", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->yqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->vTempBufferIndex),
        size2D(h->weightType, net->vSlice.n, net->vSlice.d0),
        NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});

    // QK RMS Norm (Qwen3 only)
    if (h->archType == LLM_QWEN3 || h->archType == LLM_QWEN3_MOE) {
        att.addOp(OP_INV_RMS, "block_norm_pre_q", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->qBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->invRmsBufferIndex),
            size0(),
            NnInvRmsOpConfig{h->normEpsilon, nQNormColumns});
        att.addOp(
            OP_RMS_NORM, "block_norm_q", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->qBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->qBufferIndex),
            net->qkRmsNormSize,
            NnRmsNormOpConfig{buf->invRmsBufferIndex, nQNormColumns});

        att.addOp(OP_INV_RMS, "block_norm_pre_k", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->kTempBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->invRmsBufferIndex),
            size0(),
            NnInvRmsOpConfig{h->normEpsilon, nKNormColumns});
        att.addOp(
            OP_RMS_NORM, "block_norm_k", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->kTempBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->kTempBufferIndex),
            net->qkRmsNormSize,
            NnRmsNormOpConfig{buf->invRmsBufferIndex, nKNormColumns});
    }

    // RoPE
    att.addOp(
        OP_ROPE, "block_rope_q", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->qBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->qBufferIndex),
        size0(),
        NnRopeOpConfig{h->ropeType, 1, net->positionPipeIndex, buf->ropeCacheBufferIndex,
            h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen,
            ropeSlice});
    att.addOp(
        OP_ROPE, "block_rope_k", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->kTempBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->kTempBufferIndex),
        size0(),
        NnRopeOpConfig{h->ropeType, 0, net->positionPipeIndex, buf->ropeCacheBufferIndex,
            h->ropeScalingFactor, h->ropeScalingLowFreqFactor, h->ropeScalingHighFreqFactory, h->ropeScalingOrigMaxSeqLen,
            ropeSlice});

    // KV Cache shift
    att.addOp(
        OP_SHIFT, "block_shift_k", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->kTempBufferIndex),
        pointerRawConfig(SRC_BUFFER, kBufferIndex),
        size0(),
        NnShiftOpCodeConfig{net->positionPipeIndex});
    att.addOp(
        OP_SHIFT, "block_shift_v", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->vTempBufferIndex),
        pointerRawConfig(SRC_BUFFER, vBufferIndex),
        size0(),
        NnShiftOpCodeConfig{net->positionPipeIndex});

    // Multi-head attention
    att.addOp(
        OP_MULTIHEAD_ATT, "block_multihead_att", layerIndex,
        pointerBatchedSliceConfig(SRC_BUFFER, buf->zBufferIndex),
        pointerBatchedSliceConfig(SRC_BUFFER, buf->zBufferIndex),
        size0(),
        NnMultiHeadAttOpConfig{
            multiHeadAttSlice.nHeads, multiHeadAttSlice.nHeads0,
            h->nKvHeads, h->headDim, h->seqLen,
            net->qSlice.d0, kvCacheSlice.kvDim0,
            net->positionPipeIndex,
            buf->qBufferIndex,
            kBufferIndex,
            vBufferIndex,
            buf->attBufferIndex});
    att.addOp(
        OP_CAST, "block_cast_y2", layerIndex,
        pointerBatchedSliceConfig(SRC_BUFFER, buf->zBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->zqSliceBufferIndex),
        size0(),
        NnCastOpCodeConfig{});

    // Output projection
    att.addOp(
        OP_MATMUL, "block_matmul_wo", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->zqSliceBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
        size2D(h->weightType, net->woSlice.n0, net->woSlice.d),
        NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});

    if (isLastSegmentForWorker) {
        // Last segment for worker: Output accumulated residual + attention result
        att.addOp(
            OP_MERGE_ADD, "block_merge_add_att_boundary", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            size0(),
            NnMergeAddOpCodeConfig{});
        att.addOp(
            OP_CAST, "block_cast_d", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
            size0(),
            NnCastOpCodeConfig{});
    } else {
        // Internal segment: Output only attention result (delta)
        att.addOp(
            OP_CAST, "block_cast_d", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
            pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
            size0(),
            NnCastOpCodeConfig{});
    }

    nodeBuilder->addSegment(att.build());
}

// ==================================================================================
// FFN Segment (Non-MoE)
// ==================================================================================

void buildFFNSegment(
    NnNodeConfigBuilder *nodeBuilder,
    const LlmBufferIndices *buf,
    LlmNet *net,
    NnUint layerIndex,
    NnUint zqPipeIndex,
    bool isLastSegmentForWorker,
    bool isFirstSegmentForWorker
) {
    LlmHeader *h = net->header;
    NnUint moeExpertIndexesBufferIndex = 0;

    NnSegmentConfigBuilder ff;

    // Residual connection
    if (isFirstSegmentForWorker) {
        // First segment for worker: Previous worker sent accumulated residual
        // Just copy it to xBuffer (no addition needed)
        ff.addOp(
            OP_CAST, "block_cast_input", layerIndex,
            pointerBatchConfig(SRC_PIPE, zqPipeIndex),
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            size0(),
            NnCastOpCodeConfig{});
    } else {
        // Internal segment: Add FFN delta to accumulated residual
        ff.addOp(
            OP_MERGE_ADD, "block_merge_add2", layerIndex,
            pointerBatchConfig(SRC_PIPE, zqPipeIndex),
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            size0(),
            NnMergeAddOpCodeConfig{});
    }

    // RMS Norm
    ff.addOp(
        OP_INV_RMS, "block_norm_pre_1", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->invRmsBufferIndex),
        size0(),
        NnInvRmsOpConfig{h->normEpsilon, 1});
    ff.addOp(
        OP_RMS_NORM, "block_norm_1", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
        net->rmsNormSize,
        NnRmsNormOpConfig{buf->invRmsBufferIndex, 1});

    if (buf->yBufferIndex != buf->yqBufferIndex) {
        ff.addOp(
            OP_CAST, "block_cast_y3", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->yqBufferIndex),
            size0(),
            NnCastOpCodeConfig{});
    }

    // W1 and W3 projections
    ff.addOp(
        OP_MATMUL, "block_matmul_w1", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->yqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->dBufferIndex),
        size2D(h->weightType, net->w1Slice.n, net->w1Slice.d0),
        NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});
    ff.addOp(
        OP_MATMUL, "block_matmul_w3", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->yqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->lBufferIndex),
        size2D(h->weightType, net->w3Slice.n, net->w3Slice.d0),
        NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});

    // SiLU activation
    ff.addOp(
        OP_SILU, "block_act", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->dBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->dBufferIndex),
        size0(),
        NnSiluOpCodeConfig{});

    // Element-wise multiplication
    ff.addOp(
        OP_MUL, "block_mul", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->dBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->dBufferIndex),
        size0(),
        NnMulOpCodeConfig{buf->lBufferIndex});

    if (buf->dBufferIndex != buf->dqBufferIndex) {
        ff.addOp(
            OP_CAST, "block_cast_d2", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->dBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->dqBufferIndex),
            size0(),
            NnCastOpCodeConfig{});
    }

    // W2 projection
    ff.addOp(
        OP_MATMUL, "block_matmul_w2", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->dqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
        size2D(h->weightType, net->w2Slice.n0, net->w2Slice.d),
        NnMatmulOpConfig{0, 0, moeExpertIndexesBufferIndex});

    if (isLastSegmentForWorker) {
        // Last segment for worker: Output accumulated residual + FFN result
        ff.addOp(
            OP_MERGE_ADD, "block_merge_add_boundary", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            size0(),
            NnMergeAddOpCodeConfig{});
        ff.addOp(
            OP_CAST, "block_cast_d3", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
            size0(),
            NnCastOpCodeConfig{});
    } else {
        // Internal layer: Output only FFN result (delta)
        ff.addOp(
            OP_CAST, "block_cast_d3", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
            pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
            size0(),
            NnCastOpCodeConfig{});
    }

    nodeBuilder->addSegment(ff.build());
}

// ==================================================================================
// MoE FFN Segment
// ==================================================================================

void buildMoEFFNSegment(
    NnNodeConfigBuilder *nodeBuilder,
    const LlmBufferIndices *buf,
    LlmNet *net,
    NnUint layerIndex,
    NnUint zqPipeIndex,
    bool isLastSegmentForWorker,
    bool isFirstSegmentForWorker
) {
    LlmHeader *h = net->header;

    NnSegmentConfigBuilder ff;

    // Residual connection
    if (isFirstSegmentForWorker) {
        // First segment for worker: Previous worker sent accumulated residual
        // Just copy it to xBuffer (no addition needed)
        ff.addOp(
            OP_CAST, "block_cast_input", layerIndex,
            pointerBatchConfig(SRC_PIPE, zqPipeIndex),
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            size0(),
            NnCastOpCodeConfig{});
    } else {
        // Internal segment: Add FFN delta to accumulated residual
        ff.addOp(
            OP_MERGE_ADD, "block_merge_add2", layerIndex,
            pointerBatchConfig(SRC_PIPE, zqPipeIndex),
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            size0(),
            NnMergeAddOpCodeConfig{});
    }

    // RMS Norm
    ff.addOp(
        OP_INV_RMS, "block_norm_pre_1", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->invRmsBufferIndex),
        size0(),
        NnInvRmsOpConfig{h->normEpsilon, 1});
    ff.addOp(
        OP_RMS_NORM, "block_norm_1", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
        net->rmsNormSize,
        NnRmsNormOpConfig{buf->invRmsBufferIndex, 1});

    // Repeat Y for MoE processing
    ff.addOp(
        OP_REPEAT_Z, "block_moe_y_repeat", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->moeYqBufferIndex),
        size0(),
        NnRepeatZOpCodeConfig{});

    // MoE gate
    ff.addOp(
        OP_MATMUL, "block_moe_gate", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->moeGtBufferIndex),
        net->moeGateSize,
        NnMatmulOpConfig{0, 0, buf->moeExpertIndexesBufferIndex});
    ff.addOp(
        OP_SOFTMAX, "block_moe_softmax", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->moeGtBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->moeGtBufferIndex),
        size0(),
        NnSoftmaxOpCodeConfig{});
    ff.addOp(
        OP_MOE_GATE, "block_moe_gate2", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->moeGtBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->moeSBufferIndex),
        size0(),
        NnMoeGateOpCodeConfig{h->nActiveExperts, 1u, buf->moeExpertIndexesBufferIndex});

    // W1 and W3 projections (expert-specific)
    ff.addOp(
        OP_MATMUL, "block_matmul_w1", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->moeYqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->moeDBufferIndex),
        size3D(h->weightType, h->nExperts, net->w1Slice.n, net->w1Slice.d0),
        NnMatmulOpConfig{h->nExperts, h->nActiveExperts, buf->moeExpertIndexesBufferIndex});
    ff.addOp(
        OP_MATMUL, "block_matmul_w3", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->moeYqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->moeLBufferIndex),
        size3D(h->weightType, h->nExperts, net->w3Slice.n, net->w3Slice.d0),
        NnMatmulOpConfig{h->nExperts, h->nActiveExperts, buf->moeExpertIndexesBufferIndex});

    // SiLU activation
    ff.addOp(
        OP_SILU, "block_act", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->moeDBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->moeDBufferIndex),
        size0(),
        NnSiluOpCodeConfig{});

    // Element-wise multiplication
    ff.addOp(
        OP_MUL, "block_mul", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->moeDBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->moeDBufferIndex),
        size0(),
        NnMulOpCodeConfig{buf->moeLBufferIndex});

    if (buf->moeDBufferIndex != buf->moeDQBufferIndex) {
        ff.addOp(
            OP_CAST, "block_cast_d2", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->moeDBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->moeDQBufferIndex),
            size0(),
            NnCastOpCodeConfig{});
    }

    // W2 projection (expert-specific)
    ff.addOp(
        OP_MATMUL, "block_matmul_w2", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->moeDQBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->moeYBufferIndex),
        size3D(h->weightType, h->nExperts, net->w2Slice.n0, net->w2Slice.d),
        NnMatmulOpConfig{h->nExperts, h->nActiveExperts, buf->moeExpertIndexesBufferIndex});

    // Scale by gate weights
    ff.addOp(
        OP_SCALE, "block_moe_scale", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->moeYBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->moeYBufferIndex),
        size0(),
        NnScaleOpCodeConfig{buf->moeSBufferIndex});

    // Merge expert outputs
    ff.addOp(
        OP_MERGE_SUM, "block_moe_merge_sum", layerIndex,
        pointerBatchConfig(SRC_BUFFER, buf->moeYBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
        size0(),
        NnMergeSumOpCodeConfig{});

    if (isLastSegmentForWorker) {
        // Last segment for worker: Output accumulated residual + MoE result
        ff.addOp(
            OP_MERGE_ADD, "block_merge_add_boundary", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            size0(),
            NnMergeAddOpCodeConfig{});
        ff.addOp(
            OP_CAST, "block_cast_d3", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
            pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
            size0(),
            NnCastOpCodeConfig{});
    } else {
        // Internal layer: Output only MoE result (delta)
        ff.addOp(
            OP_CAST, "block_cast_d3", layerIndex,
            pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
            pointerBatchedSliceConfig(SRC_PIPE, zqPipeIndex),
            size0(),
            NnCastOpCodeConfig{});
    }

    nodeBuilder->addSegment(ff.build());
}

// ==================================================================================
// Classifier Segment
// ==================================================================================

void buildClassifierSegment(
    NnNodeConfigBuilder *nodeBuilder,
    const LlmBufferIndices *buf,
    LlmNet *net,
    NnUint zqPipeIndex
) {
    LlmHeader *h = net->header;

    NnSegmentConfigBuilder end;

    // Final residual connection
    end.addOp(
        OP_MERGE_ADD, "final_merge_add", 0,
        pointerBatchConfig(SRC_PIPE, zqPipeIndex),
        pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
        size0(),
        NnMergeAddOpCodeConfig{});

    // Final RMS Norm
    end.addOp(
        OP_INV_RMS, "final_norm_pre", 0,
        pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->invRmsBufferIndex),
        size0(),
        NnInvRmsOpConfig{h->normEpsilon, 1});
    end.addOp(
        OP_RMS_NORM, "final_norm", 0,
        pointerBatchConfig(SRC_BUFFER, buf->xBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
        net->rmsNormSize,
        NnRmsNormOpConfig{buf->invRmsBufferIndex, 1});
    if (buf->yBufferIndex != buf->yqBufferIndex) {
        end.addOp(
            OP_CAST, "final_cast_y", 0,
            pointerBatchConfig(SRC_BUFFER, buf->yBufferIndex),
            pointerBatchConfig(SRC_BUFFER, buf->yqBufferIndex),
            size0(),
            NnCastOpCodeConfig{});
    }

    // Classifier projection
    end.addOp(
        OP_MATMUL, "final_matmul_logits", 0,
        pointerBatchConfig(SRC_BUFFER, buf->yqBufferIndex),
        pointerBatchConfig(SRC_BUFFER, buf->logitsSliceBufferIndex),
        size2D(h->weightType, net->wclsSlice.n, net->wclsSlice.d0),
        NnMatmulOpConfig{});
    end.addOp(
        OP_CAST, "final_cast_logits", 0,
        pointerBatchConfig(SRC_BUFFER, buf->logitsSliceBufferIndex),
        pointerBatchedSliceConfig(SRC_PIPE, net->logitsPipeIndex),
        size0(),
        NnCastOpCodeConfig{});

    nodeBuilder->addSegment(end.build());
}

void releaseLlmNet(LlmNet *net) {
    releaseNodeConfig(&net->nodeConfig);
    releaseNetConfig(&net->netConfig);
}
