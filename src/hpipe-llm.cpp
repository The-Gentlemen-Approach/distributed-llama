#include "hpipe-llm.hpp"
#include "llm-builder.hpp"
#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "mmap.hpp"
#include <algorithm>
#include <cstring>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cerrno>

// ==================================================================================
// Build Network (Partial for HPipe - Segment Based)
// ==================================================================================

SimpleLlmNet buildHpipeLlmNet(SimpleLlmHeader *h, NnUint nBatches, int startSegment, int endSegment) {
    SimpleLlmNet n;

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
    bool firstAttentionSeen = false;
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

                // First attention segment for this worker should use CAST instead of MERGE_ADD
                bool isFirstForWorker = !firstAttentionSeen && (startSegment > 0);
                firstAttentionSeen = true;

                buildAttentionSegment(
                    &nodeBuilder, &buffers, &n, layerIndex,
                    kBufferIndex, vBufferIndex, config.zqPipeIndex,
                    config.ropeSlice, config.kvCacheSlice, config.multiHeadAttSlice,
                    config.nQNormColumns, config.nKNormColumns,
                    isFirstForWorker
                );
            } else {
                // FFN Segment
                bool isLastForWorker = (seg == endSegment);
                bool isFirstForWorker = (seg == startSegment);

                if (h->archType == SIMPLE_QWEN3_MOE) {
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

// ==================================================================================
// HPipeLlmInference Implementation
// ==================================================================================

HPipeLlmInference::HPipeLlmInference(SimpleLlmNet *net, NnNetExecution *execution, NnExecutor *executor, int startSegment, int endSegment) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[net->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->execution = execution;
    this->executor = executor;
    this->startSegment = startSegment;
    this->endSegment = endSegment;

    int maxSegment = 2 * header->nLayers + 1;
    if (this->endSegment == -1) this->endSegment = maxSegment;

    // Determine Input/Output Pipes
    // zqPipeIndex is assumed to be 4 based on llm-builder logic (POS, TOK, X, LG, ZQ)
    // If not, we should look it up from netConfig if possible, but here we hardcode or pass it.
    // For now, assuming standard build order:
    this->zqPipeIndex = 4; 

    // Input Pipe Logic
    if (this->startSegment == 0) {
        // Embedding segment takes TOK pipe as input, but its output is X pipe.
        // Wait, buildEmbeddingSegment inputs are TOK and X. X is output.
        // So for the *next* segment, input would be X.
        // But here 'inputPipe' means "where do I put data received from network?"
        // If I am segment 0, I receive Tokens (integers) from Root/Network? 
        // Or if I am Root, I set tokens directly.
        // If this node starts at segment 0, it behaves as the Source.
        // The `tokenPipe` member handles the token input.
        // So `inputPipe` is only relevant if `startSegment > 0`.
        
        // However, technically, after embedding, data is in X pipe.
        // If startSegment == 0, we don't need `inputPipe` for receiving tensors.
        this->inputPipe = nullptr; 
    } else if (this->startSegment == 1) {
        // Segment 1 (Layer 0 Attn) expects input from Embedding.
        // Embedding writes to X pipe.
        // So if we are receiving from a node that did Embedding, we receive into X pipe.
        this->inputPipe = (float *)execution->pipes[net->xPipeIndex];
    } else {
        // Other segments (FFN, later Layers) expect input from ZQ pipe (previous layer/block output).
        // Wait, Attn output is Y/ZQ? 
        // In simple-llm.cpp:
        // Attn writes to zqPipeIndex (via CAST).
        // FFN inputs from zqPipeIndex (via MERGE_ADD).
        // So yes, ZQ pipe is the highway.
        this->inputPipe = (float *)execution->pipes[zqPipeIndex];
    }

    // Output Pipe Logic
    if (this->endSegment == maxSegment) {
        // Classifier writes to LG pipe (Logits).
        this->outputPipe = (float *)execution->pipes[net->logitsPipeIndex];
    } else if (this->endSegment == 0) {
        // Embedding writes to X pipe.
        this->outputPipe = (float *)execution->pipes[net->xPipeIndex];
    } else {
        // Layers write to ZQ pipe.
        this->outputPipe = (float *)execution->pipes[zqPipeIndex];
    }
}

void HPipeLlmInference::setBatchSize(NnUint batchSize) {
    execution->setBatchSize(batchSize);
}

void HPipeLlmInference::setPosition(NnUint position) {
    if (position + execution->batchSize - 1 >= header->seqLen) {
        throw std::runtime_error("Position exceeds sequence length");
    }
    for (NnUint i = 0; i < execution->batchSize; i++)
        positionPipe[i] = (float)(position + i);
}

void HPipeLlmInference::setToken(NnUint batchIndex, NnUint token) {
    if (batchIndex >= execution->batchSize) {
         throw std::runtime_error("Batch index out of bounds");
    }
    tokenPipe[batchIndex] = (float)token;
}

void HPipeLlmInference::forward() {
    executor->forward();
}

// ==================================================================================
// Load Weights (Segment-based)
// ==================================================================================

class SimpleWeightLoader {
private:
    NnExecutor *executor;
    NnByte *temp;
    NnSize tempSize;

public:
    SimpleWeightLoader(NnExecutor *executor) : executor(executor), temp(nullptr), tempSize(0) {}
    ~SimpleWeightLoader() {
        if (tempSize > 0) delete[] temp;
    }

    void loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
        executor->loadWeight(opName, opIndex, 0u, nBytes, weight);
    }

    void loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight) {
        executor->loadWeight(opName, opIndex, 0u, nBytes, weight);
    }

    void loadRowMatmulSlices(const char *opName, NnUint opIndex, NnUint expertIndex, NnRowMatmulSlice *slice, NnByte *weight) {
        const NnUint offset = expertIndex * slice->sliceSize.nBytes;
        executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
    }

    void loadColMatmulSlices(const char *opName, NnUint opIndex, NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight) {
        const NnUint offset = expertIndex * slice->sliceSize.nBytes;
        executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
    }
};

void loadHpipeLlmNetWeight(const char *path, SimpleLlmNet *net, NnExecutor *executor, int startSegment, int endSegment) {
    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);
    std::unique_ptr<MmapFile, void(*)(MmapFile *)> fdPtr(&file, closeMmapFile);
    
    printf("💿 Loading weights for segments [%d, %d]...\n", startSegment, endSegment);

    NnByte *data = (NnByte *)file.data;
    NnByte *b = &data[net->header->headerSize];
    
    SimpleWeightLoader loader(executor);
    int maxSegment = 2 * net->header->nLayers + 1;

    // Embedding (Segment 0)
    if (startSegment <= 0 && endSegment >= 0) {
        loader.loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);
    }
    b += net->tokenEmbeddingSize.nBytes;

    // Layers (Segments 1 to 2*N)
    for (NnUint layerIndex = 0u; layerIndex < net->header->nLayers; layerIndex++) {
        int attentionSegment = 2 * layerIndex + 1;
        int ffnSegment = 2 * layerIndex + 2;
        
        bool loadAttention = (attentionSegment >= startSegment && attentionSegment <= endSegment);
        bool loadFFN = (ffnSegment >= startSegment && ffnSegment <= endSegment);

        // Attention weights
        if (loadAttention) {
            loader.loadRowMatmulSlices("block_matmul_q", layerIndex, 0u, &net->qSlice, b);
        }
        b += net->qSlice.size.nBytes;

        if (loadAttention) {
            loader.loadRowMatmulSlices("block_matmul_k", layerIndex, 0u, &net->kSlice, b);
        }
        b += net->kSlice.size.nBytes;

        if (loadAttention) {
            loader.loadRowMatmulSlices("block_matmul_v", layerIndex, 0u, &net->vSlice, b);
        }
        b += net->vSlice.size.nBytes;

        if (loadAttention) {
            loader.loadColMatmulSlices("block_matmul_wo", layerIndex, 0u, &net->woSlice, b);
        }
        b += net->woSlice.size.nBytes;

        // FFN weights
        if (net->header->nExperts > 0u) {
            // MoE model
            if (loadFFN) {
                loader.loadAll("block_moe_gate", layerIndex, net->moeGateSize.nBytes, b);
            }
            b += net->moeGateSize.nBytes;
            
            for (NnUint expertIndex = 0u; expertIndex < net->header->nExperts; expertIndex++) {
                if (loadFFN) {
                    loader.loadRowMatmulSlices("block_matmul_w1", layerIndex, expertIndex, &net->w1Slice, b);
                }
                b += net->w1Slice.sliceSize.nBytes;
                
                if (loadFFN) {
                    loader.loadColMatmulSlices("block_matmul_w2", layerIndex, expertIndex, &net->w2Slice, b);
                }
                b += net->w2Slice.sliceSize.nBytes;
                
                if (loadFFN) {
                    loader.loadRowMatmulSlices("block_matmul_w3", layerIndex, expertIndex, &net->w3Slice, b);
                }
                b += net->w3Slice.sliceSize.nBytes;
            }
        } else {
            // Standard FFN
            if (loadFFN) {
                loader.loadRowMatmulSlices("block_matmul_w1", layerIndex, 0u, &net->w1Slice, b);
            }
            b += net->w1Slice.size.nBytes;
            
            if (loadFFN) {
                loader.loadColMatmulSlices("block_matmul_w2", layerIndex, 0u, &net->w2Slice, b);
            }
            b += net->w2Slice.size.nBytes;
            
            if (loadFFN) {
                loader.loadRowMatmulSlices("block_matmul_w3", layerIndex, 0u, &net->w3Slice, b);
            }
            b += net->w3Slice.size.nBytes;
        }

        // QK RMS Norms (Qwen3 models)
        if (net->header->archType == SIMPLE_QWEN3 || net->header->archType == SIMPLE_QWEN3_MOE) {
            if (loadAttention) {
                loader.loadAll("block_norm_q", layerIndex, net->qkRmsNormSize.nBytes, b);
            }
            b += net->qkRmsNormSize.nBytes;
            
            if (loadAttention) {
                loader.loadAll("block_norm_k", layerIndex, net->qkRmsNormSize.nBytes, b);
            }
            b += net->qkRmsNormSize.nBytes;
        }

        // Layer RMS Norms
        if (loadAttention) {
            loader.loadAll("block_norm_0", layerIndex, net->rmsNormSize.nBytes, b);
        }
        b += net->rmsNormSize.nBytes;
        
        if (loadFFN) {
            loader.loadAll("block_norm_1", layerIndex, net->rmsNormSize.nBytes, b);
        }
        b += net->rmsNormSize.nBytes;
    }

    // Classifier (Last segment)
    if (startSegment <= maxSegment && endSegment >= maxSegment) {
        loader.loadAll("final_norm", 0, net->rmsNormSize.nBytes, b);
    }
    b += net->rmsNormSize.nBytes;

    if (startSegment <= maxSegment && endSegment >= maxSegment) {
        loader.loadRowMatmulSlices("final_matmul_logits", 0, 0u, &net->wclsSlice, b);
    }

    printf("✓ Weights loaded for segments [%d, %d]\n", startSegment, endSegment);
}
