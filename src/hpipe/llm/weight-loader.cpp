#include "hpipe/llm/weight-loader.hpp"
#include "common/mmap.hpp"
#include <cstdio>
#include <memory>
#include <vector>

// ==================================================================================
// Load Weights (Segment-based)
// ==================================================================================

class SegmentWeightLoader {
private:
    NnExecutor *executor;
    NnByte *temp;
    NnSize tempSize;

public:
    SegmentWeightLoader(NnExecutor *executor) : executor(executor), temp(nullptr), tempSize(0) {}
    ~SegmentWeightLoader() {
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

    SegmentWeightLoader loader(executor);
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
