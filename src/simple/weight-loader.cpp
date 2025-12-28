#include "simple/weight-loader.hpp"
#include "common/mmap.hpp"
#include <cstdio>
#include <memory>
#include <vector>
#include <string>

// ==================================================================================
// Load Weights (Local Implementation)
// ==================================================================================

// Helper class for local weight loading, replacing NnRootWeightLoader functionality
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
        // Assuming nNodes = 1 for simple execution
        // Direct load without slicing logic for distributed nodes
        const NnUint offset = expertIndex * slice->sliceSize.nBytes;
        executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
    }

    void loadColMatmulSlices(const char *opName, NnUint opIndex, NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight) {
        // Assuming nNodes = 1
        const NnUint offset = expertIndex * slice->sliceSize.nBytes;
        executor->loadWeight(opName, opIndex, offset, slice->sliceSize.nBytes, weight);
    }
};

void loadLlmNetWeight(const char *path, LlmNet *net, NnExecutor *executor) {
    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);

    // Simple execution assumes 1 node for now
    // assert(net->netConfig.nNodes == 1u);

#if DEBUG_USE_MMAP_FOR_WEIGHTS
    // pass
#else
    std::unique_ptr<MmapFile, void(*)(MmapFile *)> fdPtr(&file, closeMmapFile);
    printf("[INFO] Loading weights...\n");
#endif

    Timer timer;
    NnByte *data = (NnByte *)file.data;
    NnByte *b = &data[net->header->headerSize];
    
    SimpleWeightLoader loader(executor);

    loader.loadRoot("embedding", 0, net->tokenEmbeddingSize.nBytes, b);
    b += net->tokenEmbeddingSize.nBytes;

    for (NnUint layerIndex = 0u; layerIndex < net->header->nLayers; layerIndex++) {
        loader.loadRowMatmulSlices("block_matmul_q", layerIndex, 0u, &net->qSlice, b);
        b += net->qSlice.size.nBytes;
        loader.loadRowMatmulSlices("block_matmul_k", layerIndex, 0u, &net->kSlice, b);
        b += net->kSlice.size.nBytes;
        loader.loadRowMatmulSlices("block_matmul_v", layerIndex, 0u, &net->vSlice, b);
        b += net->vSlice.size.nBytes;
        loader.loadColMatmulSlices("block_matmul_wo", layerIndex, 0u, &net->woSlice, b);
        b += net->woSlice.size.nBytes;

        if (net->header->nExperts > 0u) {
            loader.loadAll("block_moe_gate", layerIndex, net->moeGateSize.nBytes, b);
            b += net->moeGateSize.nBytes;
            
            for (NnUint expertIndex = 0u; expertIndex < net->header->nExperts; expertIndex++) {
                loader.loadRowMatmulSlices("block_matmul_w1", layerIndex, expertIndex, &net->w1Slice, b);
                b += net->w1Slice.sliceSize.nBytes; // Per expert size
                loader.loadColMatmulSlices("block_matmul_w2", layerIndex, expertIndex, &net->w2Slice, b);
                b += net->w2Slice.sliceSize.nBytes;
                loader.loadRowMatmulSlices("block_matmul_w3", layerIndex, expertIndex, &net->w3Slice, b);
                b += net->w3Slice.sliceSize.nBytes;
            }
        } else {
            loader.loadRowMatmulSlices("block_matmul_w1", layerIndex, 0u, &net->w1Slice, b);
            b += net->w1Slice.size.nBytes;
            loader.loadColMatmulSlices("block_matmul_w2", layerIndex, 0u, &net->w2Slice, b);
            b += net->w2Slice.size.nBytes;
            loader.loadRowMatmulSlices("block_matmul_w3", layerIndex, 0u, &net->w3Slice, b);
            b += net->w3Slice.size.nBytes;
        }

        if (net->header->archType == LLM_QWEN3 || net->header->archType == LLM_QWEN3_MOE) {
            loader.loadAll("block_norm_q", layerIndex, net->qkRmsNormSize.nBytes, b);
            b += net->qkRmsNormSize.nBytes;
            loader.loadAll("block_norm_k", layerIndex, net->qkRmsNormSize.nBytes, b);
            b += net->qkRmsNormSize.nBytes;
        }

        loader.loadAll("block_norm_0", layerIndex, net->rmsNormSize.nBytes, b);
        b += net->rmsNormSize.nBytes;
        loader.loadAll("block_norm_1", layerIndex, net->rmsNormSize.nBytes, b);
        b += net->rmsNormSize.nBytes;

        if (timer.elapsedMiliseconds() > 10000)
            printf("[INFO] Loaded %u/%u\n", layerIndex + 1, net->header->nLayers);
    }

    loader.loadAll("final_norm", 0u, net->rmsNormSize.nBytes, b);
    b += net->rmsNormSize.nBytes;
    loader.loadRowMatmulSlices("final_matmul_logits", 0u, 0u, &net->wclsSlice, b);
    b += net->wclsSlice.size.nBytes;

    long long missingBytes = (long long)(b - data) - net->header->fileSize;
    if (missingBytes != 0u)
        throw std::runtime_error("Missing bytes in weight file: " + std::to_string(missingBytes));
    printf("[INFO] Weights loaded\n");
}
