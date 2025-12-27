#include "simple-llm.hpp"
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
// Helper Functions
// ==================================================================================

static const char *hiddenActToString(SimpleLlmHiddenAct act) {
    if (act == SIMPLE_HIDDEN_ACT_GELU) return "Gelu";
    if (act == SIMPLE_HIDDEN_ACT_SILU) return "Silu";
    throw std::runtime_error("Unsupported hidden act");
}

static const char *ropeTypeToString(NnRopeType type) {
    if (type == ROPE_LLAMA) return "Llama";
    if (type == ROPE_LLAMA3_1) return "Llama3.1";
    if (type == ROPE_FALCON) return "Falcon";
    throw std::runtime_error("Unsupported rope type");
}

static const char *archTypeToString(SimpleLlmArchType type) {
    if (type == SIMPLE_LLAMA) return "Llama";
    if (type == SIMPLE_QWEN3) return "Qwen3";
    if (type == SIMPLE_QWEN3_MOE) return "Qwen3 MoE";
    throw std::runtime_error("Unsupported architecture");
}

static float convertNormEpsilon(int value) {
    if (value == 5) return 1e-05f;
    if (value == 6) return 1e-06f;
    throw std::runtime_error("Unsupported norm epsilon");
}

// ==================================================================================
// Load & Print Header
// ==================================================================================

SimpleLlmHeader loadSimpleLlmHeader(const char *path, const unsigned int maxSeqLen, NnFloatType syncType) {
    SimpleLlmHeader header;
    std::memset(&header, 0, sizeof(SimpleLlmHeader));
    header.weightType = F_UNK;
    header.hiddenAct = SIMPLE_HIDDEN_ACT_SILU;
    header.ropeType = ROPE_LLAMA;
    header.ropeTheta = 10000.0f;
    header.ropeScalingFactor = 1.0f;
    header.normEpsilon = 1e-5f;
    header.moeHiddenDim = 0u;

    std::unique_ptr<FILE, int(*)(FILE *)> fdPtr(fopen(path, "rb"), fclose);
    FILE *fd = fdPtr.get();
    if (fd == NULL)
        throw std::runtime_error(std::string("Cannot open model file (") + path + std::string("): ") + std::strerror(errno));

    int magic;
    if (fread(&magic, sizeof(int), 1, fd) != 1)
        throw std::runtime_error("Cannot read magic value");

    if (magic == 0xABCD00 || magic == 0xABCD01)
        throw std::runtime_error("Old model format is not supported");
    if (magic != 0xA00ABCD)
        throw std::runtime_error("Unsupported magic number");

    if (fread(&header.headerSize, sizeof(int), 1, fd) != 1)
        throw std::runtime_error("Cannot read header size");

    std::vector<int> bufferPtr(header.headerSize);
    int *buffer = &bufferPtr[0];
    if (fread(buffer, header.headerSize, 1, fd) != 1)
        throw std::runtime_error("Cannot read header values");

    int nKv = (header.headerSize - 2 * sizeof(int)) / sizeof(int);

    for (int i = 0; i < nKv; i += 2) {
        int key = buffer[i];
        int value = buffer[i + 1];
        if (key == SIMPLE_VERSION) header.version = value;
        else if (key == SIMPLE_ARCH_TYPE) header.archType = (SimpleLlmArchType)value;
        else if (key == SIMPLE_DIM) header.dim = value;
        else if (key == SIMPLE_HIDDEN_DIM) header.hiddenDim = value;
        else if (key == SIMPLE_N_LAYERS) header.nLayers = value;
        else if (key == SIMPLE_N_HEADS) header.nHeads = value;
        else if (key == SIMPLE_N_KV_HEADS) header.nKvHeads = value;
        else if (key == SIMPLE_N_EXPERTS) header.nExperts = value;
        else if (key == SIMPLE_N_ACTIVE_EXPERTS) header.nActiveExperts = value;
        else if (key == SIMPLE_VOCAB_SIZE) header.vocabSize = value;
        else if (key == SIMPLE_SEQ_LEN) header.seqLen = value;
        else if (key == SIMPLE_HIDDEN_ACT) header.hiddenAct = (SimpleLlmHiddenAct)value;
        else if (key == SIMPLE_ROPE_THETA) header.ropeTheta = (float)value;
        else if (key == SIMPLE_WEIGHT_FLOAT_TYPE) header.weightType = (NnFloatType)value;
        else if (key == SIMPLE_ROPE_SCALING_FACTOR) header.ropeScalingFactor = (float)value;
        else if (key == SIMPLE_ROPE_SCALING_LOW_FREQ_FACTOR) header.ropeScalingLowFreqFactor = (float)value;
        else if (key == SIMPLE_ROPE_SCALING_HIGH_FREQ_FACTORY) header.ropeScalingHighFreqFactory = (float)value;
        else if (key == SIMPLE_ROPE_SCALING_ORIG_MAX_SEQ_LEN) header.ropeScalingOrigMaxSeqLen = value;
        else if (key == SIMPLE_ROPE_TYPE) header.ropeType = (NnRopeType)value;
        else if (key == SIMPLE_HEAD_DIM) header.headDim = value;
        else if (key == SIMPLE_NORM_EPSILON) header.normEpsilon = convertNormEpsilon(value);
        else if (key == SIMPLE_MOE_HIDDEN_DIM) header.moeHiddenDim = value;
        else throw std::runtime_error("Unsupported header key");
    }

    if (header.weightType == F_UNK)
        throw std::runtime_error("Model does not specify weight type");

    header.origSeqLen = header.seqLen;
    if (maxSeqLen > 0 && header.seqLen > maxSeqLen)
        header.seqLen = maxSeqLen;

    if (header.headDim == 0)
        header.headDim = header.dim / header.nHeads;
    header.qDim = header.headDim * header.nHeads;
    header.kvDim = header.headDim * header.nKvHeads;
    header.syncType = syncType;
    header.fileSize = (NnSize)seekToEnd(fd);

    if (header.archType == SIMPLE_QWEN3 || header.archType == SIMPLE_QWEN3_MOE)
        header.ropeType = ROPE_FALCON;
    return header;
}

void printSimpleLlmHeader(SimpleLlmHeader *header) {
    printf("💡 Arch: %s\n", archTypeToString(header->archType));
    printf("💡 HiddenAct: %s\n", hiddenActToString(header->hiddenAct));
    printf("💡 Dim: %u\n", header->dim);
    printf("💡 HeadDim: %u\n", header->headDim);
    printf("💡 QDim: %u\n", header->qDim);
    printf("💡 KvDim: %u\n", header->kvDim);
    printf("💡 HiddenDim: %u\n", header->hiddenDim);
    printf("💡 VocabSize: %u\n", header->vocabSize);
    printf("💡 nLayers: %u\n", header->nLayers);
    printf("💡 nHeads: %u\n", header->nHeads);
    printf("💡 nKvHeads: %u\n", header->nKvHeads);
    if (header->seqLen != header->origSeqLen) {
        printf("💡 OrigSeqLen: %u\n", header->origSeqLen);
    }
    if (header->nExperts > 0) {
        printf("💡 nExperts: %u\n", header->nExperts);
        printf("💡 nActiveExperts: %u\n", header->nActiveExperts);
        printf("💡 MoeHiddenDim: %u\n", header->moeHiddenDim);
    }
    printf("💡 SeqLen: %u\n", header->seqLen);
    printf("💡 NormEpsilon: %f\n", header->normEpsilon);
    printf("💡 RopeType: %s\n", ropeTypeToString(header->ropeType));
    printf("💡 RopeTheta: %.0f\n", header->ropeTheta);
    if (header->ropeType == ROPE_LLAMA3_1) {
        printf("💡 RopeScaling: f=%.1f, l=%.1f, h=%.1f, o=%d\n",
            header->ropeScalingFactor,
            header->ropeScalingLowFreqFactor,
            header->ropeScalingHighFreqFactory,
            header->ropeScalingOrigMaxSeqLen);
    }
}

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

void releaseSimpleLlmNet(SimpleLlmNet *net) {
    releaseNodeConfig(&net->nodeConfig);
    releaseNetConfig(&net->netConfig);
}

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

void loadSimpleLlmNetWeight(const char *path, SimpleLlmNet *net, NnExecutor *executor) {
    MmapFile file;
    openMmapFile(&file, path, net->header->fileSize);

    // Simple execution assumes 1 node for now
    // assert(net->netConfig.nNodes == 1u);

#if DEBUG_USE_MMAP_FOR_WEIGHTS
    // pass
#else
    std::unique_ptr<MmapFile, void(*)(MmapFile *)> fdPtr(&file, closeMmapFile);
    printf("💿 Loading weights...\n");
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

        if (net->header->archType == SIMPLE_QWEN3 || net->header->archType == SIMPLE_QWEN3_MOE) {
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
            printf("💿 Loaded %u/%u\n", layerIndex + 1, net->header->nLayers);
    }

    loader.loadAll("final_norm", 0u, net->rmsNormSize.nBytes, b);
    b += net->rmsNormSize.nBytes;
    loader.loadRowMatmulSlices("final_matmul_logits", 0u, 0u, &net->wclsSlice, b);
    b += net->wclsSlice.size.nBytes;

    long long missingBytes = (long long)(b - data) - net->header->fileSize;
    if (missingBytes != 0u)
        throw std::runtime_error("Missing bytes in weight file: " + std::to_string(missingBytes));
    printf("💿 Weights loaded\n");
}

// ==================================================================================
// SimpleLlmInference Implementation
// ==================================================================================

SimpleLlmInference::SimpleLlmInference(SimpleLlmNet *net, NnNetExecution *execution, NnExecutor *executor) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[net->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->execution = execution;
    this->executor = executor;
}

void SimpleLlmInference::setBatchSize(NnUint batchSize) {
    execution->setBatchSize(batchSize);
}

void SimpleLlmInference::setPosition(NnUint position) {
    // Basic safety checks
    // assert(position >= 0); // Unsigned is always >= 0
    if (position + execution->batchSize - 1 >= header->seqLen) {
        throw std::runtime_error("Position exceeds sequence length");
    }

    // Set position for each item in batch
    for (NnUint i = 0; i < execution->batchSize; i++)
        positionPipe[i] = (float)(position + i);
}

void SimpleLlmInference::setToken(NnUint batchIndex, NnUint token) {
    if (batchIndex >= execution->batchSize) {
         throw std::runtime_error("Batch index out of bounds");
    }
    tokenPipe[batchIndex] = (float)token;
}

void SimpleLlmInference::forward() {
    executor->forward();
}
