#include "llm-types.hpp"
#include "common/mmap.hpp"
#include <algorithm>
#include <cstring>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cerrno>
#include <cstdio>
#include <string>

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
