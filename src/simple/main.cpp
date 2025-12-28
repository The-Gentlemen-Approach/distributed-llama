#include <cstring>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <ctime>

// Removed app.hpp dependency
#include "nn/nn-core.hpp"
#include "nn/nn-config-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-cpu-ops.hpp"
// Removed nn-network.hpp dependency
#include "nn/nn-executor.hpp"
#include "common/llm-types.hpp"
#include "common/llm-builder.hpp"
#include "common/tokenizer.hpp"
#include "simple/inference.hpp"
#include "simple/network-builder.hpp"
#include "simple/weight-loader.hpp"

// ==================================================================================
// 0. Simplified AppCliArgs (Copied & Simplified from src/app.hpp)
// ==================================================================================

// Helper for parsing float types
static NnFloatType parseFloatType(char *val) {
    if (std::strcmp(val, "f32") == 0) return F_32;
    if (std::strcmp(val, "f16") == 0) return F_16;
    if (std::strcmp(val, "q40") == 0) return F_Q40;
    if (std::strcmp(val, "q80") == 0) return F_Q80;
    throw std::runtime_error("Invalid float type: " + std::string(val));
}

static ChatTemplateType parseChatTemplateType(char *val) {
    if (std::strcmp(val, "llama2") == 0) return TEMPLATE_LLAMA2;
    if (std::strcmp(val, "llama3") == 0) return TEMPLATE_LLAMA3;
    if (std::strcmp(val, "deepSeek3") == 0) return TEMPLATE_DEEP_SEEK3;
    throw std::runtime_error("Invalid chat template type: " + std::string(val));
}

class AppCliArgs {
public:
    char *mode;
    NnUint nThreads;
    NnUint nBatches;
    bool info;
    bool help;

    char *modelPath;
    char *tokenizerPath;
    char *prompt;
    NnFloatType syncType;
    // Removed worker related args
    float temperature;
    float topp;
    NnUint steps;
    bool benchmark;
    unsigned long long seed;
    ChatTemplateType chatTemplateType;
    NnUint maxSeqLen;
    // Removed netTurbo
    int gpuIndex;
    int gpuSegmentFrom;
    int gpuSegmentTo;

    static AppCliArgs parse(int argc, char **argv, bool hasMode) {
        AppCliArgs args;
        args.info = true;
        args.help = false;
        args.mode = nullptr;
        args.nBatches = 32;
        args.nThreads = 1;
        args.modelPath = nullptr;
        args.tokenizerPath = nullptr;
        args.prompt = nullptr;
        args.syncType = F_32;
        args.temperature = 0.8f;
        args.topp = 0.9f;
        args.steps = 0;
        args.seed = (unsigned long long)time(nullptr);
        args.chatTemplateType = TEMPLATE_UNKNOWN;
        args.maxSeqLen = 0;
        args.gpuIndex = -1;
        args.gpuSegmentFrom = -1;
        args.gpuSegmentTo = -1;
        args.benchmark = false;

        int i = 1;
        if (hasMode && argc > 1) {
            args.mode = argv[1];
            i++;
        }
        
        for (int x = 0; x < argc; x++) {
            if ((std::strcmp(argv[x], "--usage") == 0) ||
                (std::strcmp(argv[x], "--help") == 0) ||
                (std::strcmp(argv[x], "-h") == 0)) {
                args.help = true;
                return args;
            }
        }

        for (; i + 1 < argc; i += 2) {
            char *name = argv[i];
            char *value = argv[i + 1];
            if (std::strcmp(name, "--model") == 0) {
                args.modelPath = value;
            } else if (std::strcmp(name, "--tokenizer") == 0) {
                args.tokenizerPath = value;
            } else if (std::strcmp(name, "--prompt") == 0) {
                args.prompt = value;
            } else if (std::strcmp(name, "--buffer-float-type") == 0) {
                args.syncType = parseFloatType(value);
            } else if (std::strcmp(name, "--nthreads") == 0) {
                args.nThreads = atoi(value);
            } else if (std::strcmp(name, "--steps") == 0) {
                args.steps = atoi(value);
            } else if (std::strcmp(name, "--temperature") == 0) {
                args.temperature = atof(value);
            } else if (std::strcmp(name, "--topp") == 0) {
                args.topp = atof(value);
            } else if (std::strcmp(name, "--seed") == 0) {
                args.seed = atoll(value);
            } else if (std::strcmp(name, "--chat-template") == 0) {
                args.chatTemplateType = parseChatTemplateType(value);
            } else if (std::strcmp(name, "--max-seq-len") == 0) {
                args.maxSeqLen = (unsigned int)atoi(value);
            } else if (std::strcmp(name, "--gpu-index") == 0) {
                args.gpuIndex = atoi(value);
            } else if (std::strcmp(name, "--gpu-segments") == 0) {
                char *separator = std::strstr(value, ":");
                if (separator == NULL)
                    throw std::runtime_error("GPU segments expected in the format <from>:<to>");
                args.gpuSegmentFrom = atoi(value);
                args.gpuSegmentTo = atoi(separator + 1);
            } else {
                // Ignore unknown options for simplified version or throw?
                // For now throw to be safe
                throw std::runtime_error("Unknown option: " + std::string(name));
            }
        }

        if (args.nThreads < 1)
            throw std::runtime_error("Number of threads must be at least 1");
        return args;
    }
};

// ==================================================================================
// 1. Inference Logic (from src/dllama.cpp)
// ==================================================================================

struct SimpleInferenceContext {
    AppCliArgs *args;
    SimpleLlmHeader *header;
    SimpleLlmInference *inference;
    Tokenizer *tokenizer;
    Sampler *sampler;
    NnExecutor *executor;
};

static void inference(SimpleInferenceContext *context) {
    if (context->args->prompt == nullptr)
        throw std::runtime_error("Prompt is required");
    if (context->args->steps == 0)
        throw std::runtime_error("Number of steps is required");

    std::vector<int> inputTokensVec(std::strlen(context->args->prompt) + 3);
    int *inputTokens = inputTokensVec.data();

    NnUint pos = 0;
    int nInputTokens;
    context->tokenizer->encode(context->args->prompt, inputTokens, &nInputTokens, false, false);

    if (nInputTokens > context->header->seqLen)
        throw std::runtime_error("The number of prompt tokens is greater than the sequence length");
    if (nInputTokens > context->args->steps)
        throw std::runtime_error("The number of prompt tokens is greater than the number of steps");

    NnUint evalTotalTime = 0;
    NnUint predTotalTime = 0;

    int token = inputTokens[pos];
    printf("%s\n", context->args->prompt);

    // Phase 1: Evaluation
    for (;;) {
        long remainingTokens = nInputTokens - 1 - (long)pos;
        if (remainingTokens <= 0)
            break;
        NnUint batchSize = remainingTokens < context->args->nBatches
            ? remainingTokens
            : context->args->nBatches;

        context->inference->setBatchSize(batchSize);
        context->inference->setPosition(pos);
        for (NnUint i = 0; i < batchSize; i++)
            context->inference->setToken(i, inputTokens[pos + i]);

        context->inference->forward();

        pos += batchSize;
        token = inputTokens[pos + 1];

        NnUint evalTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        printf("🔷️ Eval%5u ms | (%d tokens)\n", evalTime / 1000, batchSize);
        evalTotalTime += evalTime;
    }

    fflush(stdout);

    // Phase 2: Prediction
    context->inference->setBatchSize(1);
    context->tokenizer->resetDecoder();

    const NnUint maxPos = std::min(context->header->seqLen, context->args->steps);
    for (; pos < maxPos; pos++) {
        context->inference->setPosition(pos);
        context->inference->setToken(0, token);
        context->inference->forward();

        token = context->sampler->sample(context->inference->logitsPipe);
        char *piece = context->tokenizer->decode(token);

        NnUint predTime = context->executor->getTotalTime(STEP_EXECUTE_OP);
        printf("🔶 Pred%5u ms | %s\n", predTime / 1000, piece == nullptr ? "~" : piece);
        fflush(stdout);
        predTotalTime += predTime;
    }

    // Stats
    NnUint nEvalTokens = nInputTokens - 1;
    NnUint nPredTokens = pos - nEvalTokens;
    float evalTotalTimeMs = evalTotalTime / 1000.0;
    float predTotalTimeMs = predTotalTime / 1000.0;
    printf("\n");
    printf("Evaluation\n");
    printf("   nBatches: %d\n", context->args->nBatches);
    printf("    nTokens: %d\n", nEvalTokens);
    if (evalTotalTimeMs > 0)
        printf("   tokens/s: %3.2f (%3.2f ms/tok)\n", (nEvalTokens * 1000) / evalTotalTimeMs, evalTotalTimeMs / ((float) nEvalTokens));
    printf("Prediction\n");
    printf("    nTokens: %d\n", nPredTokens);
    if (predTotalTimeMs > 0)
        printf("   tokens/s: %3.2f (%3.2f ms/tok)\n", (nPredTokens * 1000) / predTotalTimeMs, predTotalTimeMs / ((float) nPredTokens));
}

// ==================================================================================
// 2. Simple Runner
// ==================================================================================

static std::vector<NnExecutorDevice> resolveDevices(AppCliArgs *args, NnNetConfig *netConfig, NnNodeConfig *nodeConfig, NnNetExecution *netExecution) {
    std::vector<NnExecutorDevice> devices;
    // GPU Logic removed/simplifed: Always CPU for simplicity, or can be added back if NnCpuDevice depends on it?
    // NnCpuDevice is in nn-cpu.hpp.
    // GPU support requires DLLAMA_VULKAN macro and nn-vulkan.hpp. 
    // Keeping it simple: CPU only.
    devices.push_back(NnExecutorDevice(new NnCpuDevice(netConfig, nodeConfig, netExecution), -1, -1));
    return devices;
}

void runSimpleApp(AppCliArgs *args) {
    // 1. Load Header
    SimpleLlmHeader header = loadSimpleLlmHeader(args->modelPath, args->maxSeqLen, args->syncType);

    if (header.weightType == F_Q40 && header.syncType == F_32) {
        printf("⚠️ Automatically switching buffer type to Q80 for Q40 model compatibility.\n");
        header.syncType = F_Q80;
    }

    // 2. Tokenizer
    Tokenizer tokenizer(args->tokenizerPath);
    if (args->info && tokenizer.vocabSize != header.vocabSize)
        printf("Warning: Tokenizer vocab size (%d) != Model vocab size (%d)\n", tokenizer.vocabSize, header.vocabSize);

    // 3. Sampler
    Sampler sampler(tokenizer.vocabSize, args->temperature, args->topp, args->seed);

    // 4. Build Network
    SimpleLlmNet net = buildSimpleLlmNet(&header, args->nBatches);
    std::unique_ptr<SimpleLlmNet, void(*)(SimpleLlmNet *)> netPtr(&net, releaseSimpleLlmNet);

    NnNodeConfig *rootNodeConfig = &net.nodeConfig;

    if (args->info) {
        printSimpleLlmHeader(&header);
    }

    // 5. Execution Context
    NnNetExecution execution(args->nThreads, &net.netConfig);
    
    // Fake Synchronizer (No Network)
    std::unique_ptr<NnNodeSynchronizer> synchronizer(new NnFakeNodeSynchronizer());

    // 6. Executor
    std::vector<NnExecutorDevice> devices = resolveDevices(args, &net.netConfig, rootNodeConfig, &execution);
    NnExecutor executor(&net.netConfig, rootNodeConfig, &devices, &execution, synchronizer.get(), args->benchmark);

    // 7. Load Weights (Locally)
    loadSimpleLlmNetWeight(args->modelPath, &net, &executor);

    // 8. Inference Object
    SimpleLlmInference rootInference(&net, &execution, &executor);

    // 9. Context
    SimpleInferenceContext context;
    context.args = args;
    context.header = &header;
    context.inference = &rootInference;
    context.sampler = &sampler;
    context.tokenizer = &tokenizer;
    context.executor = &executor;

    // 10. Run Inference
    inference(&context); 
}

// ==================================================================================
// 3. Main Entry
// ==================================================================================

int main(int argc, char **argv) {
    initQuants(); // Initialize Quantization Tables

    try {
        AppCliArgs args = AppCliArgs::parse(argc, argv, false);
        
        if (args.help || args.modelPath == nullptr) {
            printf("Usage: ./simple-dllama --model <path> --tokenizer <path> --prompt \"...\"\n");
            return 0;
        }

        args.benchmark = true;
        runSimpleApp(&args);

    } catch (const std::exception &e) {
        printf("🚨 Error: %s\n", e.what());
        return 1;
    }

    return 0;
}
