/**
 * hpipe-worker.cpp: H-Pipe Worker Node Implementation
 *
 * Executes assigned model segments in pipeline fashion.
 */

#include "hpipe/network/worker.hpp"
#include "hpipe/llm/inference.hpp"
#include "hpipe/llm/network-builder.hpp"
#include "hpipe/llm/weight-loader.hpp"
#include "hpipe/core/utils.hpp"
#include "common/llm-types.hpp"
#include "common/llm-builder.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-executor.hpp"
#include "nn/nn-quants.hpp"
#ifdef DLLAMA_VULKAN
#include "nn/nn-vulkan.hpp"
#endif
#include <iostream>
#include <vector>
#include <cstring>
#include <stdexcept>
#include <memory>

struct WorkerArgs {
    int port;
    int nThreads;
    int gpuIndex;
    int gpuSegmentFrom;
    int gpuSegmentTo;

    static WorkerArgs parse(int argc, char** argv) {
        WorkerArgs args;
        args.port = 9999;
        args.nThreads = 4;
        args.gpuIndex = -1;
        args.gpuSegmentFrom = -1;
        args.gpuSegmentTo = -1;

        for (int i = 1; i + 1 < argc; i += 2) {
            if (std::strcmp(argv[i], "--port") == 0) {
                args.port = std::atoi(argv[i + 1]);
            } else if (std::strcmp(argv[i], "--nthreads") == 0) {
                args.nThreads = std::atoi(argv[i + 1]);
            } else if (std::strcmp(argv[i], "--gpu-index") == 0) {
                args.gpuIndex = std::atoi(argv[i + 1]);
            } else if (std::strcmp(argv[i], "--gpu-segments") == 0) {
                char* separator = std::strstr(argv[i + 1], ":");
                if (separator == NULL)
                    throw std::runtime_error("GPU segments expected in the format <from>:<to>");
                args.gpuSegmentFrom = std::atoi(argv[i + 1]);
                args.gpuSegmentTo = std::atoi(separator + 1);
            }
        }
        return args;
    }
};

void printUsage() {
    std::cout << "H-Pipe Worker:\n";
    std::cout << "  ./hpipe-worker --port <port> [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --nthreads <n>           Number of CPU threads (default: 4)\n";
    std::cout << "  --gpu-index <n>          GPU device index (default: -1, CPU only)\n";
    std::cout << "  --gpu-segments <from>:<to>  GPU segment range (e.g., 0:10)\n";
}

void runWorker(const WorkerArgs& args) {
    LOG("🔶 H-Pipe Worker starting on port " << args.port << "...");

    try {
        initSockets();
        auto network = HPipeWorkerNetwork::serve(args.port);
        LOG("✓ Network initialized");

        // Receive config
        HPipeConfig config = network->recvConfigFromRoot();
        network->sendConfigAck();

        LOG("✓ Config received:");
        LOG("  Worker ID: " << config.worker_id);
        LOG("  Segments: [" << config.segment_range.start
                  << ", " << config.segment_range.end << "]");
        LOG("  First: " << network->getIsFirstWorker());
        LOG("  Last: " << network->getIsLastWorker());

        LlmHeader header = config.model_header;

        if (header.weightType == F_Q40 && header.syncType == F_32) {
            LOG("⚠️  Automatically switching buffer type to Q80 for Q40 model compatibility.");
            header.syncType = F_Q80;
        }

        // Build network for assigned segments
        const NnUint nBatches = 32;
        LlmNet net = buildHpipeLlmNet(
            &header, nBatches,
            config.segment_range.start,
            config.segment_range.end
        );
        std::unique_ptr<LlmNet, void(*)(LlmNet*)> netPtr(&net, releaseLlmNet);

        LOG("✓ Network built");

        // Create execution context
        NnNetExecution execution(args.nThreads, &net.netConfig);
        std::unique_ptr<NnNodeSynchronizer> synchronizer(new NnFakeNodeSynchronizer());

        // Resolve devices (GPU + CPU)
        std::vector<NnExecutorDevice> devices;

        if (args.gpuIndex >= 0) {
#ifdef DLLAMA_VULKAN
            devices.push_back(NnExecutorDevice(
                new NnVulkanDevice(args.gpuIndex, &net.netConfig, &net.nodeConfig, &execution),
                args.gpuSegmentFrom,
                args.gpuSegmentTo
            ));
            LOG("✓ GPU device added: index=" << args.gpuIndex
                << ", segments=[" << args.gpuSegmentFrom << ":" << args.gpuSegmentTo << "]");
#else
            throw std::runtime_error("This build does not support GPU. Rebuild with DLLAMA_VULKAN=1");
#endif
        }

        if (args.gpuIndex < 0 || (args.gpuSegmentFrom >= 0 && args.gpuSegmentTo >= 0)) {
            devices.push_back(NnExecutorDevice(
                new NnCpuDevice(&net.netConfig, &net.nodeConfig, &execution), -1, -1
            ));
            LOG("✓ CPU device added");
        }

        NnExecutor executor(&net.netConfig, &net.nodeConfig, &devices, &execution,
                           synchronizer.get(), false);

        LOG("✓ Executor created");

        // Load weights for assigned segments
        loadHpipeLlmNetWeight(config.model_path, &net, &executor,
                             config.segment_range.start, config.segment_range.end);
        LOG("✓ Weights loaded");

        // Create inference object
        HPipeLlmInference inference(&net, &execution, &executor,
                                   config.segment_range.start,
                                   config.segment_range.end);

        LOG("\n🚀 Worker ready to process chunks\n");

        // Pipeline loop
        int chunksProcessed = 0;
        
        // Calculate max buffer size (F32 is the largest possible)
        const size_t MAX_DATA_SIZE = header.dim * nBatches * sizeof(float);
        std::vector<uint8_t> recvBuffer(MAX_DATA_SIZE);

        // Determine input/output types based on segments
        // IMPORTANT: Always use F32 for inter-worker activation transfer to prevent
        // quantization loss accumulation. Only model weights use Q40/Q80.
        size_t inputBytesPerToken = getBytes(F_32, header.dim);
        size_t outputBytesPerToken = getBytes(F_32, header.dim);

        const size_t bytesPerLogit = sizeof(float); // Logits are always F32

        while (true) {
            HPipeHeader chunkHeader;

            try {
                network->recvFromPrev(&chunkHeader, recvBuffer.data(), MAX_DATA_SIZE);
            } catch (const std::exception& e) {
                LOG("📡 Pipeline ended");
                break;
            }

            LOG("📥 Chunk " << chunkHeader.seq_id
                      << " (step=" << chunkHeader.step
                      << ", tokens=" << chunkHeader.n_tokens << ")"
                      << " Worker " << config.worker_id);

            // Set batch size and position
            inference.setBatchSize(chunkHeader.n_tokens);
            inference.setPosition(chunkHeader.step);

            // Copy input
            if (!network->getIsFirstWorker() && inference.inputPipe != nullptr) {
                // Receive data into input pipe
                size_t inputSize = chunkHeader.n_tokens * inputBytesPerToken;
                std::memcpy(inference.inputPipe, recvBuffer.data(), inputSize);
            } else if (network->getIsFirstWorker()) {
                // First worker receives tokens (float -> uint cast)
                float* floatBuffer = reinterpret_cast<float*>(recvBuffer.data());
                for (NnUint i = 0; i < chunkHeader.n_tokens; i++) {
                    inference.setToken(i, static_cast<NnUint>(floatBuffer[i]));
                }
            }

            // Execute forward pass
            inference.forward();

            // Debug: Log output activations for first token of first chunk
            if (chunkHeader.seq_id == 0 && chunkHeader.n_tokens > 0) {
                if (network->getIsLastWorker()) {
                    LOG("🔍 [Worker " << config.worker_id << "] Chunk 0 logits[0:5]: "
                        << inference.logitsPipe[0] << " "
                        << inference.logitsPipe[1] << " "
                        << inference.logitsPipe[2] << " "
                        << inference.logitsPipe[3] << " "
                        << inference.logitsPipe[4]);
                } else if (inference.outputPipe != nullptr) {
                    LOG("🔍 [Worker " << config.worker_id << "] Chunk 0 output[0:5]: "
                        << inference.outputPipe[0] << " "
                        << inference.outputPipe[1] << " "
                        << inference.outputPipe[2] << " "
                        << inference.outputPipe[3] << " "
                        << inference.outputPipe[4]);
                }
            }

            // Prepare output
            size_t outputSize;
            void* outputData;

            if (network->getIsLastWorker()) {
                // Last worker outputs logits (always F32)
                outputSize = header.vocabSize * chunkHeader.n_tokens * bytesPerLogit;
                outputData = inference.logitsPipe;
            } else {
                // Intermediate workers output activations
                outputSize = chunkHeader.n_tokens * outputBytesPerToken;
                outputData = inference.outputPipe;
            }

            chunkHeader.data_size = outputSize;

            // Send to next
            network->sendToNext(chunkHeader, outputData, outputSize);

            LOG("📤 Chunk " << chunkHeader.seq_id << " sent");
            chunksProcessed++;
        }

        LOG("\n✅ Processed " << chunksProcessed << " chunks");
        cleanupSockets();

    } catch (const std::exception& e) {
        std::cerr << getTimestamp() << " ❌ Error: " << e.what() << "\n";
        cleanupSockets();
        throw;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    try {
        initQuants();
        WorkerArgs args = WorkerArgs::parse(argc, argv);
        runWorker(args);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
