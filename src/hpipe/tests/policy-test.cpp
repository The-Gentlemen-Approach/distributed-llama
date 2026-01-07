#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>
#include "hpipe/core/policies.hpp"
#include "common/llm-types.hpp"


void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(80, '=') << "\n";
}

void printDeviceProfiles(const std::vector<DeviceProfile>& profiles) {
    std::cout << "\nDevice Profiles:\n";
    for (size_t i = 0; i < profiles.size(); i++) {
        std::cout << "  Device " << i << ": "
                  << profiles[i].getTflops() << " TFLOPS, "
                  << profiles[i].getBandwidth() << " GB/s, "
                  << profiles[i].getLatency() * 1e6 << " us latency\n";
    }
}

void printSegmentRanges(const std::vector<SegmentRange>& ranges) {
    std::cout << "\nSegment Assignment:\n";
    for (size_t i = 0; i < ranges.size(); i++) {
        std::cout << "  Worker " << i << ": segments ["
                  << ranges[i].start << ", " << ranges[i].end << "] "
                  << "(count: " << (ranges[i].end - ranges[i].start + 1) << ")\n";
    }
}

void printSlicingResult(const std::vector<ChunkTask>& tasks) {
    std::cout << "\nSequence Slicing Result:\n";
    std::cout << "  Total slices: " << tasks.size() << "\n";
    std::cout << "  Slice sizes: ";
    for (const auto& task : tasks) {
        std::cout << task.tokens.size() << " ";
    }
    std::cout << "\n";
}


std::vector<DeviceProfile> createDeviceProfiles(const std::string& profile) {
    std::vector<DeviceProfile> profiles;

    if (profile == "homogeneous") {
        // 모든 디바이스가 동일
        for (int i = 0; i < 4; i++) {
            profiles.push_back(DeviceProfile(100.0f, 50.0f, 1e-5f));
        }
    }
    else if (profile == "heterogeneous") {
        // 성능이 다양한 디바이스
        profiles.push_back(DeviceProfile(400.0f, 100.0f, 1e-5f)); // Fast GPU
        profiles.push_back(DeviceProfile(200.0f, 50.0f, 1e-5f));  // Mid GPU
        profiles.push_back(DeviceProfile(100.0f, 25.0f, 2e-5f));  // Slow GPU
        profiles.push_back(DeviceProfile(50.0f, 10.0f, 5e-5f));   // Very slow
    }
    else if (profile == "network-bottleneck") {
        // 계산은 빠르지만 통신이 느림
        for (int i = 0; i < 4; i++) {
            profiles.push_back(DeviceProfile(400.0f, 1.0f, 1e-2f)); // Very high latency (10ms)
        }
    }
    else {
        // Default: homogeneous
        for (int i = 0; i < 4; i++) {
            profiles.push_back(DeviceProfile(100.0f, 50.0f, 1e-5f));
        }
    }

    return profiles;
}

void testWorkloadPartitioning(const LlmHeader& header, const std::vector<DeviceProfile>& profiles, int nWorkers) {
    printHeader("Test: Workload Partitioning");

    std::cout << "\nModel Configuration:\n";
    std::cout << "  Layers: " << header.nLayers << "\n";
    std::cout << "  Hidden Dim: " << header.hiddenDim << "\n";
    std::cout << "  Vocab Size: " << header.vocabSize << "\n";
    std::cout << "  Sequence Length: " << header.seqLen << "\n";
    std::cout << "  Workers: " << nWorkers << "\n";

    printDeviceProfiles(profiles);

    // Test OptimalWorkloadPartitioningPolicy
    std::cout << "\n--- Optimal Workload Partitioning ---\n";
    OptimalWorkloadPartitioningPolicy optimalPolicy(profiles);
    auto optimalRanges = optimalPolicy.assignSegments(header, nWorkers);
    printSegmentRanges(optimalRanges);

    // Test UniformPartitioningPolicy
    std::cout << "\n--- Uniform Partitioning ---\n";
    UniformPartitioningPolicy uniformPolicy;
    auto uniformRanges = uniformPolicy.assignSegments(header, nWorkers);
    printSegmentRanges(uniformRanges);

    // Compare computation time estimates
    std::cout << "\nEstimated Execution Times:\n";
    for (size_t i = 0; i < optimalRanges.size(); i++) {
        double optimalTime = 0.0;
        for (int k = optimalRanges[i].start; k <= optimalRanges[i].end; k++) {
            optimalTime += profiles[i].computationTime(header, k);
        }
        optimalTime += profiles[i].communicationTime(header);

        std::cout << "  Worker " << i << " (Optimal): "
                  << std::fixed << std::setprecision(6) << optimalTime << " s\n";
    }
}

void testSequenceSlicing(const LlmHeader& header, const std::vector<DeviceProfile>& profiles,
                         const std::vector<SegmentRange>& ranges, int promptLength) {
    printHeader("Test: Sequence Slicing");

    std::cout << "\nPrompt Length: " << promptLength << " tokens\n";
    std::cout << "Workers: " << profiles.size() << "\n";

    // Create dummy prompt tokens
    std::vector<int> promptTokens(promptLength);
    for (int i = 0; i < promptLength; i++) {
        promptTokens[i] = i % header.vocabSize;
    }

    // Test OptimalSequenceSlicingPolicy
    std::cout << "\n--- Optimal Sequence Slicing ---\n";
    OptimalSequenceSlicingPolicy optimalPolicy(header, profiles, ranges);
    auto optimalSlices = optimalPolicy.schedule(promptTokens, header.seqLen);
    printSlicingResult(optimalSlices);

    // Test FixedChunkScheduler
    std::cout << "\n--- Fixed Chunk Scheduling (128) ---\n";
    FixedChunkScheduler fixedPolicy(128);
    auto fixedSlices = fixedPolicy.schedule(promptTokens, header.seqLen);
    printSlicingResult(fixedSlices);

    // Analyze optimal slicing pattern
    std::cout << "\nOptimal Slicing Analysis:\n";
    int totalHistory = 0;
    for (size_t i = 0; i < optimalSlices.size() && i < 10; i++) {
        int sliceSize = optimalSlices[i].tokens.size();
        double maxStageTime = 0.0;
        int bottleneckWorker = 0;

        std::cout << "  Slice " << i << ": size=" << sliceSize << ", history=" << totalHistory << "\n";

        for (size_t w = 0; w < profiles.size(); w++) {
            double compTime = 0.0;
            for (int k = ranges[w].start; k <= ranges[w].end; k++) {
                compTime += profiles[w].computationTime(header, k, sliceSize, totalHistory);
            }
            double commTime = profiles[w].communicationTime(header, sliceSize);
            double stageTime = compTime + commTime;

            std::cout << "    Worker " << w << ": comp=" << std::fixed << std::setprecision(6)
                      << compTime << "s, comm=" << commTime << "s, total=" << stageTime << "s\n";

            if (stageTime > maxStageTime) {
                maxStageTime = stageTime;
                bottleneckWorker = w;
            }
        }

        std::cout << "    -> Bottleneck: Worker " << bottleneckWorker
                  << " (" << std::fixed << std::setprecision(6) << maxStageTime << "s)\n\n";

        totalHistory += sliceSize;
    }

    if (optimalSlices.size() > 10) {
        std::cout << "  ... (" << (optimalSlices.size() - 10) << " more slices)\n";
    }
}

void printUsage(const char* program) {
    std::cout << "Usage: " << program << " --model <path> [options]\n";
    std::cout << "\nRequired:\n";
    std::cout << "  --model <path>          Path to model file (.m)\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --seqlen <n>            Sequence length override (default: from model)\n";
    std::cout << "  --workers <n>           Number of workers (default: 4)\n";
    std::cout << "  --profile <type>        Device profile: homogeneous, heterogeneous, network-bottleneck (default: homogeneous)\n";
    std::cout << "  --prompt <n>            Prompt length for slicing test (default: 512)\n";
    std::cout << "  --test <type>           Test type: partitioning, slicing, all (default: all)\n";
    std::cout << "\nDevice Profiles:\n";
    std::cout << "  homogeneous:       All devices identical (100 TFLOPS, 50 GB/s, 10us latency)\n";
    std::cout << "  heterogeneous:     Mixed performance devices\n";
    std::cout << "  network-bottleneck: Fast compute, slow network (400 TFLOPS, 1 GB/s, 100us latency)\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program << " --model models/llama7b/model.m --workers 4 --profile heterogeneous\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    // Default parameters
    const char* modelPath = nullptr;
    int maxSeqLen = 0;
    int nWorkers = 4;
    int promptLength = 512;
    std::string deviceProfile = "homogeneous";
    std::string testType = "all";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (strcmp(argv[i], "--seqlen") == 0 && i + 1 < argc) {
            maxSeqLen = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--workers") == 0 && i + 1 < argc) {
            nWorkers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            deviceProfile = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            promptLength = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            testType = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    // Validate required arguments
    if (!modelPath) {
        std::cerr << "Error: --model argument is required\n\n";
        printUsage(argv[0]);
        return 1;
    }

    // Load model header from actual model file
    std::cout << "Loading model from: " << modelPath << "\n";
    LlmHeader header = loadLlmHeader(modelPath, maxSeqLen, F_Q80);
    std::cout << "Model loaded successfully\n";

    // Create device profiles
    auto profiles = createDeviceProfiles(deviceProfile);
    if (profiles.size() < (size_t)nWorkers) {
        profiles.resize(nWorkers, profiles[0]);
    } else if (profiles.size() > (size_t)nWorkers) {
        profiles.resize(nWorkers);
    }

    // Run tests
    if (testType == "partitioning" || testType == "all") {
        testWorkloadPartitioning(header, profiles, nWorkers);
    }

    if (testType == "slicing" || testType == "all") {
        // First get segment ranges for slicing test
        OptimalWorkloadPartitioningPolicy policy(profiles);
        auto ranges = policy.assignSegments(header, nWorkers);
        testSequenceSlicing(header, profiles, ranges, promptLength);
    }

    return 0;
}
