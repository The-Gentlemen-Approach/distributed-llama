/**
 * hpipe-root.cpp: H-Pipe Root Node Implementation
 *
 * Coordinates pipeline execution.
 * - Workload Distribution: Optimal Policy (Algorithm 1) applied.
 * - Sequence Slicing: Fixed Chunk Size (Original) maintained.
 */

#include "hpipe/network/root.hpp"
#include "hpipe/core/types.hpp"
#include "hpipe/core/policies.hpp" // OptimalWorkloadPartitioningPolicy가 포함됨
#include "hpipe/core/utils.hpp"
#include "common/llm-types.hpp"
#include "common/tokenizer.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <cmath>

struct RootArgs {
    char* modelPath;
    char* tokenizerPath;
    char* prompt;
    std::vector<std::string> workerAddrs;
    int nThreads;
    float temperature;
    float topp;
    int steps;
    unsigned long long seed;
    int chunkSize; // [유지] Sequence Slicing은 기존 방식대로 고정 크기 사용

    static RootArgs parse(int argc, char** argv) {
        RootArgs args;
        args.modelPath = nullptr;
        args.tokenizerPath = nullptr;
        args.prompt = nullptr;
        args.nThreads = 4;
        args.temperature = 0.8f;
        args.topp = 0.9f;
        args.steps = 256;
        args.seed = (unsigned long long)time(nullptr);
        args.chunkSize = 128; // Default chunk size

        int i = 1;
        while (i < argc) {
            char* name = argv[i];

            if (std::strcmp(name, "--model") == 0 && i + 1 < argc) {
                args.modelPath = argv[++i];
            } else if (std::strcmp(name, "--tokenizer") == 0 && i + 1 < argc) {
                args.tokenizerPath = argv[++i];
            } else if (std::strcmp(name, "--prompt") == 0 && i + 1 < argc) {
                args.prompt = argv[++i];
            } else if (std::strcmp(name, "--workers") == 0) {
                i++;
                while (i < argc && argv[i][0] != '-') {
                    args.workerAddrs.push_back(argv[i]);
                    i++;
                }
                i--;
            } else if (std::strcmp(name, "--nthreads") == 0 && i + 1 < argc) {
                args.nThreads = std::atoi(argv[++i]);
            } else if (std::strcmp(name, "--temperature") == 0 && i + 1 < argc) {
                args.temperature = std::atof(argv[++i]);
            } else if (std::strcmp(name, "--topp") == 0 && i + 1 < argc) {
                args.topp = std::atof(argv[++i]);
            } else if (std::strcmp(name, "--steps") == 0 && i + 1 < argc) {
                args.steps = std::atoi(argv[++i]);
            } else if (std::strcmp(name, "--seed") == 0 && i + 1 < argc) {
                args.seed = std::atoll(argv[++i]);
            } else if (std::strcmp(name, "--chunk-size") == 0 && i + 1 < argc) {
                args.chunkSize = std::atoi(argv[++i]);
            }
            i++;
        }

        if (!args.modelPath || !args.tokenizerPath || !args.prompt || args.workerAddrs.empty()) {
            throw std::runtime_error("Missing required arguments");
        }

        return args;
    }
};

void printUsage() {
    std::cout << "H-Pipe Root:\n";
    std::cout << "  ./hpipe-root --model <path> --tokenizer <path> --prompt <text>\n";
    std::cout << "               --workers <addr1> <addr2> ... [options]\n";
}

// [추가] Workload Distribution을 위한 가상 디바이스 프로필 생성
// 실제 환경에서는 네트워크 통신으로 받아오지만, 여기서는 시뮬레이션 값을 사용합니다.
std::vector<DeviceProfile> getMockDeviceProfiles(int nWorkers) {
    std::vector<DeviceProfile> profiles;
    for (int i = 0; i < nWorkers; i++) {
        // [시나리오] 짝수 워커는 고성능, 홀수 워커는 저성능으로 가정하여 부하 분산 효과 테스트
        if (i % 2 == 0) {
            profiles.push_back(DeviceProfile(300.0f, 80.0f)); // High-End
        } else {
            profiles.push_back(DeviceProfile(60.0f, 16.0f));  // Low-End
        }
    }
    return profiles;
}

void runRoot(const RootArgs& args) {
    LOG("🔷 H-Pipe Root starting...");

    try {
        // Load model header
        LOG("📂 Loading model header...");
        LlmHeader header = loadLlmHeader(args.modelPath, 0, F_32);

        if (header.weightType == F_Q40 && header.syncType == F_32) {
            LOG("⚠️  Automatically switching buffer type to Q80 for Q40 model compatibility.");
            header.syncType = F_Q80;
        }

        LOG("✓ Model: " << header.nLayers << " layers, dim=" << header.dim);

        // Load tokenizer
        LOG("📂 Loading tokenizer...");
        Tokenizer tokenizer(args.tokenizerPath);
        LOG("✓ Tokenizer loaded");

        // Tokenize prompt
        LOG("🔤 Tokenizing: \"" << args.prompt << "\"");
        std::vector<int> promptTokensVec(std::strlen(args.prompt) + 3);
        int* promptTokens = promptTokensVec.data();
        int nPromptTokens;
        tokenizer.encode(args.prompt, promptTokens, &nPromptTokens, false, false);
        LOG("✓ " << nPromptTokens << " tokens");

        // ----------------------------------------------------------------
        // [변경됨] Workload Partitioning: Uniform -> Optimal (Algorithm 1)
        // ----------------------------------------------------------------
        int nWorkers = args.workerAddrs.size();
        
        // 1. 디바이스 프로필 생성 (Algorithm 1 입력값)
        auto profiles = getMockDeviceProfiles(nWorkers);

        // 2. 최적 할당 정책 적용
        LOG("🧠 Calculating Optimal Workload Distribution (Algorithm 1)...");
        OptimalWorkloadPartitioningPolicy workloadPolicy(profiles);
        std::vector<SegmentRange> segmentRanges = workloadPolicy.assignSegments(header, nWorkers);

        LOG("📊 Segment partitioning (nWorkers=" << nWorkers << "):");
        for (size_t i = 0; i < segmentRanges.size(); i++) {
            int nSegs = segmentRanges[i].end - segmentRanges[i].start + 1;
            LOG("  Worker " << i << ": [" << segmentRanges[i].start
                      << ", " << segmentRanges[i].end << "] -> " << nSegs << " segments");
        }
        // ----------------------------------------------------------------

        // Connect to workers
        LOG("🔗 Connecting to workers...");
        initSockets();

        std::vector<char*> hosts(nWorkers);
        std::vector<int> ports(nWorkers);

        for (int i = 0; i < nWorkers; i++) {
            size_t colonPos = args.workerAddrs[i].find(':');
            std::string host = args.workerAddrs[i].substr(0, colonPos);
            std::string port = args.workerAddrs[i].substr(colonPos + 1);
            hosts[i] = new char[256];
            std::memset(hosts[i], 0, 256);
            std::strncpy(hosts[i], host.c_str(), 255);
            ports[i] = std::stoi(port);
        }

        auto network = HPipeRootNetwork::connect(nWorkers, hosts.data(), ports.data());
        LOG("✓ Connected");

        // Send configs
        LOG("📤 Sending configs...");
        std::vector<HPipeConfig> configs;

        for (int i = 0; i < nWorkers; i++) {
            HPipeConfig config;
            std::memset(&config, 0, sizeof(HPipeConfig));

            config.worker_id = i;
            config.total_workers = nWorkers;
            
            // [중요] 계산된 최적 범위(segmentRanges)를 전송
            config.segment_range = segmentRanges[i];

            if (i < nWorkers - 1) {
                size_t colonPos = args.workerAddrs[i + 1].find(':');
                std::string nextHost = args.workerAddrs[i + 1].substr(0, colonPos);
                std::string nextPort = args.workerAddrs[i + 1].substr(colonPos + 1);
                strncpy(config.next_host, nextHost.c_str(), sizeof(config.next_host) - 1);
                config.next_port = std::stoi(nextPort);
            }

            strncpy(config.model_path, args.modelPath, sizeof(config.model_path) - 1);
            config.model_header = header;
            configs.push_back(config);
        }

        network->broadcastConfig(configs);
        LOG("✓ Config sent");

        // Initialize sampler
        Sampler sampler(header.vocabSize, args.temperature, args.topp, args.seed);

        // ----------------------------------------------------------------
        // Prefill phase: [유지] Fixed Chunk Scheduler (Original Code)
        // ----------------------------------------------------------------
        LOG("🚀 Prefill phase (" << nPromptTokens << " tokens):");

        tokenizer.resetDecoder();

        // 기존 로직 유지: 사용자가 입력한 chunkSize 사용
        FixedChunkScheduler scheduler(args.chunkSize);
        std::vector<int> promptVec(promptTokens, promptTokens + nPromptTokens);
        std::vector<ChunkTask> chunks = scheduler.schedule(promptVec, header.seqLen);

        LOG("  " << chunks.size() << " chunks (Fixed Size: " << args.chunkSize << ")");

        std::vector<int> generatedTokens;
        int currentPos = 0;

        // Pipeline parallelization logic (Original)
        
        // Phase 1: Send all chunks to first worker
        std::vector<std::vector<float>> chunkTokenData(chunks.size());
        for (size_t chunkIdx = 0; chunkIdx < chunks.size(); chunkIdx++) {
            const auto& chunk = chunks[chunkIdx];
            HPipeHeader chunkHeader;
            chunkHeader.magic = HPIPE_HEADER_MAGIC;
            chunkHeader.seq_id = chunk.seq_id;
            chunkHeader.step = currentPos;
            chunkHeader.n_tokens = chunk.tokens.size();
            chunkHeader.data_size = chunk.tokens.size() * sizeof(float);

            chunkTokenData[chunkIdx].resize(chunk.tokens.size());
            for (size_t i = 0; i < chunk.tokens.size(); i++) {
                chunkTokenData[chunkIdx][i] = static_cast<float>(chunk.tokens[i]);
            }

            network->sendToFirstWorker(chunkHeader, chunkTokenData[chunkIdx].data(), chunkHeader.data_size);

            currentPos += chunk.tokens.size();
        }

        // Phase 2: Receive all results from last worker
        currentPos = 0;
        for (size_t chunkIdx = 0; chunkIdx < chunks.size(); chunkIdx++) {
            const auto& chunk = chunks[chunkIdx];

            std::vector<float> logitsData(header.vocabSize * chunk.tokens.size());
            HPipeHeader resultHeader;
            network->recvFromLastWorker(&resultHeader, logitsData.data(),
                                       header.vocabSize * chunk.tokens.size() * sizeof(float));

            if (chunkIdx == chunks.size() - 1) {
                float* lastLogits = logitsData.data() + header.vocabSize * (chunk.tokens.size() - 1);

                int nextToken = sampler.sample(lastLogits);
                generatedTokens.push_back(nextToken);

                char* piece = tokenizer.decode(nextToken);
                std::cout << (piece ? piece : "~") << std::flush;
            }

            currentPos += chunk.tokens.size();
        }

        LOG("\n\n🔄 Decoding phase:");

        // Decoding phase (Original)
        tokenizer.resetDecoder();
        for (int step = 0; step < args.steps && currentPos < (int)header.seqLen; step++) {
            int lastToken = generatedTokens.back();

            HPipeHeader chunkHeader;
            chunkHeader.magic = HPIPE_HEADER_MAGIC;
            chunkHeader.seq_id = chunks.size() + step;
            chunkHeader.step = currentPos;
            chunkHeader.n_tokens = 1;
            chunkHeader.data_size = sizeof(float);

            float tokenData = static_cast<float>(lastToken);
            network->sendToFirstWorker(chunkHeader, &tokenData, chunkHeader.data_size);

            std::vector<float> logitsData(header.vocabSize);
            HPipeHeader resultHeader;
            network->recvFromLastWorker(&resultHeader, logitsData.data(),
                                       header.vocabSize * sizeof(float));

            int nextToken = sampler.sample(logitsData.data());
            generatedTokens.push_back(nextToken);

            char* piece = tokenizer.decode(nextToken);
            std::cout << (piece ? piece : "~") << std::flush;

            currentPos++;

            if (tokenizer.isEos(nextToken)) {
                break;
            }
        }

        LOG("\n\n✅ Complete! Generated " << generatedTokens.size() << " tokens");

        cleanupSockets();
        for (int i = 0; i < nWorkers; i++) {
            delete[] hosts[i];
        }

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
        RootArgs args = RootArgs::parse(argc, argv);
        runRoot(args);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
