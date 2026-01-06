/**
 * hpipe-root.cpp: H-Pipe Root Node Implementation
 *
 * Coordinates pipeline execution.
 * - Workload Distribution: Optimal Policy (Algorithm 1) applied.
 * - Sequence Slicing: Optimal Policy (Algorithm 2 + Latency Awareness) applied.
 */

#include "hpipe/network/root.hpp"
#include "hpipe/core/types.hpp"
#include "hpipe/core/policies.hpp" // OptimalWorkloadPartitioningPolicy & OptimalSequenceSlicingPolicy
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
    int chunkSize; 
    int maxSeqLen; 

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
        args.chunkSize = 128; 
        args.maxSeqLen = 0; 

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
            } else if ((std::strcmp(name, "--max-seq-len") == 0 || std::strcmp(name, "--seq-len") == 0) && i + 1 < argc) {
                args.maxSeqLen = std::atoi(argv[++i]);
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
    std::cout << "\nOptions:\n";
    std::cout << "  --nthreads <n>      Number of threads (default: 4)\n";
    std::cout << "  --max-seq-len <n>   Maximum sequence length (default: model limit)\n";
    std::cout << "  --chunk-size <n>    Sequence chunk size (Ignored when using OptPolicy)\n";
    std::cout << "  --steps <n>         Number of steps to generate (default: 256)\n";
    std::cout << "  --temperature <f>   Sampling temperature (default: 0.8)\n";
    std::cout << "  --topp <f>          Sampling top-p (default: 0.9)\n";
    std::cout << "  --seed <n>          Random seed\n";
}

// [수정됨] Workload Distribution을 위한 가상 디바이스 프로필 생성
// Latency(고정 오버헤드) 값을 3번째 인자로 추가하여 전달합니다.
std::vector<DeviceProfile> getMockDeviceProfiles(int nWorkers) {
    std::vector<DeviceProfile> profiles;
    
    // 일반적인 이더넷 환경을 가정하여 200us (0.0002초)의 고정 오버헤드 설정
    // 이 값이 클수록 알고리즘은 더 큰 청크로 묶으려고 합니다.
    float latency = 2e-4f; 

    // 기존 설정값에 latency 인자 추가 (TFLOPS, Bandwidth, Latency)
    // nWorkers 수만큼 생성하도록 반복문으로 처리하거나, 필요한 만큼 push_back
    for(int i=0; i<nWorkers; ++i) {
        if (i == 0) profiles.push_back(DeviceProfile(100.0f, 100.0f, latency));
        else if (i == 1) profiles.push_back(DeviceProfile(100.0f, 100.0f, latency));
        else if (i == 2) profiles.push_back(DeviceProfile(100.0f, 100.0f, latency));
        else if (i == 3) profiles.push_back(DeviceProfile(100.0f, 10.0f, latency));
        else if (i == 4) profiles.push_back(DeviceProfile(400.0f, 10.0f, latency));
        else profiles.push_back(DeviceProfile(400.0f, 10000.0f, latency)); // Fallback
    }

    return profiles;
}

void runRoot(const RootArgs& args) {
    LOG("🔷 H-Pipe Root starting...");

    try {
        // Load model header
        LOG("📂 Loading model header...");
        LlmHeader header = loadLlmHeader(args.modelPath, args.maxSeqLen, F_Q80);

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
        // [Stage 1] Workload Partitioning: Optimal (Algorithm 1)
        // ----------------------------------------------------------------
        int nWorkers = args.workerAddrs.size();
        
        // 1. 디바이스 프로필 생성 (Latency 포함됨)
        auto profiles = getMockDeviceProfiles(nWorkers);

        // 2. 최적 할당 정책 적용
        LOG("🧠 Calculating Optimal Workload Partitioning (Algorithm 1)...");
        OptimalWorkloadPartitioningPolicy workloadPolicy(profiles);
        std::vector<SegmentRange> segmentRanges = workloadPolicy.assignSegments(header, nWorkers);

        LOG("📊 Segment Partitioning Result:");
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
            
            // 할당된 레이어 범위 전송
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
        // [Stage 2] Prefill phase: Optimal Sequence Slicing (Algorithm 2)
        // ----------------------------------------------------------------
        LOG("🚀 Prefill phase (" << nPromptTokens << " tokens):");

        tokenizer.resetDecoder();

        // [변경] OptimalSequenceSlicingPolicy 사용 (Latency 적용됨)
        LOG("🔪 Calculating Optimal Sequence Slicing (Algorithm 2)...");
        
        OptimalSequenceSlicingPolicy slicingPolicy(header, profiles, segmentRanges);
        std::vector<int> promptVec(promptTokens, promptTokens + nPromptTokens);
        
        // 동적 슬라이싱 수행
        std::vector<ChunkTask> chunks = slicingPolicy.schedule(promptVec, header.seqLen);

        // 결과 출력 (Algorithm 2의 결과인 가변 크기 확인용)
        LOG("  -> Generated " << chunks.size() << " dynamic chunks:");
        for(const auto& chunk : chunks) {
             LOG("     [SeqID " << chunk.seq_id << "] Size: " << chunk.tokens.size() 
                 << " (Start: " << chunk.start_pos << ")");
        }

        std::vector<int> generatedTokens;
        int currentPos = 0;

        // Pipeline execution (Logic remains same, data distribution changes)
        
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

        // Decoding phase (Token-by-token generation remains same)
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
