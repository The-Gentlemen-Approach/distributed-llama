/**
 * hpipe-pipeline-test.cpp: H-Pipe 파이프라인 오버랩 테스트
 *
 * 테스트 시나리오:
 * - Root가 여러 청크를 연속으로 전송
 * - Worker들이 비동기로 청크를 받아 처리하고 다음으로 전달
 * - 여러 청크가 동시에 파이프라인에 존재
 * - ACK를 통한 백프레셔 제어
 *
 * 실행 방법:
 * 터미널 1: ./hpipe-pipeline-test worker 9999 0
 * 터미널 2: ./hpipe-pipeline-test worker 10000 1
 * 터미널 3: ./hpipe-pipeline-test root 127.0.0.1:9999 127.0.0.1:10000
 */

#include "hpipe-network.hpp"
#include "simple-llm.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <thread>
#include <chrono>
#include <atomic>
#include <iomanip>

// 전역 시작 시간

// 타임스탬프 출력 헬퍼
std::string getTimestamp() {
    // 1. 현재 시스템 시간 가져오기
    auto now = std::chrono::system_clock::now();
    
    // 2. 밀리초(ms) 부분만 따로 계산 (time_t는 초 단위까지만 표현 가능하므로)
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

    // 3. 포맷팅을 위해 time_t 구조체로 변환
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm = *std::localtime(&now_c); 

    std::ostringstream oss;
    
    // 4. [시:분:초.밀리초] 형식으로 출력
    // %H(시), %M(분), %S(초)
    oss << "[" << std::put_time(&now_tm, "%H:%M:%S") 
        << "." << std::setfill('0') << std::setw(3) << ms.count() << "]";

    return oss.str();
}

#define LOG(msg) std::cout << getTimestamp() << " " << msg << "\n"

void printUsage() {
    std::cout << "Usage:\n";
    std::cout << "  Worker mode: ./hpipe-pipeline-test worker <port> <worker_id>\n";
    std::cout << "  Root mode:   ./hpipe-pipeline-test root <worker1_host:port> <worker2_host:port> ...\n";
}

SimpleLlmHeader createDummyHeader() {
    SimpleLlmHeader header;
    std::memset(&header, 0, sizeof(SimpleLlmHeader));
    header.version = 100;
    header.archType = SIMPLE_LLAMA;
    header.nLayers = 128;
    header.dim = 512;
    header.hiddenDim = 1024;
    header.nHeads = 8;
    header.headDim = 64;
    header.nKvHeads = 8;
    header.nExperts = 0;
    header.nActiveExperts = 0;
    header.vocabSize = 32000;
    header.seqLen = 2048;
    header.origSeqLen = 2048;
    header.weightType = F_32;
    header.syncType = F_32;
    return header;
}

// Worker의 파이프라인 처리 (비동기)
void testWorkerPipeline(int port, int expectedWorkerId) {
    LOG("=== H-Pipe Pipeline Worker Test (Port: " << port << ") ===");

    try {
        initSockets();
        auto network = HPipeWorkerNetwork::serve(port);

        LOG("✓ Network initialized");
        LOG("Worker ID: " << network->getWorkerId());
        LOG("Is First: " << network->getIsFirstWorker());
        LOG("Is Last: " << network->getIsLastWorker());

        // 설정 수신
        HPipeConfig config = network->recvConfigFromRoot();
        network->sendConfigAck();
        LOG("✓ Config received - layers [" << config.layer_range.start
            << ", " << config.layer_range.end << "]");

        // 파이프라인 처리: 여러 청크를 연속으로 처리
        const int MAX_DATA_SIZE = 1024 * sizeof(float);
        std::vector<float> buffer(1024);
        int chunksProcessed = 0;

        LOG("Starting pipeline processing...");

        while (true) {
            HPipeHeader header;

            // 이전 노드로부터 데이터 수신
            try {
                network->recvFromPrev(&header, buffer.data(), MAX_DATA_SIZE);
            } catch (const std::exception& e) {
                // 연결 종료 시 루프 탈출
                LOG("Pipeline ended: " << e.what());
                break;
            }

            LOG("✓ Received chunk " << header.seq_id
                << " (step=" << header.step << ", tokens=" << header.n_tokens << ")");

            // 청크 처리 시뮬레이션 (실제로는 레이어 연산)
            std::this_thread::sleep_for(std::chrono::milliseconds(50));

            // 데이터 변형 (각 워커가 값을 증가시킴)
            for (int i = 0; i < 10; i++) {
                buffer[i] += 1.0f;
            }

            // 다음 노드로 전송
            network->sendToNext(header, buffer.data(), header.data_size);
            LOG("✓ Sent chunk " << header.seq_id << " to next");

            chunksProcessed++;
        }

        LOG("=== Worker " << network->getWorkerId()
            << " processed " << chunksProcessed << " chunks ===");

        cleanupSockets();
    } catch (const std::exception& e) {
        std::cerr << getTimestamp() << " Error: " << e.what() << "\n";
    }
}

// Root의 파이프라인 전송 (여러 청크 연속 전송)
void testRootPipeline(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Error: Not enough arguments for root mode\n";
        printUsage();
        return;
    }

    // 워커 주소 파싱
    std::vector<std::string> workerAddrs;
    for (int i = 2; i < argc; i++) {
        workerAddrs.push_back(argv[i]);
    }

    int nWorkers = workerAddrs.size();
    std::vector<char*> hosts(nWorkers);
    std::vector<int> ports(nWorkers);

    for (int i = 0; i < nWorkers; i++) {
        std::string addr = workerAddrs[i];
        size_t colonPos = addr.find(':');
        if (colonPos == std::string::npos) {
            std::cerr << "Error: Invalid address format: " << addr << "\n";
            return;
        }

        std::string host = addr.substr(0, colonPos);
        std::string port = addr.substr(colonPos + 1);

        hosts[i] = strdup(host.c_str());
        ports[i] = std::stoi(port);
    }

    LOG("=== H-Pipe Pipeline Root Test ===");
    LOG("Workers: " << nWorkers);

    try {
        initSockets();

        LOG("Waiting for workers to start...");
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto network = HPipeRootNetwork::connect(nWorkers, hosts.data(), ports.data());

        // 설정 생성 및 배포
        std::vector<HPipeConfig> configs;
        SimpleLlmHeader modelHeader = createDummyHeader();

        for (int i = 0; i < nWorkers; i++) {
            HPipeConfig config;
            std::memset(&config, 0, sizeof(HPipeConfig));
            config.worker_id = i;
            config.total_workers = nWorkers;

            int layersPerWorker = modelHeader.nLayers / nWorkers;
            config.layer_range.start = i * layersPerWorker;
            if (i == nWorkers - 1) {
                config.layer_range.end = modelHeader.nLayers - 1;
            } else {
                config.layer_range.end = (i + 1) * layersPerWorker - 1;
            }

            if (i < nWorkers - 1) {
                std::string nextAddr = workerAddrs[i + 1];
                size_t colonPos = nextAddr.find(':');
                std::string nextHost = nextAddr.substr(0, colonPos);
                std::string nextPort = nextAddr.substr(colonPos + 1);

                strncpy(config.next_host, nextHost.c_str(), sizeof(config.next_host) - 1);
                config.next_port = std::stoi(nextPort);
            }

            config.model_header = modelHeader;
            configs.push_back(config);
        }

        network->broadcastConfig(configs);
        LOG("✓ Config broadcast complete");

        // 파이프라인 테스트: 여러 청크를 연속으로 전송
        const int NUM_CHUNKS = 5;
        const int TEST_DATA_SIZE = 1024;

        LOG("");
        LOG("Sending " << NUM_CHUNKS << " chunks to pipeline...");

        // 전송된 청크 데이터 저장 (검증용)
        std::vector<std::vector<float>> sentChunks(NUM_CHUNKS);

        for (int chunkId = 0; chunkId < NUM_CHUNKS; chunkId++) {
            std::vector<float> testData(TEST_DATA_SIZE);
            for (int i = 0; i < TEST_DATA_SIZE; i++) {
                testData[i] = static_cast<float>(chunkId * 1000 + i);
            }
            sentChunks[chunkId] = testData;

            HPipeHeader header;
            header.magic = HPIPE_HEADER_MAGIC;
            header.seq_id = chunkId;
            header.step = 0;
            header.n_tokens = 10;
            header.data_size = TEST_DATA_SIZE * sizeof(float);

            network->sendToFirstWorker(header, testData.data(), header.data_size);
            LOG("✓ Sent chunk " << chunkId);

            // 파이프라인이 오버랩되도록 짧은 딜레이
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
        }

        LOG("");
        LOG("✓ All chunks sent to pipeline");
        LOG("Waiting for results...");
        LOG("");

        // 결과 수신
        int chunksReceived = 0;
        for (int i = 0; i < NUM_CHUNKS; i++) {
            std::vector<float> resultData(TEST_DATA_SIZE);
            HPipeHeader resultHeader;

            network->recvFromLastWorker(&resultHeader, resultData.data(), TEST_DATA_SIZE * sizeof(float));

            LOG("✓ Received result for chunk " << resultHeader.seq_id);

            // 결과 검증: 각 워커가 +1씩 했으므로 원본 + nWorkers 여야 함
            float expected = sentChunks[resultHeader.seq_id][0] + nWorkers;
            float actual = resultData[0];

            if (std::abs(actual - expected) < 0.01f) {
                LOG("  ✓ Data verified: " << sentChunks[resultHeader.seq_id][0]
                    << " + " << nWorkers << " = " << actual);
            } else {
                LOG("  ✗ Data mismatch: expected " << expected
                    << ", got " << actual);
            }

            chunksReceived++;
        }

        LOG("");
        LOG("=== Pipeline Test Complete ===");
        LOG("Chunks sent: " << NUM_CHUNKS);
        LOG("Chunks received: " << chunksReceived);

        if (chunksReceived == NUM_CHUNKS) {
            LOG("✅ Pipeline test PASSED!");
        } else {
            LOG("❌ Pipeline test FAILED!");
        }

        cleanupSockets();
    } catch (const std::exception& e) {
        std::cerr << getTimestamp() << " Error: " << e.what() << "\n";
    }

    for (int i = 0; i < nWorkers; i++) {
        free(hosts[i]);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "worker") {
        if (argc < 4) {
            std::cerr << "Error: worker mode requires port and worker_id\n";
            printUsage();
            return 1;
        }
        int port = std::stoi(argv[2]);
        int workerId = std::stoi(argv[3]);
        testWorkerPipeline(port, workerId);
    } else if (mode == "root") {
        testRootPipeline(argc, argv);
    } else {
        std::cerr << "Error: Invalid mode '" << mode << "'\n";
        printUsage();
        return 1;
    }

    return 0;
}
