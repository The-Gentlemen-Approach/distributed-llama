/**
 * hpipe-network-test.cpp: H-Pipe 네트워크 테스트
 *
 * 테스트 시나리오:
 * 1. Root → Worker1 → Worker2 → Root 파이프라인 생성
 * 2. Root가 모든 워커에게 설정 배포
 * 3. Root가 첫 워커에게 데이터 청크 전송
 * 4. Worker1이 처리 후 Worker2로 전달
 * 5. Worker2가 처리 후 Root로 결과 반환
 * 6. ACK 기반 흐름 제어 테스트
 *
 * 실행 방법:
 * 터미널 1: ./hpipe-network-test worker 9999 0
 * 터미널 2: ./hpipe-network-test worker 10000 1
 * 터미널 3: ./hpipe-network-test root 127.0.0.1:9999 127.0.0.1:10000
 */

#include "hpipe/network/root.hpp"
#include "hpipe/network/worker.hpp"
#include "common/llm-types.hpp"
#include "hpipe/core/utils.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <thread>
#include <chrono>

void printUsage() {
    std::cout << "Usage:\n";
    std::cout << "  Worker mode: ./hpipe-network-test worker <port> <worker_id>\n";
    std::cout << "  Root mode:   ./hpipe-network-test root <worker1_host:port> <worker2_host:port> ...\n";
    std::cout << "\nExample:\n";
    std::cout << "  Terminal 1: ./hpipe-network-test worker 9999 0\n";
    std::cout << "  Terminal 2: ./hpipe-network-test worker 10000 1\n";
    std::cout << "  Terminal 3: ./hpipe-network-test root 127.0.0.1:9999 127.0.0.1:10000\n";
}

// 테스트용 더미 모델 헤더 생성
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

// Root 노드 테스트
void testRoot(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Error: Not enough arguments for root mode\n";
        printUsage();
        return;
    }

    // 워커 주소 파싱 (argv[2]부터 시작)
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

    std::cout << "=== H-Pipe Root Test ===\n";
    std::cout << "Workers: " << nWorkers << "\n";

    try {
        // 소켓 초기화
        initSockets();

        // 워커 대기 (워커들이 먼저 시작할 시간 확보)
        std::cout << "Waiting for workers to start...\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // 워커들에게 연결
        auto network = HPipeRootNetwork::connect(nWorkers, hosts.data(), ports.data());

        // 설정 생성 및 배포
        std::vector<HPipeConfig> configs;
        SimpleLlmHeader modelHeader = createDummyHeader();

        for (int i = 0; i < nWorkers; i++) {
            HPipeConfig config;
            std::memset(&config, 0, sizeof(HPipeConfig));
            config.worker_id = i;
            config.total_workers = nWorkers;

            // 세그먼트 할당 (균등 분배 + 마지막 워커가 나머지 처리)
            // Total segments = 2 * nLayers + 2 (embedding + layers*2 + classifier)
            int totalSegments = 2 * modelHeader.nLayers + 2;
            int segmentsPerWorker = totalSegments / nWorkers;
            config.segment_range.start = i * segmentsPerWorker;

            if (i == nWorkers - 1) {
                // 마지막 워커는 남은 모든 세그먼트를 담당
                config.segment_range.end = totalSegments - 1;
            } else {
                config.segment_range.end = (i + 1) * segmentsPerWorker - 1;
            }

            // 다음 워커 주소 설정
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

            std::cout << "Config for Worker " << i << ": segments ["
                      << config.segment_range.start << ", "
                      << config.segment_range.end << "] ("
                      << (config.segment_range.end - config.segment_range.start + 1)
                      << " segments)\n";
        }

        network->broadcastConfig(configs);
        std::cout << "✓ Config broadcast complete\n";

        // 테스트 데이터 전송
        const int TEST_DATA_SIZE = 1024;
        std::vector<float> testData(TEST_DATA_SIZE);
        for (int i = 0; i < TEST_DATA_SIZE; i++) {
            testData[i] = static_cast<float>(i);
        }

        HPipeHeader header;
        header.magic = HPIPE_HEADER_MAGIC;
        header.seq_id = 1;
        header.step = 0;
        header.n_tokens = 10;
        header.data_size = TEST_DATA_SIZE * sizeof(float);

        std::cout << "Sending test data to first worker...\n";
        network->sendToFirstWorker(header, testData.data(), header.data_size);
        std::cout << "✓ Test data sent\n";

        // 마지막 워커로부터 결과 수신
        std::vector<float> resultData(TEST_DATA_SIZE);
        HPipeHeader resultHeader;

        std::cout << "Waiting for result from last worker...\n";
        network->recvFromLastWorker(&resultHeader, resultData.data(), TEST_DATA_SIZE * sizeof(float));
        std::cout << "✓ Result received\n";

        // 결과 검증
        std::cout << "Result header: seq_id=" << resultHeader.seq_id
                  << ", step=" << resultHeader.step
                  << ", n_tokens=" << resultHeader.n_tokens
                  << ", data_size=" << resultHeader.data_size << "\n";

        bool dataMatch = true;
        for (int i = 0; i < 10; i++) {
            if (std::abs(resultData[i] - testData[i]) > 0.01f) {
                dataMatch = false;
                break;
            }
        }

        if (dataMatch) {
            std::cout << "✓ Data verification passed!\n";
        } else {
            std::cout << "✗ Data verification failed!\n";
        }

        // 통계 출력
        NnSize sent, recv;
        network->getStats(&sent, &recv);
        std::cout << "Network stats: sent=" << sent << " bytes, recv=" << recv << " bytes\n";

        std::cout << "=== Root Test Complete ===\n";

        cleanupSockets();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }

    // 메모리 정리
    for (int i = 0; i < nWorkers; i++) {
        free(hosts[i]);
    }
}

// Worker 노드 테스트
void testWorker(int port, int expectedWorkerId) {
    std::cout << "=== H-Pipe Worker Test (Port: " << port << ") ===\n";

    try {
        // 소켓 초기화
        initSockets();

        // 네트워크 초기화 (루트 연결 대기)
        auto network = HPipeWorkerNetwork::serve(port);

        std::cout << "✓ Network initialized\n";
        std::cout << "Worker ID: " << network->getWorkerId() << "\n";
        std::cout << "Is First: " << network->getIsFirstWorker() << "\n";
        std::cout << "Is Last: " << network->getIsLastWorker() << "\n";

        // 설정 수신
        HPipeConfig config = network->recvConfigFromRoot();
        std::cout << "✓ Config received\n";
        std::cout << "Assigned segments: [" << config.segment_range.start
                  << ", " << config.segment_range.end << "]\n";
        std::cout << "Total workers: " << config.total_workers << "\n";

        // Config 수신 확인 ACK 전송
        network->sendConfigAck();
        std::cout << "✓ Config ACK sent\n";

        // 데이터 수신
        const int MAX_DATA_SIZE = 1024 * sizeof(float);
        std::vector<float> buffer(1024);
        HPipeHeader header;

        std::cout << "Waiting for data from prev...\n";
        network->recvFromPrev(&header, buffer.data(), MAX_DATA_SIZE);
        std::cout << "✓ Data received from prev\n";
        std::cout << "Header: seq_id=" << header.seq_id
                  << ", step=" << header.step
                  << ", n_tokens=" << header.n_tokens
                  << ", data_size=" << header.data_size << "\n";

        // 간단한 처리 시뮬레이션 (여기서는 그대로 전달)
        std::cout << "Processing data...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // 다음 노드로 전송
        std::cout << "Sending data to next...\n";
        network->sendToNext(header, buffer.data(), header.data_size);
        std::cout << "✓ Data sent to next\n";

        // 통계 출력
        NnSize sent, recv;
        network->getStats(&sent, &recv);
        std::cout << "Network stats: sent=" << sent << " bytes, recv=" << recv << " bytes\n";

        std::cout << "=== Worker Test Complete ===\n";

        cleanupSockets();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
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
        testWorker(port, workerId);
    } else if (mode == "root") {
        testRoot(argc, argv);
    } else {
        std::cerr << "Error: Invalid mode '" << mode << "'\n";
        printUsage();
        return 1;
    }

    return 0;
}
