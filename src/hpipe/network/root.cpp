#include "hpipe/network/root.hpp"
#include <cstring>
#include <cassert>
#include <stdexcept>
#include <cstdio>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>
#endif

// ==================================================================================
// HPipeRootNetwork: 루트 노드 구현
// ==================================================================================

std::unique_ptr<HPipeRootNetwork> HPipeRootNetwork::connect(
    int nWorkers,
    char** hosts,
    int* ports
) {
    assert(nWorkers > 0);

    printf("🔷 HPipeRoot: Connecting to %d workers...\n", nWorkers);

    // 모든 워커에게 연결
    std::vector<NnSocket> sockets;
    sockets.reserve(nWorkers);

    for (int i = 0; i < nWorkers; i++) {
        printf("🔷 HPipeRoot: Connecting to worker %d at %s:%d\n", i, hosts[i], ports[i]);

        // Connect using nn-network utility
        struct addrinfo hints, *addr = NULL;
        std::memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;

        char portStr[11];
        snprintf(portStr, sizeof(portStr), "%d", ports[i]);

        int addrinfoError = getaddrinfo(hosts[i], portStr, &hints, &addr);
        if (addrinfoError != 0 || addr == NULL) {
            throw NnConnectionSocketException("Cannot resolve worker address");
        }

        // Retry connection to handle slow worker startup or port forwarding setup
        int maxRetries = 60;  // 60 attempts * 1 second = 60 seconds
        int sock = -1;
        bool connected = false;

        for (int retry = 0; retry < maxRetries; retry++) {
            sock = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
            if (sock < 0) {
                if (retry < maxRetries - 1) {
                    #ifndef _WIN32
                    sleep(1);
                    #else
                    Sleep(1000);
                    #endif
                    continue;
                }
                freeaddrinfo(addr);
                throw std::runtime_error("Cannot create socket");
            }

            if (::connect(sock, addr->ai_addr, addr->ai_addrlen) == 0) {
                connected = true;
                break;
            }

            // Connection failed, close socket and retry
            #ifdef _WIN32
            closesocket(sock);
            #else
            close(sock);
            #endif
            sock = -1;

            if (retry < maxRetries - 1) {
                if (retry == 0) {
                    printf("🔷 HPipeRoot: Worker %d not ready yet, retrying...\n", i);
                }
                #ifndef _WIN32
                sleep(1);
                #else
                Sleep(1000);
                #endif
            }
        }

        freeaddrinfo(addr);

        if (!connected || sock < 0) {
            throw NnConnectionSocketException("Cannot connect to worker after retries");
        }
        sockets.emplace_back(sock);
        printf("🔷 HPipeRoot: Connected to worker %d\n", i);

        // 워커에게 토폴로지 정보 전송
        int workerId = i;
        int totalWorkers = nWorkers;
        bool isFirst = (i == 0);
        bool isLast = (i == nWorkers - 1);

        writeSocket(sock, &workerId, sizeof(workerId));
        writeSocket(sock, &totalWorkers, sizeof(totalWorkers));
        writeSocket(sock, &isFirst, sizeof(isFirst));
        writeSocket(sock, &isLast, sizeof(isLast));

        // 다음 워커 정보 전송 (마지막 워커가 아닐 때)
        if (!isLast) {
            writeSocket(sock, hosts[i + 1], 256);  // 고정 크기
            writeSocket(sock, &ports[i + 1], sizeof(ports[i + 1]));
        }

        printf("🔷 HPipeRoot: Sent topology info to worker %d (First=%d, Last=%d)\n",
               i, isFirst, isLast);
    }

    // 모든 워커가 파이프라인 연결을 완료할 때까지 대기
    printf("🔷 HPipeRoot: Waiting for all workers to be ready...\n");
    for (int i = 0; i < nWorkers; i++) {
        // 각 워커로부터 준비 완료 ACK 수신
        readHPipeAck(sockets[i].fd);
        printf("🔷 HPipeRoot: Worker %d is ready\n", i);
    }
    printf("🔷 HPipeRoot: All workers ready\n");

    // 첫 워커 = sockets[0], 마지막 워커 = sockets[nWorkers-1]
    int firstSocket = sockets[0].release();
    int lastSocket;

    if (nWorkers == 1) {
        // 첫 워커와 마지막 워커가 같음 (이미 release했으므로 복사)
        lastSocket = firstSocket;
    } else {
        lastSocket = sockets[nWorkers - 1].release();
    }

    // 나머지 소켓들은 설정 배포용으로 유지 (중간 워커들)
    std::vector<int> configSockets;
    for (int i = 1; i < nWorkers - 1; i++) {
        configSockets.push_back(sockets[i].release());
    }

    printf("🔷 HPipeRoot: All workers connected\n");
    printf("🔷 HPipeRoot: firstSocket=%d, lastSocket=%d, configSockets.size()=%zu\n",
           firstSocket, lastSocket, configSockets.size());
    return std::unique_ptr<HPipeRootNetwork>(
        new HPipeRootNetwork(nWorkers, firstSocket, lastSocket, std::move(configSockets))
    );
}

HPipeRootNetwork::HPipeRootNetwork(
    int nWorkers,
    int firstSocket,
    int lastSocket,
    std::vector<int>&& configSocks
) : HPipeNetwork(),
    firstWorkerSocket(firstSocket),
    lastWorkerSocket(lastSocket),
    nWorkers(nWorkers),
    configSockets(std::move(configSocks))
{
    printf("🔷 HPipeRootNetwork constructor: nWorkers=%d (configSockets.size=%zu, first==last=%d)\n",
           this->nWorkers, this->configSockets.size(), firstSocket == lastSocket);
    printf("🔷 HPipeRootNetwork: firstWorkerSocket=%d, lastWorkerSocket=%d\n",
           this->firstWorkerSocket, this->lastWorkerSocket);
    for (size_t i = 0; i < this->configSockets.size(); i++) {
        printf("🔷 HPipeRootNetwork: configSockets[%zu]=%d\n", i, this->configSockets[i]);
    }
}

HPipeRootNetwork::~HPipeRootNetwork() {
    if (firstWorkerSocket >= 0) {
        destroySocket(firstWorkerSocket);
    }
    for (int sock : configSockets) {
        if (sock >= 0) {
            destroySocket(sock);
        }
    }
    if (lastWorkerSocket >= 0 && lastWorkerSocket != firstWorkerSocket) {
        destroySocket(lastWorkerSocket);
    }
    printf("🔷 HPipeRoot: Network closed\n");
}

void HPipeRootNetwork::broadcastConfig(const std::vector<HPipeConfig>& configs) {
    printf("🔷 HPipeRoot: broadcastConfig called with configs.size()=%zu, nWorkers=%d\n",
           configs.size(), nWorkers);
    assert(configs.size() == static_cast<size_t>(nWorkers));

    printf("🔷 HPipeRoot: Broadcasting config to %d workers\n", nWorkers);

    // 첫 워커에게 설정 전송
    printf("🔷 HPipeRoot: Sending config to worker 0 via socket fd=%d\n", firstWorkerSocket);
    sendConfig(firstWorkerSocket, configs[0]);
    printf("🔷 HPipeRoot: Config sent to worker 0 (first)\n");

    // 중간 워커들에게 설정 전송
    for (size_t i = 0; i < configSockets.size(); i++) {
        printf("🔷 HPipeRoot: Sending config to worker %zu via socket fd=%d\n", i + 1, configSockets[i]);
        sendConfig(configSockets[i], configs[i + 1]);
        printf("🔷 HPipeRoot: Config sent to worker %zu\n", i + 1);
    }

    // 마지막 워커에게 설정 전송 (첫 워커와 다를 때만)
    if (firstWorkerSocket != lastWorkerSocket) {
        printf("🔷 HPipeRoot: Sending config to worker %zu via socket fd=%d\n", configs.size() - 1, lastWorkerSocket);
        sendConfig(lastWorkerSocket, configs[configs.size() - 1]);
        printf("🔷 HPipeRoot: Config sent to worker %zu (last)\n", configs.size() - 1);
    }

    printf("🔷 HPipeRoot: All configs broadcast complete\n");

    // 모든 워커가 config를 수신했는지 확인하는 ACK 대기
    printf("🔷 HPipeRoot: Waiting for config ACK from all workers...\n");
    for (int i = 0; i < nWorkers; i++) {
        int sock = (i == 0) ? firstWorkerSocket : (i == nWorkers - 1) ? lastWorkerSocket : configSockets[i - 1];
        readHPipeAck(sock);
        printf("🔷 HPipeRoot: Worker %d acknowledged config receipt\n", i);
    }
    printf("🔷 HPipeRoot: All workers acknowledged config\n");
}

void HPipeRootNetwork::sendToFirstWorker(
    const HPipeHeader& header,
    const void* data,
    NnSize dataSize
) {
    sendHeader(firstWorkerSocket, header);
    if (dataSize > 0) {
        sendData(firstWorkerSocket, data, dataSize);
    }
}

void HPipeRootNetwork::recvFromLastWorker(
    HPipeHeader* header,
    void* data,
    NnSize maxDataSize
) {
    *header = recvHeader(lastWorkerSocket);
    if (header->data_size > 0) {
        if (header->data_size > maxDataSize) {
            throw std::runtime_error("Received data size exceeds buffer");
        }
        recvData(lastWorkerSocket, data, header->data_size);
    }
}

void HPipeRootNetwork::waitAckFromFirstWorker() {
    HPipeControl ack = recvControl(firstWorkerSocket);
    printf("🔷 HPipeRoot: ACK received from first worker (seq_id=%u)\n", ack.seq_id);
}

void HPipeRootNetwork::sendAckToLastWorker() {
    HPipeControl ack;
    ack.magic = HPIPE_CONTROL_MAGIC;
    ack.seq_id = 0;  // Can be set appropriately
    sendControl(lastWorkerSocket, ack);
    printf("🔷 HPipeRoot: ACK sent to last worker\n");
}
