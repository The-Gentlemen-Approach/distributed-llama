#include "hpipe-network.hpp"
#include <cstring>
#include <cassert>
#include <stdexcept>

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

#define HPIPE_ACK 23571115  // Different from nn-network ACK

// Helper functions for ACK packets
static void writeHPipeAck(int socket) {
    NnUint packet = HPIPE_ACK;
    writeSocket(socket, &packet, sizeof(packet));
}

static void readHPipeAck(int socket) {
    NnUint packet;
    readSocket(socket, &packet, sizeof(packet));
    if (packet != HPIPE_ACK) {
        throw std::runtime_error("Invalid H-Pipe ack packet");
    }
}

// ==================================================================================
// HPipeNetwork: 기본 클래스 구현
// ==================================================================================

HPipeNetwork::HPipeNetwork() : sentBytes(0), recvBytes(0) {}

HPipeNetwork::~HPipeNetwork() {}

void HPipeNetwork::getStats(NnSize *sent, NnSize *recv) {
    *sent = sentBytes;
    *recv = recvBytes;
    resetStats();
}

void HPipeNetwork::resetStats() {
    sentBytes = 0;
    recvBytes = 0;
}

void HPipeNetwork::sendConfig(int socket, const HPipeConfig& config) {
    writeSocket(socket, &config, sizeof(HPipeConfig));
    sentBytes += sizeof(HPipeConfig);
}

HPipeConfig HPipeNetwork::recvConfig(int socket) {
    HPipeConfig config;
    readSocket(socket, &config, sizeof(HPipeConfig));
    recvBytes += sizeof(HPipeConfig);
    return config;
}

void HPipeNetwork::sendHeader(int socket, const HPipeHeader& header) {
    // Validate magic number
    if (header.magic != HPIPE_HEADER_MAGIC) {
        throw std::runtime_error("Invalid HPipeHeader magic number");
    }
    writeSocket(socket, &header, sizeof(HPipeHeader));
    sentBytes += sizeof(HPipeHeader);
}

HPipeHeader HPipeNetwork::recvHeader(int socket) {
    HPipeHeader header;
    readSocket(socket, &header, sizeof(HPipeHeader));
    recvBytes += sizeof(HPipeHeader);

    // Validate magic number
    if (header.magic != HPIPE_HEADER_MAGIC) {
        throw std::runtime_error("Invalid HPipeHeader magic number received");
    }
    return header;
}

void HPipeNetwork::sendControl(int socket, const HPipeControl& control) {
    // Validate magic number
    if (control.magic != HPIPE_CONTROL_MAGIC) {
        throw std::runtime_error("Invalid HPipeControl magic number");
    }
    writeSocket(socket, &control, sizeof(HPipeControl));
    sentBytes += sizeof(HPipeControl);
}

HPipeControl HPipeNetwork::recvControl(int socket) {
    HPipeControl control;
    readSocket(socket, &control, sizeof(HPipeControl));
    recvBytes += sizeof(HPipeControl);

    // Validate magic number
    if (control.magic != HPIPE_CONTROL_MAGIC) {
        throw std::runtime_error("Invalid HPipeControl magic number received");
    }
    return control;
}

void HPipeNetwork::sendData(int socket, const void* data, NnSize size) {
    writeSocket(socket, data, size);
    sentBytes += size;
}

void HPipeNetwork::recvData(int socket, void* data, NnSize size) {
    readSocket(socket, data, size);
    recvBytes += size;
}

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

        int sock = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
        if (sock < 0) {
            throw std::runtime_error("Cannot create socket");
        }

        if (::connect(sock, addr->ai_addr, addr->ai_addrlen) != 0) {
            throw NnConnectionSocketException("Cannot connect to worker");
        }

        freeaddrinfo(addr);
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

// ==================================================================================
// HPipeWorkerNetwork: 워커 노드 구현
// ==================================================================================

std::unique_ptr<HPipeWorkerNetwork> HPipeWorkerNetwork::serve(int port) {
    printf("🔶 HPipeWorker: Starting on port %d...\n", port);

    // 서버 소켓 생성
    NnSocket serverSocket(createServerSocket(port));
    printf("🔶 HPipeWorker: Listening on port %d\n", port);

    // 1. 루트로부터 연결 수락 (설정 수신용)
    int rootFd = acceptSocket(serverSocket.fd);
    NnSocket rootSock(rootFd);
    printf("🔶 HPipeWorker: Root connected\n");

    // 2. 루트로부터 초기 정보 수신 (worker ID, topology info)
    int workerId;
    int totalWorkers;
    bool isFirst;
    bool isLast;

    readSocket(rootFd, &workerId, sizeof(workerId));
    readSocket(rootFd, &totalWorkers, sizeof(totalWorkers));
    readSocket(rootFd, &isFirst, sizeof(isFirst));
    readSocket(rootFd, &isLast, sizeof(isLast));

    printf("🔶 HPipeWorker: ID=%d, Total=%d, First=%d, Last=%d\n",
           workerId, totalWorkers, isFirst, isLast);

    // 3. Next 노드 연결 설정 (먼저 연결)
    NnSocket nextSock;
    if (!isLast) {
        // 다음 워커에게 연결
        char nextHost[256];
        int nextPort;
        readSocket(rootFd, nextHost, 256);
        readSocket(rootFd, &nextPort, sizeof(nextPort));

        printf("🔶 HPipeWorker: Connecting to next worker at %s:%d\n", nextHost, nextPort);

        // 다음 워커가 준비될 때까지 재시도
        int maxRetries = 50;  // 최대 5초 대기 (50 * 100ms)
        int sock = -1;
        for (int retry = 0; retry < maxRetries; retry++) {
            struct addrinfo hints, *addr = NULL;
            std::memset(&hints, 0, sizeof(hints));
            hints.ai_family = AF_INET;
            hints.ai_socktype = SOCK_STREAM;
            hints.ai_protocol = IPPROTO_TCP;

            char portStr[11];
            snprintf(portStr, sizeof(portStr), "%d", nextPort);

            int addrinfoError = getaddrinfo(nextHost, portStr, &hints, &addr);
            if (addrinfoError != 0 || addr == NULL) {
                #ifndef _WIN32
                usleep(100000);  // 100ms
                #else
                Sleep(100);
                #endif
                continue;
            }

            sock = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
            if (sock < 0) {
                freeaddrinfo(addr);
                #ifndef _WIN32
                usleep(100000);
                #else
                Sleep(100);
                #endif
                continue;
            }

            if (::connect(sock, addr->ai_addr, addr->ai_addrlen) == 0) {
                // 연결 성공!
                freeaddrinfo(addr);
                break;
            }

            // 연결 실패, 재시도
            #ifdef _WIN32
            closesocket(sock);
            #else
            close(sock);
            #endif
            sock = -1;
            freeaddrinfo(addr);

            #ifndef _WIN32
            usleep(100000);  // 100ms
            #else
            Sleep(100);
            #endif
        }

        if (sock < 0) {
            throw NnConnectionSocketException("Cannot connect to next worker after retries");
        }

        nextSock.assign(sock);
        printf("🔶 HPipeWorker: Connected to next worker\n");
    } else {
        // 마지막 워커는 루트로 결과를 보냄 (nextSock은 비워둠)
        printf("🔶 HPipeWorker: Last worker - next is root\n");
    }

    // 4. Prev 노드 연결 설정 (나중에 수락)
    NnSocket prevSock;
    if (!isFirst) {
        // 이전 워커로부터 연결 수락
        printf("🔶 HPipeWorker: Waiting for prev worker connection...\n");
        int prevFd = acceptSocket(serverSocket.fd);
        prevSock.assign(prevFd);
        printf("🔶 HPipeWorker: Prev worker connected\n");
    } else {
        // 첫 워커는 루트로부터 데이터를 받음 (prevSock은 비워둠)
        printf("🔶 HPipeWorker: First worker - prev is root\n");
    }

    printf("🔶 HPipeWorker: Network initialized\n");

    // Root에게 준비 완료 신호 전송
    writeHPipeAck(rootFd);
    printf("🔶 HPipeWorker: Sent ready ACK to root\n");

    // release()를 사용하여 소켓 소유권을 이전 (close 방지)
    int rootSocketFd = rootSock.release();
    int prevSocketFd = isFirst ? -1 : prevSock.release();
    int nextSocketFd = isLast ? -1 : nextSock.release();

    return std::unique_ptr<HPipeWorkerNetwork>(
        new HPipeWorkerNetwork(
            rootSocketFd,
            prevSocketFd,
            nextSocketFd,
            workerId,
            isFirst,
            isLast
        )
    );
}

HPipeWorkerNetwork::HPipeWorkerNetwork(
    int rootFd,
    int prevFd,
    int nextFd,
    int workerId,
    bool isFirst,
    bool isLast
) : HPipeNetwork(),
    rootSocket(rootFd),
    prevSocket(prevFd),
    nextSocket(nextFd),
    workerId(workerId),
    isFirstWorker(isFirst),
    isLastWorker(isLast)
{
    assert(rootSocket >= 0);
}

HPipeWorkerNetwork::~HPipeWorkerNetwork() {
    if (rootSocket >= 0) destroySocket(rootSocket);
    if (prevSocket >= 0) destroySocket(prevSocket);
    if (nextSocket >= 0) destroySocket(nextSocket);
    printf("🔶 HPipeWorker %d: Network closed\n", workerId);
}

HPipeConfig HPipeWorkerNetwork::recvConfigFromRoot() {
    printf("🔶 HPipeWorker %d: Attempting to recv config from rootSocket=%d\n", workerId, rootSocket);
    HPipeConfig config = recvConfig(rootSocket);
    printf("🔶 HPipeWorker %d: Config received from root\n", workerId);
    return config;
}

void HPipeWorkerNetwork::sendConfigAck() {
    writeHPipeAck(rootSocket);
    printf("🔶 HPipeWorker %d: Config ACK sent to root\n", workerId);
}

void HPipeWorkerNetwork::recvFromPrev(
    HPipeHeader* header,
    void* data,
    NnSize maxDataSize
) {
    // 첫 워커는 루트로부터, 아니면 이전 워커로부터 수신
    int socket = isFirstWorker ? rootSocket : prevSocket;

    if (socket < 0) {
        throw std::runtime_error("Invalid socket in recvFromPrev");
    }

    *header = recvHeader(socket);

    if (header->data_size > 0) {
        if (header->data_size > maxDataSize) {
            throw std::runtime_error("Received data size exceeds buffer");
        }
        recvData(socket, data, header->data_size);
    }
}

void HPipeWorkerNetwork::sendToNext(
    const HPipeHeader& header,
    const void* data,
    NnSize dataSize
) {
    // 마지막 워커는 루트로, 아니면 다음 워커로 전송
    int socket = isLastWorker ? rootSocket : nextSocket;

    if (socket < 0) {
        throw std::runtime_error("Invalid socket in sendToNext");
    }

    sendHeader(socket, header);

    if (dataSize > 0) {
        sendData(socket, data, dataSize);
    }
}

void HPipeWorkerNetwork::sendAckToPrev() {
    HPipeControl ack;
    ack.magic = HPIPE_CONTROL_MAGIC;
    ack.seq_id = 0;  // Can be set appropriately

    int socket = isFirstWorker ? rootSocket : prevSocket;

    if (socket < 0) {
        throw std::runtime_error("Invalid socket in sendAckToPrev");
    }

    sendControl(socket, ack);
    printf("🔶 HPipeWorker %d: ACK sent to prev\n", workerId);
}

void HPipeWorkerNetwork::waitAckFromNext() {
    int socket = isLastWorker ? rootSocket : nextSocket;

    if (socket < 0) {
        throw std::runtime_error("Invalid socket in waitAckFromNext");
    }

    HPipeControl ack = recvControl(socket);
    printf("🔶 HPipeWorker %d: ACK received from next (seq_id=%u)\n", workerId, ack.seq_id);
}
