#include "hpipe/network/worker.hpp"
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

    // 3. Root에게 Phase 1 ACK 전송 (토폴로지 수신 완료)
    writeHPipeAck(rootFd);
    printf("🔶 HPipeWorker: Sent Phase 1 ACK to root\n");

    // ==================================================================================
    // PHASE 2: Root로부터 next/prev 정보 수신 후 파이프라인 구성
    // ==================================================================================

    // 4. Next 노드 연결 설정
    NnSocket nextSock;
    if (!isLast) {
        // 다음 워커 정보 수신
        char nextHost[256];
        int nextPort;
        readSocket(rootFd, nextHost, 256);
        readSocket(rootFd, &nextPort, sizeof(nextPort));

        printf("🔶 HPipeWorker: Received next worker info: %s:%d\n", nextHost, nextPort);
        printf("🔶 HPipeWorker: Connecting to next worker at %s:%d\n", nextHost, nextPort);

        // 다음 워커가 준비될 때까지 재시도
        // Increased timeout to handle slow worker startup (model download, etc.)
        int maxRetries = 900;  // 최대 90초 대기 (900 * 100ms)
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

    // 5. Prev 노드 연결 설정 (나중에 수락)
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

    printf("🔶 HPipeWorker: Pipeline network initialized\n");

    // Root에게 Phase 2 ACK 전송 (파이프라인 구성 완료)
    writeHPipeAck(rootFd);
    printf("🔶 HPipeWorker: Sent Phase 2 ACK to root\n");

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
