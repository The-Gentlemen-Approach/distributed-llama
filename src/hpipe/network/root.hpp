#ifndef HPIPE_NETWORK_ROOT_HPP
#define HPIPE_NETWORK_ROOT_HPP

#include "hpipe/network/base.hpp"
#include <vector>
#include <memory>

/**
 * HPipeRootNetwork: 루트 노드 네트워크
 */
class HPipeRootNetwork : public HPipeNetwork {
private:
    int firstWorkerSocket;   // 첫 워커와의 연결
    int lastWorkerSocket;    // 마지막 워커와의 연결
    std::vector<int> configSockets;  // 설정 배포용 소켓들 (중간 워커들)
    int nWorkers;            // 총 워커 수

public:
    /**
     * connect: 루트 모드 - 모든 워커에게 연결
     */
    static std::unique_ptr<HPipeRootNetwork> connect(
        int nWorkers,
        char** hosts,
        int* ports
    );

    HPipeRootNetwork(int nWorkers, int firstSocket, int lastSocket, std::vector<int>&& configSocks);
    ~HPipeRootNetwork() override;

    // 설정 배포
    void broadcastConfig(const std::vector<HPipeConfig>& configs);

    // 데이터 송수신 (첫/마지막 워커)
    void sendToFirstWorker(const HPipeHeader& header, const void* data, NnSize dataSize);
    void recvFromLastWorker(HPipeHeader* header, void* data, NnSize maxDataSize);

    // ACK 흐름 제어
    void waitAckFromFirstWorker();
    void sendAckToLastWorker();
};

#endif
