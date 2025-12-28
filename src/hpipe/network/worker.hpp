#ifndef HPIPE_NETWORK_WORKER_HPP
#define HPIPE_NETWORK_WORKER_HPP

#include "hpipe/network/base.hpp"
#include <memory>

/**
 * HPipeWorkerNetwork: 워커 노드 네트워크
 */
class HPipeWorkerNetwork : public HPipeNetwork {
private:
    int rootSocket;        // 루트와의 연결 (설정 수신용)
    int prevSocket;        // 이전 노드 소켓 (데이터 수신, -1이면 없음)
    int nextSocket;        // 다음 노드 소켓 (데이터 전송, -1이면 없음)

    int workerId;          // 워커 ID
    bool isFirstWorker;    // 첫 워커 여부
    bool isLastWorker;     // 마지막 워커 여부

public:
    /**
     * serve: 워커 모드 - 루트 및 파이프라인 연결 대기
     */
    static std::unique_ptr<HPipeWorkerNetwork> serve(int port);

    HPipeWorkerNetwork(
        int rootFd,
        int prevFd,
        int nextFd,
        int workerId,
        bool isFirst,
        bool isLast
    );
    ~HPipeWorkerNetwork() override;

    // 설정 수신
    HPipeConfig recvConfigFromRoot();
    void sendConfigAck();  // Config 수신 확인 ACK 전송

    // 파이프라인 데이터 흐름
    void recvFromPrev(HPipeHeader* header, void* data, NnSize maxDataSize);
    void sendToNext(const HPipeHeader& header, const void* data, NnSize dataSize);

    // ACK 흐름 제어
    void sendAckToPrev();
    void waitAckFromNext();

    // Getter
    int getWorkerId() const { return workerId; }
    bool getIsFirstWorker() const { return isFirstWorker; }
    bool getIsLastWorker() const { return isLastWorker; }
};

#endif
