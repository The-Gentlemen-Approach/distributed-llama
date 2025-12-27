/**
 * hpipe-network.hpp: H-Pipe 파이프라인 통신 계층
 *
 * 주요 구성요소:
 * - HPipeNetwork: 파이프라인 토폴로지 기반 통신 관리
 * - HPipeRootNetwork: 루트 노드 (워커 설정 배포 및 결과 수집)
 * - HPipeWorkerNetwork: 워커 노드 (prev→self→next 파이프라인 처리)
 *
 * 통신 구조:
 * - 파이프라인 토폴로지: Root → Worker1 → Worker2 → ... → Root
 * - ACK 기반 역방향 흐름 제어 (Backpressure)
 * - 청크 기반 데이터 전송 (HPipeHeader + Data)
 *
 * nn-network와의 차이점:
 * - 스타 토폴로지 → 선형 파이프라인 토폴로지
 * - 워커가 이전/다음 워커와 직접 연결
 * - 청크 단위 스트리밍 처리
 */
#ifndef HPIPE_NETWORK_HPP
#define HPIPE_NETWORK_HPP

#include "hpipe-core.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-core.hpp"
#include <memory>
#include <vector>

// Magic numbers for protocol validation
#define HPIPE_HEADER_MAGIC 0x48504950  // "HPIP"
#define HPIPE_CONTROL_MAGIC 0x41434B33 // "ACK3"

/**
 * HPipeNetwork: H-Pipe 파이프라인 통신 기반 클래스
 *
 * 역할:
 * - 파이프라인 토폴로지에서의 데이터 송수신
 * - HPipe 프로토콜 패킷 직렬화/역직렬화
 * - 통신 통계 추적
 */
class HPipeNetwork {
protected:
    NnSize sentBytes;    // 전송 바이트 통계
    NnSize recvBytes;    // 수신 바이트 통계

public:
    HPipeNetwork();
    virtual ~HPipeNetwork();

    // 통계 조회 및 리셋
    void getStats(NnSize *sent, NnSize *recv);
    void resetStats();

    // 프로토콜 패킷 전송/수신
    void sendConfig(int socket, const HPipeConfig& config);
    HPipeConfig recvConfig(int socket);

    void sendHeader(int socket, const HPipeHeader& header);
    HPipeHeader recvHeader(int socket);

    void sendControl(int socket, const HPipeControl& control);
    HPipeControl recvControl(int socket);

    // 데이터 전송/수신 (헤더와 함께)
    void sendData(int socket, const void* data, NnSize size);
    void recvData(int socket, void* data, NnSize size);
};

/**
 * HPipeRootNetwork: 루트 노드 네트워크
 *
 * 역할:
 * - 모든 워커에게 초기 설정 배포 (HPipeConfig)
 * - 첫 번째 워커에게 청크 전송
 * - 마지막 워커로부터 결과 수신
 * - ACK 수신으로 파이프라인 흐름 제어
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
     * @param nWorkers: 워커 수
     * @param hosts: 워커 호스트 배열
     * @param ports: 워커 포트 배열
     * @return HPipeRootNetwork 인스턴스
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

/**
 * HPipeWorkerNetwork: 워커 노드 네트워크
 *
 * 역할:
 * - 루트로부터 설정 수신
 * - 이전 노드(prev)로부터 데이터 수신
 * - 다음 노드(next)로 데이터 전송
 * - ACK 기반 역방향 흐름 제어
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
     * @param port: 수신 대기 포트
     * @return HPipeWorkerNetwork 인스턴스
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

#endif // HPIPE_NETWORK_HPP
