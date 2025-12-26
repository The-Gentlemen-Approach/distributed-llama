/**
 * nn-network.hpp: TCP 소켓 기반 노드 간 통신
 *
 * 주요 구성요소:
 * - 소켓 유틸리티 함수들
 * - NnNetwork: 다중 노드 통신 관리
 * - NnNetworkNodeSynchronizer: 네트워크 기반 동기화 구현
 *
 * 통신 구조:
 * - 루트 노드: 모든 워커와 연결 (스타 토폴로지)
 * - 워커 노드: 루트 노드와만 연결
 * - TCP로 안정적인 데이터 전송
 * - 터보 모드: TCP_NODELAY로 지연 시간 최소화
 */
#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include "nn-executor.hpp"

#define ROOT_SOCKET_INDEX 0  // 워커 관점에서 루트 노드의 소켓 인덱스

// ========== 소켓 유틸리티 함수 ==========

void initSockets();          // 소켓 시스템 초기화 (Windows에서 필요)
void cleanupSockets();       // 소켓 시스템 정리
int acceptSocket(int serverSocket);         // 클라이언트 연결 수락
void setReuseAddr(int socket);              // SO_REUSEADDR 설정
void writeSocket(int socket, const void* data, NnSize size);  // 데이터 전송
void readSocket(int socket, void* data, NnSize size);         // 데이터 수신
int createServerSocket(int port);           // 서버 소켓 생성
void destroySocket(int serverSocket);       // 소켓 종료

/**
 * NnConnectionSocketException: 연결 관련 예외
 * 소켓 생성, 바인딩, 연결 실패 시 발생
 */
class NnConnectionSocketException : public std::runtime_error {
public:
    NnConnectionSocketException(const std::string message);
};

/**
 * NnTransferSocketException: 데이터 전송 관련 예외
 * 읽기/쓰기 실패, 연결 끊김 시 발생
 */
class NnTransferSocketException : public std::runtime_error {
public:
    int code;  // 에러 코드
    NnTransferSocketException(int code, const std::string message);
};

/**
 * NnSocket: 소켓 RAII 래퍼
 * 소켓의 생명주기를 자동 관리 (소멸자에서 자동 close)
 */
class NnSocket {
public:
    int fd;  // 소켓 파일 디스크립터

    NnSocket();
    NnSocket(int fd);
    ~NnSocket();
    void assign(int fd);   // 소켓 할당
    int release();         // 소유권 해제 (close 하지 않음)
};

/**
 * NnSocketIo: 소켓 I/O 작업 기술자
 * writeMany/readMany에서 여러 소켓에 동시 작업 시 사용
 */
struct NnSocketIo {
    NnUint socketIndex;  // 소켓 인덱스
    const void *data;    // 데이터 포인터
    NnSize size;         // 데이터 크기
};

/**
 * NnNetwork: 다중 노드 통신 관리
 *
 * 역할:
 * - 여러 노드와의 TCP 연결 관리
 * - 데이터 송수신 (동기식)
 * - 통신 통계 추적
 * - 터보 모드 (TCP_NODELAY) 지원
 *
 * 사용 패턴:
 * - 루트: connect()로 모든 워커와 연결
 * - 워커: serve()로 루트의 연결 대기
 */
class NnNetwork {
private:
    int *sockets;       // 소켓 파일 디스크립터 배열
    NnSize *sentBytes;  // 소켓별 전송 바이트 통계
    NnSize *recvBytes;  // 소켓별 수신 바이트 통계

public:
    /**
     * serve: 워커 모드 - 루트의 연결을 대기
     * @param port: 수신 대기 포트
     * @return NnNetwork 인스턴스 (루트와의 연결 1개)
     */
    static std::unique_ptr<NnNetwork> serve(int port);

    /**
     * connect: 루트 모드 - 모든 워커에게 연결
     * @param nSockets: 워커 수
     * @param hosts: 워커 호스트 배열
     * @param ports: 워커 포트 배열
     * @return NnNetwork 인스턴스 (워커들과의 연결)
     */
    static std::unique_ptr<NnNetwork> connect(NnUint nSockets, char **hosts, NnUint *ports);

    NnUint nSockets;  // 연결된 소켓 수

    NnNetwork(std::vector<NnSocket> *sockets);
    ~NnNetwork();

    void setTurbo(bool enabled);  // TCP_NODELAY 설정 (낮은 지연 시간)

    // 단일 소켓 I/O
    void write(const NnUint socketIndex, const void *data, const NnSize size);
    void read(const NnUint socketIndex, void *data, const NnSize size);

    // ACK 동기화 (빈 메시지로 동기화 포인트 생성)
    void writeAck(const NnUint socketIndex);
    void readAck(const NnUint socketIndex);

    // 논블로킹 읽기 시도
    bool tryReadWithMaxAttempts(NnUint socketIndex, void *data, NnSize size, unsigned long maxAttempts);

    // 다중 소켓 I/O (병렬)
    void writeMany(NnUint n, NnSocketIo *ios);  // 여러 소켓에 동시 전송
    void writeAll(void *data, NnSize size);     // 모든 소켓에 브로드캐스트
    void readMany(NnUint n, NnSocketIo *ios);   // 여러 소켓에서 동시 수신

    // 통계
    void getStats(NnSize *sentBytes, NnSize *recvBytes);
    void resetStats();
};

/**
 * NnNetworkNodeSynchronizer: 네트워크 기반 노드 동기화
 *
 * 세그먼트 실행 후 노드들 간 중간 결과를 교환
 * 동기화 타입에 따라 다른 패턴 사용:
 * - SYNC_WITH_ROOT: 루트가 모든 노드에 브로드캐스트
 * - SYNC_NODE_SLICES: 각 노드가 자신의 슬라이스를 전송
 * - SYNC_NODE_SLICES_EXCEPT_ROOT: 워커만 루트에게 전송
 */
class NnNetworkNodeSynchronizer : public NnNodeSynchronizer {
private:
    NnNetwork *network;
    NnNetExecution *execution;
    NnNetConfig *netConfig;
    NnNodeConfig *nodeConfig;
public:
    NnNetworkNodeSynchronizer(NnNetwork *network, NnNetExecution *execution, NnNetConfig *netConfig, NnNodeConfig *nodeConfig);
    ~NnNetworkNodeSynchronizer() override {};
    void sync(NnUint segmentIndex, NnUint nThreads, NnUint threadIndex) override;
};

class NnRootConfigWriter {
private:
    NnNetwork *network;
public:
    NnRootConfigWriter(NnNetwork *network);
    void writeNet(NnUint socketIndex, NnNetConfig *config);
    void writeNode(NnUint socketIndex, NnNodeConfig *config);
    void writeToWorkers(NnNetConfig *netConfig, NnNodeConfig *nodeConfigs);
};

class NnWorkerConfigReader {
private:
    NnNetwork *network;
public:
    NnWorkerConfigReader(NnNetwork *network);
    NnNetConfig readNet();
    NnNodeConfig readNode();
};

class NnRootWeightLoader {
private:
    NnExecutor *executor;
    NnNetwork *network;
    NnUint nNodes;
    NnByte *temp;
    NnSize tempSize;
public:
    NnRootWeightLoader(NnExecutor *executor, NnNetwork *network, NnUint nNodes);
    ~NnRootWeightLoader();
    void writeWeight(NnUint nodeIndex, const char *opName, NnUint opIndex, NnSize offset, NnSize nBytes, NnByte *weight);
    NnSize loadRoot(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadAll(const char *opName, NnUint opIndex, NnSize nBytes, NnByte *weight);
    NnSize loadRowMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnRowMatmulSlice *slice, NnByte *weight);
    NnSize loadColMatmulSlices(const char *opName, const NnUint opIndex, const NnUint expertIndex, NnColMatmulSlice *slice, NnByte *weight);
    void finish();
private:
    void allocate(NnSize size);};

class NnWorkerWeightReader {
private:
    NnExecutor *executor;
    NnNetwork *network;
    NnByte *temp;
    NnUint tempSize;
public:
    NnWorkerWeightReader(NnExecutor *executor, NnNetwork *network);
    ~NnWorkerWeightReader();
    void read();
private:
    void allocate(NnUint size);
};

#endif
