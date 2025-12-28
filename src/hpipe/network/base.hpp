#ifndef HPIPE_NETWORK_BASE_HPP
#define HPIPE_NETWORK_BASE_HPP

#include "hpipe/core/types.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-core.hpp"
#include <memory>
#include <vector>

// Magic numbers for protocol validation
#define HPIPE_HEADER_MAGIC 0x48504950  // "HPIP"
#define HPIPE_CONTROL_MAGIC 0x41434B33 // "ACK3"
#define HPIPE_ACK 23571115             // Connection ACK

/**
 * HPipeNetwork: H-Pipe 파이프라인 통신 기반 클래스
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
    
    // Connection helpers
    static void writeHPipeAck(int socket);
    static void readHPipeAck(int socket);
};

#endif // HPIPE_NETWORK_BASE_HPP