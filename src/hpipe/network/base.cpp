#include "hpipe/network/base.hpp"
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

void HPipeNetwork::writeHPipeAck(int socket) {
    NnUint packet = HPIPE_ACK;
    writeSocket(socket, &packet, sizeof(packet));
}

void HPipeNetwork::readHPipeAck(int socket) {
    NnUint packet;
    readSocket(socket, &packet, sizeof(packet));
    if (packet != HPIPE_ACK) {
        throw std::runtime_error("Invalid H-Pipe ack packet");
    }
}
