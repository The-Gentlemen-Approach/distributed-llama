#include "device-profile.hpp"

DeviceProfile::DeviceProfile(float tf, float bw, float lat)
    : tflops(tf), bandwidth(bw), latency(lat) {}

double DeviceProfile::computationTime(const LlmHeader& header, int segmentIdx) const {
    double flops = calcSegmentFlops(header, segmentIdx);
    return flops / (tflops * 1e12);
}

double DeviceProfile::communicationTime(const LlmHeader& header) const {
    double dataSize = (double)header.seqLen * header.hiddenDim * sizeof(float);
    return dataSize / (bandwidth * 1e9);
}

double DeviceProfile::computationTime(const LlmHeader& header, int segmentIdx, int sliceSize, int historySize) const {
    double flops = calcSliceFlops(header, segmentIdx, sliceSize, historySize);
    return flops / (tflops * 1e12);
}

double DeviceProfile::communicationTime(const LlmHeader& header, int sliceSize) const {
    double dataSize = (double)sliceSize * header.hiddenDim * sizeof(float);
    return dataSize / (bandwidth * 1e9) + latency;
}

double DeviceProfile::calcSegmentFlops(const LlmHeader& header, int segmentIdx) {
    long long h = header.hiddenDim;
    long long seq = header.seqLen;

    if (segmentIdx == 0 || segmentIdx == 2 * header.nLayers + 1) {
        return (double)header.vocabSize * h;
    }

    bool isAttn = (segmentIdx % 2 != 0);
    if (isAttn) {
        double projOps = 4.0 * seq * h * h;
        double attnOps = 2.0 * seq * seq * h;
        return projOps + attnOps;
    } else {
        long long inter = header.moeHiddenDim > 0 ? header.moeHiddenDim : h * 4;
        return 3.0 * h * inter * seq;
    }
}

double DeviceProfile::calcSliceFlops(const LlmHeader& header, int segmentIdx, int sliceSize, int historySize) {
    long long h = header.hiddenDim;

    if (segmentIdx == 0 || segmentIdx == 2 * header.nLayers + 1) {
        return (double)header.vocabSize * h * sliceSize;
    }

    bool isAttn = (segmentIdx % 2 != 0);
    if (isAttn) {
        double projOps = 4.0 * sliceSize * h * h;
        double attnOps = 2.0 * sliceSize * (sliceSize + historySize) * h;
        return projOps + attnOps;
    } else {
        long long inter = header.moeHiddenDim > 0 ? header.moeHiddenDim : h * 4;
        return 3.0 * h * inter * sliceSize;
    }
}
