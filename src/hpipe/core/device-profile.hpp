#ifndef HPIPE_DEVICE_PROFILE_HPP
#define HPIPE_DEVICE_PROFILE_HPP

#include "hpipe/core/types.hpp"

class DeviceProfile {
public:
    DeviceProfile(float tf = 100.0f, float bw = 50.0f, float lat = 1e-5f);

    float getTflops() const { return tflops; }
    float getBandwidth() const { return bandwidth; }
    float getLatency() const { return latency; }

    double computationTime(const LlmHeader& header, int segmentIdx) const;
    double communicationTime(const LlmHeader& header) const;
    double computationTime(const LlmHeader& header, int segmentIdx, int sliceSize, int historySize) const;
    double communicationTime(const LlmHeader& header, int sliceSize) const;

private:
    float tflops;
    float bandwidth;
    float latency;

    static double calcSegmentFlops(const LlmHeader& header, int segmentIdx);
    static double calcSliceFlops(const LlmHeader& header, int segmentIdx, int sliceSize, int historySize);
};

#endif // HPIPE_DEVICE_PROFILE_HPP
