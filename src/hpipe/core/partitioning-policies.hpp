#ifndef HPIPE_PARTITIONING_POLICIES_HPP
#define HPIPE_PARTITIONING_POLICIES_HPP

#include "hpipe/core/types.hpp"
#include "hpipe/core/device-profile.hpp"
#include <vector>
#include <limits>
#include <algorithm>

class OptimalWorkloadPartitioningPolicy : public IPartitioningPolicy {
private:
    std::vector<DeviceProfile> devices;

    double calculateExecutionTime(const LlmHeader& header, int startSeg, int endSeg, int deviceIdx) {
        const auto& device = devices[deviceIdx];

        double totalCompTime = 0.0;
        for (int k = startSeg; k <= endSeg; ++k) {
            totalCompTime += device.computationTime(header, k);
        }

        return totalCompTime + device.communicationTime(header);
    }

public:
    OptimalWorkloadPartitioningPolicy(const std::vector<DeviceProfile>& device_list)
        : devices(device_list) {}

    std::vector<SegmentRange> assignSegments(const LlmHeader& header, int n_workers) override {
        if (devices.size() < n_workers) {
             devices.resize(n_workers, DeviceProfile(100.0f, 50.0f));
        }

        int N = 2 * header.nLayers + 2;
        int M = n_workers;

        std::vector<std::vector<double>> dp(M + 1, std::vector<double>(N + 1, std::numeric_limits<double>::infinity()));
        std::vector<std::vector<int>> path(M + 1, std::vector<int>(N + 1, 0));

        for (int i = 1; i <= N; ++i) {
            dp[1][i] = calculateExecutionTime(header, 0, i - 1, 0);
            path[1][i] = 0;
        }

        for (int m = 2; m <= M; ++m) {
            for (int i = 1; i <= N; ++i) {
                for (int k = 0; k < i; ++k) {
                    double current_dev_time = calculateExecutionTime(header, k, i - 1, m - 1);
                    double bottleneck = std::max(dp[m - 1][k], current_dev_time);

                    if (bottleneck < dp[m][i]) {
                        dp[m][i] = bottleneck;
                        path[m][i] = k;
                    }
                }
            }
        }

        std::vector<SegmentRange> ranges(M);
        int current_end_idx = N;

        for (int m = M; m >= 1; --m) {
            int start_idx = path[m][current_end_idx];
            ranges[m - 1] = {start_idx, current_end_idx - 1};
            current_end_idx = start_idx;
        }

        return ranges;
    }
};

class UniformSegmentPartitioningPolicy : public IPartitioningPolicy {
public:
    std::vector<SegmentRange> assignSegments(const LlmHeader& header, int n_workers) override {
        std::vector<SegmentRange> ranges;
        int totalSegments = 2 * header.nLayers + 2;
        int segments_per_worker = totalSegments / n_workers;
        int remainder = totalSegments % n_workers;
        int current_segment = 0;

        for (int i = 0; i < n_workers; i++) {
            int start = current_segment;
            int num_segments = segments_per_worker + (i < remainder ? 1 : 0);
            current_segment += num_segments;
            int end = current_segment - 1;
            ranges.push_back({start, end});
        }
        return ranges;
    }
};

class UniformPartitioningPolicy : public IPartitioningPolicy {
public:
    std::vector<SegmentRange> assignSegments(const LlmHeader& header, int n_workers) override {
        std::vector<SegmentRange> ranges;
        int nLayers = header.nLayers;
        int layers_per_worker = nLayers / n_workers;
        int remainder_layers = nLayers % n_workers;
        int current_segment = 0;

        for (int i = 0; i < n_workers; i++) {
            int start = current_segment;
            if (i == 0) current_segment++;
            int num_layers = layers_per_worker + (i < remainder_layers ? 1 : 0);
            current_segment += num_layers * 2;
            int end = (i == n_workers - 1) ? 2 * nLayers + 1 : current_segment - 1;
            ranges.push_back({start, end});
        }
        return ranges;
    }
};

#endif // HPIPE_PARTITIONING_POLICIES_HPP
