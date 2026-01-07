#ifndef HPIPE_SCHEDULING_POLICIES_HPP
#define HPIPE_SCHEDULING_POLICIES_HPP

#include "hpipe/core/types.hpp"
#include "hpipe/core/device-profile.hpp"
#include <vector>
#include <algorithm>
#include <limits>
#include <set>

class OptimalSequenceSlicingPolicy : public IPipelineScheduler {
private:
    LlmHeader header;
    std::vector<DeviceProfile> devices;
    std::vector<SegmentRange> assignedRanges;

    double calculateCost(int sliceSize, int historySize) {
        double maxStageLatency = 0.0;

        for (size_t m = 0; m < devices.size(); ++m) {
            const auto& device = devices[m];
            const auto& range = assignedRanges[m];

            double totalCompTime = 0.0;
            for (int k = range.start; k <= range.end; ++k) {
                totalCompTime += device.computationTime(header, k, sliceSize, historySize);
            }

            double stageLatency = totalCompTime + device.communicationTime(header, sliceSize);

            if (stageLatency > maxStageLatency) {
                maxStageLatency = stageLatency;
            }
        }
        return maxStageLatency;
    }

public:
    OptimalSequenceSlicingPolicy(const LlmHeader& h,
                                 const std::vector<DeviceProfile>& devs,
                                 const std::vector<SegmentRange>& ranges)
        : header(h), devices(devs), assignedRanges(ranges) {}

    std::vector<ChunkTask> schedule(const std::vector<int>& prompt_tokens, int max_seq_len) override {
        int N = prompt_tokens.size();
        int M = devices.size();

        std::vector<std::vector<double>> G(N + 1, std::vector<double>(N + 1, 0.0));
        std::set<double> T_set;

        for (int s_cur = 1; s_cur <= N; ++s_cur) {
            for (int prev = 0; prev < s_cur; ++prev) {
                int s_step = s_cur - prev;
                double cost = calculateCost(s_step, prev);
                G[s_cur][prev] = cost;
                T_set.insert(cost);
            }
        }

        std::vector<double> T(T_set.begin(), T_set.end());

        double T_star = std::numeric_limits<double>::infinity();
        std::vector<int> S_star;

        for (double t_max : T) {

            std::vector<double> L(N + 1, std::numeric_limits<double>::infinity());
            std::vector<int> S_trace(N + 1, 0);
            L[0] = 0.0;

            for (int s_cur = 1; s_cur <= N; ++s_cur) {
                for (int s_step = 1; s_step <= s_cur; ++s_step) {

                    double l_step = G[s_cur][s_cur - s_step];
                    double prev_latency = L[s_cur - s_step];

                    if (prev_latency == std::numeric_limits<double>::infinity()) continue;

                    double l_total = prev_latency + l_step;

                    if (l_step <= t_max && l_total < L[s_cur]) {
                        L[s_cur] = l_total;
                        S_trace[s_cur] = s_step;
                    }
                }
            }

            if (L[N] != std::numeric_limits<double>::infinity()) {

                std::vector<int> S_temp;
                int i = N;
                while (i > 0) {
                    S_temp.push_back(S_trace[i]);
                    i = i - S_trace[i];
                }
                std::reverse(S_temp.begin(), S_temp.end());

                double T_val = (double)(M - 1) * t_max + L[N];

                if (T_val < T_star) {
                    T_star = T_val;
                    S_star = S_temp;
                }
            }
        }

        std::vector<ChunkTask> tasks;
        int current_pos = 0;
        int seq_id = 0;

        if (S_star.empty()) {
            S_star.push_back(N);
        }

        for (int size : S_star) {
            ChunkTask task;
            task.seq_id = seq_id++;
            task.start_pos = current_pos;
            task.tokens = std::vector<int>(prompt_tokens.begin() + current_pos,
                                         prompt_tokens.begin() + current_pos + size);
            tasks.push_back(task);
            current_pos += size;
        }

        return tasks;
    }
};

class FixedChunkScheduler : public IPipelineScheduler {
private:
    int chunk_size;

public:
    FixedChunkScheduler(int chunk_size = 128) : chunk_size(chunk_size) {}

    std::vector<ChunkTask> schedule(const std::vector<int>& prompt_tokens, int max_seq_len) override {
        std::vector<ChunkTask> tasks;
        int total = prompt_tokens.size();
        int seq_id = 0;

        for (int i = 0; i < total; i += chunk_size) {
            int end = std::min(i + chunk_size, total);
            std::vector<int> chunk(prompt_tokens.begin() + i, prompt_tokens.begin() + end);

            ChunkTask task;
            task.seq_id = seq_id++;
            task.start_pos = i;
            task.tokens = chunk;
            tasks.push_back(task);
        }
        return tasks;
    }
};

#endif // HPIPE_SCHEDULING_POLICIES_HPP
