#ifndef HPIPE_SCHEDULING_POLICIES_HPP
#define HPIPE_SCHEDULING_POLICIES_HPP

#include "hpipe/core/types.hpp"
#include "hpipe/core/device-profile.hpp"
#include <vector>
#include <algorithm>
#include <limits>
#include <set>
#include <cmath> // ceil 사용을 위해 추가

class OptimalSequenceSlicingPolicy : public IPipelineScheduler {
private:
    LlmHeader header;
    std::vector<DeviceProfile> devices;
    std::vector<SegmentRange> assignedRanges;
    int wave_size; // GPU의 Tiling/Wave 크기 (예: 128, 256)

    // GPU의 계단식 연산 특성을 반영한 비용 계산 함수
    double calculateCost(int sliceSize, int historySize) {
        double maxStageLatency = 0.0;

        // [핵심 변경 사항]
        // GPU 연산 시간은 선형적(Linear)이지 않고, Wave/Tile 크기에 따라 계단식(Step-wise)으로 증가함.
        // 따라서 sliceSize를 wave_size의 배수로 '올림(Ceil)' 처리하여 '유효 연산 크기(Effective Size)'를 구함.
        // 이를 통해 TFLOPS(이론 성능)가 아닌 Achieved FLOPs(실제 성능) 기반의 시간을 모사함.
        
        int num_waves = (sliceSize + wave_size - 1) / wave_size;
        int effectiveSliceSize = num_waves * wave_size;
        
        // *참고: 만약 sliceSize가 0이면 비용도 0이어야 하므로 예외 처리 필요할 수 있으나,
        // 로직상 sliceSize >= 1 이 들어오므로 그대로 진행.

        for (size_t m = 0; m < devices.size(); ++m) {
            const auto& device = devices[m];
            const auto& range = assignedRanges[m];

            double totalCompTime = 0.0;
            for (int k = range.start; k <= range.end; ++k) {
                // 수정된 부분: sliceSize 대신 effectiveSliceSize를 사용하여
                // Padding이 포함된(실제 GPU가 수행하는) 연산 시간을 계산하도록 유도
                totalCompTime += device.computationTime(header, k, effectiveSliceSize, historySize);
            }
            
            // 통신 시간은 실제 데이터 양(sliceSize)에 비례할 수도 있고, 
            // 커널 런치 오버헤드 등을 고려해 effectiveSize를 쓸 수도 있으나,
            // 일반적으로 전송은 Byte 단위 packing이 가능하므로 sliceSize 유지 (또는 필요시 변경)
            double stageLatency = totalCompTime + device.communicationTime(header, sliceSize);

            if (stageLatency > maxStageLatency) {
                maxStageLatency = stageLatency;
            }
        }
        return maxStageLatency;
    }

public:
    // 생성자에 wave_size 추가 (기본값 256: A100/H100 등 최신 GPU의 통상적인 효율적 타일 크기 고려)
    OptimalSequenceSlicingPolicy(const LlmHeader& h,
                                 const std::vector<DeviceProfile>& devs,
                                 const std::vector<SegmentRange>& ranges,
                                 int wave_size = 256)
        : header(h), devices(devs), assignedRanges(ranges), wave_size(wave_size) {}

    std::vector<ChunkTask> schedule(const std::vector<int>& prompt_tokens, int max_seq_len) override {
        int N = prompt_tokens.size();
        int M = devices.size();

        std::vector<std::vector<double>> G(N + 1, std::vector<double>(N + 1, 0.0));
        std::set<double> T_set;

        // DP 테이블 구성
        for (int s_cur = 1; s_cur <= N; ++s_cur) {
            for (int prev = 0; prev < s_cur; ++prev) {
                int s_step = s_cur - prev;
                // calculateCost 내부에서 Wave Quantization 적용됨
                double cost = calculateCost(s_step, prev);
                G[s_cur][prev] = cost;
                T_set.insert(cost);
            }
        }

        std::vector<double> T(T_set.begin(), T_set.end());

        double T_star = std::numeric_limits<double>::infinity();
        std::vector<int> S_star;

        // Min-Max Latency 최적화 루프
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
