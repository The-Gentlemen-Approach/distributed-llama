#ifndef HPIPE_POLICY_HPP
#define HPIPE_POLICY_HPP

#include "hpipe/core/types.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <set>
#include <iostream>

// ==================================================================================
// 데이터 구조: 디바이스 성능 프로필
// ==================================================================================
struct DeviceProfile {
    float tflops;      // Tera Floating Point Operations Per Second (Computation power)
    float bandwidth;   // GB/s (Communication speed)
    float latency;     // [NEW] Fixed Overhead per operation (seconds)
                       // 통신 및 커널 실행을 위한 준비 시간 (기본값 10us)

    // 생성자 업데이트 (latency 추가)
    DeviceProfile(float tf = 100.0f, float bw = 50.0f, float lat = 1e-5f) 
        : tflops(tf), bandwidth(bw), latency(lat) {}
};


// ==================================================================================
// 레이어 할당 정책 구현체 (Partitioning Policy Implementations)
// ==================================================================================

// [Algorithm 1 Based] 최적 부하 분산 정책
class OptimalWorkloadPartitioningPolicy : public IPartitioningPolicy {
private:
    std::vector<DeviceProfile> devices;

    double estimateSegmentFlops(const LlmHeader& header, int segmentIdx) {
        long long hidden = header.hiddenDim;
        long long seq = header.seqLen; 
        
        if (segmentIdx == 0) { 
            return (double)header.vocabSize * hidden; 
        } 
        else if (segmentIdx == 2 * header.nLayers + 1) {
            return (double)header.vocabSize * hidden;
        } 
        else {
            bool isAttn = (segmentIdx % 2 != 0);
            if (isAttn) {
                return 4.0 * hidden * hidden * seq; 
            } else {
                long long intermediate = header.moeHiddenDim > 0 ? header.moeHiddenDim : header.hiddenDim * 4; 
                return 3.0 * hidden * intermediate * seq;
            }
        }
    }

    double calculateCommTime(const LlmHeader& header, int startSegmentIdx, const DeviceProfile& device) {
        double dataSizeBytes = (double)header.seqLen * header.hiddenDim * sizeof(float);
        double bandwidthBytesPerSec = device.bandwidth * 1e9; 
        return dataSizeBytes / bandwidthBytesPerSec;
    }

    double calculateExecutionTime(const LlmHeader& header, int startSeg, int endSeg, int deviceIdx) {
        double totalCompTime = 0.0;
        const auto& device = devices[deviceIdx];

        // 1. Computation Time
        for (int k = startSeg; k <= endSeg; ++k) {
            double flops = estimateSegmentFlops(header, k);
            double deviceFlopsPerSec = device.tflops * 1e12; 
            totalCompTime += (flops / deviceFlopsPerSec);
        }

        // 2. Communication Time
        double commTime = calculateCommTime(header, startSeg, device);

        // Partitioning 단계에서는 큰 단위(Layer)를 다루므로 Latency 영향이 적지만, 
        // 일관성을 위해 포함할 수도 있음. 여기서는 Slicing 이슈 해결이 주 목적이므로 
        // 기존 로직 유지 (또는 device.latency 추가 가능)
        return totalCompTime + commTime;
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

        // Initialization
        for (int i = 1; i <= N; ++i) {
            dp[1][i] = calculateExecutionTime(header, 0, i - 1, 0); 
            path[1][i] = 0; 
        }

        // DP
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

        // Backtracking
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

// 균등 할당 정책 (기존 유지)
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

// Layer-aware 할당 정책 (기존 유지)
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


// ==================================================================================
// 파이프라인 스케줄러 구현체 (Pipeline Scheduler Implementations)
// ==================================================================================

// 고정 크기 청크 스케줄러 (기존 유지)
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

// [NEW] [Algorithm 2 Implementation] 최적 시퀀스 슬라이싱 정책 (Strict Mode + Latency Awareness)
// 논문의 Algorithm 2를 기반으로 하되, 고정 오버헤드(Latency)를 비용 함수에 반영하여
// 너무 작은 슬라이스로 쪼개지는 것을 방지함.
class OptimalSequenceSlicingPolicy : public IPipelineScheduler {
private:
    LlmHeader header;
    std::vector<DeviceProfile> devices;
    std::vector<SegmentRange> assignedRanges; 

    // 단일 슬라이스 비용 계산 (Eq 1 기반 + Latency 추가)
    // sliceSize: 현재 자르는 크기
    // historySize: 이전에 처리된 누적 크기
    double calculateCost(int sliceSize, int historySize) {
        double maxStageLatency = 0.0;
        
        for (size_t m = 0; m < devices.size(); ++m) {
            const auto& device = devices[m];
            const auto& range = assignedRanges[m];
            double totalCompTime = 0.0;
            
            // 1. Computation Time
            for (int k = range.start; k <= range.end; ++k) {
                double flops = 0.0;
                long long h = header.hiddenDim;
                
                if (k == 0 || k == 2 * header.nLayers + 1) {
                    flops = (double)header.vocabSize * h * sliceSize; 
                } 
                else if (k % 2 != 0) { // Attention
                    // Cost depends on current slice AND history
                    double attnOps = 2.0 * sliceSize * (sliceSize + historySize) * h;
                    double projOps = 4.0 * sliceSize * h * h;
                    flops = attnOps + projOps;
                } 
                else { // FFN
                    long long inter = header.moeHiddenDim > 0 ? header.moeHiddenDim : h * 4;
                    flops = 3.0 * h * inter * sliceSize; 
                }
                totalCompTime += (flops / (device.tflops * 1e12));
            }

            // 2. Communication Time (Bandwidth dependent)
            double dataSize = (double)sliceSize * header.hiddenDim * sizeof(float);
            double commTime = dataSize / (device.bandwidth * 1e9);

            // [핵심 변경] 3. Fixed Overhead (Latency)
            // 통신 준비 및 커널 실행에 걸리는 고정 시간 추가
            // 이를 통해 '1개씩 자르기' 같은 비효율적 분할을 억제함
            double stageLatency = totalCompTime + commTime + device.latency;

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
        int N = prompt_tokens.size(); // Total Sequence Length
        int M = devices.size();       // Number of Workers

        // ---------------------------------------------------------
        // Pre-calculation: Construct Matrix G (Algorithm 2)
        // ---------------------------------------------------------
        std::vector<std::vector<double>> G(N + 1, std::vector<double>(N + 1, 0.0));
        std::set<double> T_set;

        for (int s_cur = 1; s_cur <= N; ++s_cur) {
            for (int prev = 0; prev < s_cur; ++prev) {
                int s_step = s_cur - prev;
                // Latency가 포함된 비용 계산 호출
                double cost = calculateCost(s_step, prev);
                G[s_cur][prev] = cost; 
                T_set.insert(cost);
            }
        }

        // 1: T <- all possible latency in G
        std::vector<double> T(T_set.begin(), T_set.end());

        // 2: T* <- infinity, S* <- None
        double T_star = std::numeric_limits<double>::infinity();
        std::vector<int> S_star;

        // 3: for t_max in T do
        for (double t_max : T) {
            
            std::vector<double> L(N + 1, std::numeric_limits<double>::infinity());
            std::vector<int> S_trace(N + 1, 0);
            L[0] = 0.0; 

            // 4: for s_cur from 1 to N do
            for (int s_cur = 1; s_cur <= N; ++s_cur) {
                // 6: for s_step from 1 to s_cur do
                for (int s_step = 1; s_step <= s_cur; ++s_step) {
                    
                    double l_step = G[s_cur][s_cur - s_step];
                    double prev_latency = L[s_cur - s_step];
                    
                    if (prev_latency == std::numeric_limits<double>::infinity()) continue;

                    double l_total = prev_latency + l_step;

                    // 9: Check constraints
                    if (l_step <= t_max && l_total < L[s_cur]) {
                        L[s_cur] = l_total;
                        S_trace[s_cur] = s_step;
                    }
                }
            }

            // Valid path found?
            if (L[N] != std::numeric_limits<double>::infinity()) {
                
                // Reconstruct slicing scheme
                std::vector<int> S_temp;
                int i = N;
                while (i > 0) {
                    S_temp.push_back(S_trace[i]);
                    i = i - S_trace[i];
                }
                std::reverse(S_temp.begin(), S_temp.end());

                // 16: Evaluate Total Pipeline Latency
                double T_val = (double)(M - 1) * t_max + L[N];

                if (T_val < T_star) {
                    T_star = T_val;
                    S_star = S_temp;
                }
            }
        }

        // Final Task Generation
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

#endif // HPIPE_POLICY_HPP
