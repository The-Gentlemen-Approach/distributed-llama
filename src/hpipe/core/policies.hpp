#ifndef HPIPE_POLICY_HPP
#define HPIPE_POLICY_HPP

#include "hpipe/core/types.hpp"
#include <cmath>
#include <algorithm>

// ==================================================================================
// 데이터 구조: 디바이스 성능 프로필
// ==================================================================================
struct DeviceProfile {
    float tflops;      // Tera Floating Point Operations Per Second (Computation power)
    float bandwidth;   // GB/s (Communication speed)
    
    // 생성자 예시 (기본값 설정)
    DeviceProfile(float tf = 100.0f, float bw = 50.0f) : tflops(tf), bandwidth(bw) {}
};


// ==================================================================================
// 레이어 할당 정책 구현체 (Partitioning Policy Implementations)
// ==================================================================================

// ... (기존 UniformSegmentPartitioningPolicy 등은 그대로 유지) ...

// [NEW] 최적 부하 분산 정책 (Algorithm 1 Based)
// 논문의 Eq(3), Eq(4)를 기반으로 Dynamic Programming을 수행하여
// 각 디바이스의 FLOPs와 Bandwidth를 고려한 최적의 분할 지점을 찾습니다.
class OptimalWorkloadPartitioningPolicy : public IPartitioningPolicy {
private:
    std::vector<DeviceProfile> devices;

    // 세그먼트 타입에 따른 연산량(FLOPs) 추정 (Heuristic)
    // 실제 구현 시에는 프로파일링 된 데이터를 사용하는 것이 가장 정확합니다.
    double estimateSegmentFlops(const LlmHeader& header, int segmentIdx) {
        // Segment 구조: 0(Emb) -> 1(Attn), 2(FFN) ... -> Last(Cls)
        long long hidden = header.hiddenDim;
        long long seq = header.seqLen; // 혹은 max_seq_len이나 평균치 사용
        
        if (segmentIdx == 0) { 
            // Embedding: Vocabulary lookup (메모리 대역폭 의존적이나 FLOPs로 환산)
            return (double)header.vocabSize * hidden; 
        } 
        else if (segmentIdx == 2 * header.nLayers + 1) {
            // Classifier: Linear projection to vocab
            return (double)header.vocabSize * hidden;
        } 
        else {
            // Layers (Attn or FFN)
            // 0번은 Emb이므로, 레이어 인덱스 계산 필요
            // segment 1,2 -> layer 0 / segment 3,4 -> layer 1
            bool isAttn = (segmentIdx % 2 != 0);
            
            if (isAttn) {
                // Attention: 4 * Projection(q,k,v,o) + Attention Score
                // Projections: 4 * (h * h) * seq
                // Attention Ops: 2 * (h/head) * seq * seq * heads ... (간략화)
                // 대략 4 * h^2
                return 4.0 * hidden * hidden * seq; 
            } else {
                // FFN (Gated): 3 * Projections (gate, up, down)
                // Usually expansion factor is intermediate_size (or hidden * 4)
                // 대략 3 * h * (intermediate) * seq
                long long intermediate = header.moeHiddenDim > 0 ? header.moeHiddenDim : header.hiddenDim * 4; 
                return 3.0 * hidden * intermediate * seq;
            }
        }
    }

    // 통신 비용 계산 (이전 디바이스 -> 현재 디바이스 데이터 전송)
    double calculateCommTime(const LlmHeader& header, int startSegmentIdx, const DeviceProfile& device) {
        // startSegmentIdx가 0이면(첫 시작) 통신 비용 없음 (입력 데이터는 제외 가정)
        //if (startSegmentIdx == 0) return 0.0;

        // Activation Tensor Size: Batch(1) * SeqLen * HiddenDim * sizeof(float)
        // FP16이면 2byte, FP32면 4byte. 여기선 FP32(4byte) 가정
        double dataSizeBytes = (double)header.seqLen * header.hiddenDim * sizeof(float);
        
        // Time = DataSize / Bandwidth (GB/s -> Bytes/s 변환 필요)
        double bandwidthBytesPerSec = device.bandwidth * 1e9; 
        return dataSizeBytes / bandwidthBytesPerSec;
    }

    // Eq(3) T(a, b, m): 디바이스 m에서 세그먼트 a부터 b까지 처리하는 총 시간
    double calculateExecutionTime(const LlmHeader& header, int startSeg, int endSeg, int deviceIdx) {
        double totalCompTime = 0.0;
        const auto& device = devices[deviceIdx];

        // 1. Computation Time: sum(t_comp)
        for (int k = startSeg; k <= endSeg; ++k) {
            double flops = estimateSegmentFlops(header, k);
            double deviceFlopsPerSec = device.tflops * 1e12; // TFLOPS -> FLOPS
            totalCompTime += (flops / deviceFlopsPerSec);
        }

        // 2. Communication Time: t_comm
        // 이 디바이스가 작업을 시작하기 위해 데이터를 받아오는 시간
        double commTime = calculateCommTime(header, startSeg, device);

        return totalCompTime + commTime;
    }

public:
    // 생성자에서 디바이스 정보 주입
    OptimalWorkloadPartitioningPolicy(const std::vector<DeviceProfile>& device_list) 
        : devices(device_list) {}

    std::vector<SegmentRange> assignSegments(const LlmHeader& header, int n_workers) override {
        // 디바이스 정보가 부족하면 기본 프로필 추가
        if (devices.size() < n_workers) {
             devices.resize(n_workers, DeviceProfile(100.0f, 50.0f)); // Default fallback
        }

        int N = 2 * header.nLayers + 2; // Total Segments
        int M = n_workers;              // Total Devices

        // DP Table: dp[m][i] 
        // m번째 디바이스까지 사용해서 i번째 세그먼트까지 처리했을 때의 Minimized Maximum Time
        std::vector<std::vector<double>> dp(M + 1, std::vector<double>(N + 1, std::numeric_limits<double>::infinity()));
        
        // Path Reconstruction Table: path[m][i] -> m번째 디바이스가 어디서부터(start index) 맡았는지 저장
        std::vector<std::vector<int>> path(M + 1, std::vector<int>(N + 1, 0));

        // ---------------------------------------------------------
        // 1. Initialization (First Worker)
        // 첫 번째 워커는 무조건 0번부터 i번까지 혼자 다 해야 함
        // ---------------------------------------------------------
        for (int i = 1; i <= N; ++i) {
            // start=0 (index 0 corresponds to 1st segment in logic, code uses 0-based)
            dp[1][i] = calculateExecutionTime(header, 0, i - 1, 0); 
            path[1][i] = 0; // 0번 세그먼트부터 시작
        }

        // ---------------------------------------------------------
        // 2. Dynamic Programming (Algorithm 1)
        // ---------------------------------------------------------
        for (int m = 2; m <= M; ++m) { // For each device m from 2 to M
            for (int i = 1; i <= N; ++i) { // For each segment endpoint i
                
                // Find best split point k (0 <= k < i)
                // m번째 디바이스는 k+1 (index로는 k) 부터 i-1 까지 담당
                for (int k = 0; k < i; ++k) {
                    // Cost on Device m: Segments k to i-1
                    // 주의: calculateExecutionTime의 인자는 0-based index
                    double current_dev_time = calculateExecutionTime(header, k, i - 1, m - 1);
                    
                    // Bottleneck time: max(previous bottleneck, current device time)
                    double bottleneck = std::max(dp[m - 1][k], current_dev_time);

                    if (bottleneck < dp[m][i]) {
                        dp[m][i] = bottleneck;
                        path[m][i] = k; // Split point: m-th device starts at k
                    }
                }
            }
        }

        // ---------------------------------------------------------
        // 3. Backtracking to find ranges
        // ---------------------------------------------------------
        std::vector<SegmentRange> ranges(M);
        int current_end_idx = N; // Total segments count (end boundary)

        for (int m = M; m >= 1; --m) {
            int start_idx = path[m][current_end_idx];
            // SegmentRange is inclusive [start, end]
            ranges[m - 1] = {start_idx, current_end_idx - 1};
            current_end_idx = start_idx;
        }

        return ranges;
    }
};

// 균등 할당 정책: 전체 세그먼트를 워커 수만큼 균등하게 나눕니다.
// 레이어가 아닌 세그먼트 단위로 분할하여 더 세밀한 부하 분산을 제공합니다.
// ⚠️ WARNING: This may split layers between workers, which currently has bugs!
class UniformSegmentPartitioningPolicy : public IPartitioningPolicy {
public:
    std::vector<SegmentRange> assignSegments(const LlmHeader& header, int n_workers) override {
        std::vector<SegmentRange> ranges;

        // Segment structure (one by one):
        // - Segment 0: Embedding
        // - Segments 1-2: Layer 0 (1=attn, 2=ffn)
        // - Segments 3-4: Layer 1 (3=attn, 4=ffn)
        // - ...
        // - Segments 2*nLayers-1, 2*nLayers: Layer nLayers-1
        // - Segment 2*nLayers+1: Classifier
        //
        // Total segments = 1 (emb) + 2*nLayers (layers) + 1 (cls) = 2*nLayers + 2

        int totalSegments = 2 * header.nLayers + 2;
        int segments_per_worker = totalSegments / n_workers;
        int remainder = totalSegments % n_workers;

        int current_segment = 0;

        for (int i = 0; i < n_workers; i++) {
            int start = current_segment;

            // Each worker gets base amount + 1 extra if they're in the remainder group
            int num_segments = segments_per_worker + (i < remainder ? 1 : 0);
            current_segment += num_segments;

            int end = current_segment - 1;

            ranges.push_back({start, end});
        }

        return ranges;
    }
};

// Layer-aware 할당 정책: 레이어를 절대 분할하지 않음 (attn + ffn은 항상 같은 워커)
// 더 안정적이지만 부하 분산이 덜 세밀합니다.
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

            // First worker gets embedding
            if (i == 0) {
                current_segment++; // Skip embedding (segment 0)
            }

            // Assign complete layers to this worker
            int num_layers = layers_per_worker + (i < remainder_layers ? 1 : 0);
            current_segment += num_layers * 2; // Each layer has 2 segments (attn + ffn)

            // Last worker gets classifier
            int end;
            if (i == n_workers - 1) {
                end = 2 * nLayers + 1; // Include classifier
            } else {
                end = current_segment - 1;
            }

            ranges.push_back({start, end});
        }

        return ranges;
    }
};


// ==================================================================================
// 파이프라인 스케줄러 구현체 (Pipeline Scheduler Implementations)
// ==================================================================================

// 고정 크기 청크 스케줄러: 프롬프트를 지정된 크기의 청크로 잘라서 스케줄링합니다.
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


// [NEW] [Algorithm 2 Implementation] 최적 시퀀스 슬라이싱 정책 (Strict Mode)
// 논문의 Algorithm 2 변수명(G, L, S)과 로직을 그대로 구현
class OptimalSequenceSlicingPolicy : public IPipelineScheduler {
private:
    LlmHeader header;
    std::vector<DeviceProfile> devices;
    std::vector<SegmentRange> assignedRanges; 

    // 단일 슬라이스 비용 계산 (Eq 1 기반)
    // sliceSize: 현재 자르는 크기 (s_step)
    // historySize: 이전에 처리된 누적 크기 (s_cur - s_step)
    double calculateCost(int sliceSize, int historySize) {
        double maxStageLatency = 0.0;
        
        for (size_t m = 0; m < devices.size(); ++m) {
            const auto& device = devices[m];
            const auto& range = assignedRanges[m];
            double totalCompTime = 0.0;
            
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
            double dataSize = (double)sliceSize * header.hiddenDim * sizeof(float);
            double commTime = dataSize / (device.bandwidth * 1e9);

            double stageLatency = totalCompTime + commTime;
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
        // Pre-calculation: Construct Matrix G (Algorithm 2 requires G as input)
        // G[i][j] : Cost of slice ending at i, starting after j.
        // i corresponds to s_cur, j corresponds to (s_cur - s_step) aka history
        // ---------------------------------------------------------
        std::vector<std::vector<double>> G(N + 1, std::vector<double>(N + 1, 0.0));
        std::set<double> T_set; // To collect all unique latencies for Line 1

        for (int s_cur = 1; s_cur <= N; ++s_cur) {
            for (int prev = 0; prev < s_cur; ++prev) {
                int s_step = s_cur - prev;
                // Cost for slice of size s_step with history size 'prev'
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
            
            // DP Tables
            // L: Array to record latency
            // S: Array to trace the sequence slicing
            std::vector<double> L(N + 1, std::numeric_limits<double>::infinity());
            std::vector<int> S_trace(N + 1, 0);

            // Base case: 0 latency for 0 tokens
            L[0] = 0.0; 

            // 4: for s_cur from 1 to N do
            for (int s_cur = 1; s_cur <= N; ++s_cur) {
                
                // 5: L[s_cur] <- infinity (Already init)

                // 6: for s_step from 1 to s_cur do
                for (int s_step = 1; s_step <= s_cur; ++s_step) {
                    
                    // 7: l_step <- G[s_cur][s_cur - s_step]
                    double l_step = G[s_cur][s_cur - s_step];

                    // 8: l_total <- L[s_cur - s_step] + l_step
                    double prev_latency = L[s_cur - s_step];
                    
                    if (prev_latency == std::numeric_limits<double>::infinity()) continue;

                    double l_total = prev_latency + l_step;

                    // 9: if l_step <= t_max && l_total < L[s_cur] then
                    // Note: 논문 슈도코드에는 "s_cur <= t_max"라고 되어 있으나(Line 9), 
                    // s_cur는 인덱스(길이)고 t_max는 시간(double)이므로 명백한 오타 혹은 문맥상
                    // "현재 슬라이스 비용(l_step) <= t_max"를 의미함. (Eq 5의 제약조건과 일치)
                    if (l_step <= t_max && l_total < L[s_cur]) {
                        
                        // 10: L[s_cur] <- l_total
                        L[s_cur] = l_total;
                        
                        // 11: S[s_cur] <- s_step
                        S_trace[s_cur] = s_step;
                    }
                }
            }

            // Valid path found to the end?
            if (L[N] != std::numeric_limits<double>::infinity()) {
                
                // 12-15: Reconstruction (Inside strict loop, effectively)
                // Derive the sequence slicing S
                std::vector<int> S_temp;
                int i = N;
                while (i > 0) {
                    S_temp.push_back(S_trace[i]);
                    i = i - S_trace[i];
                }
                std::reverse(S_temp.begin(), S_temp.end());

                // 16: T = (M - 1) * t_max + L[N]
                double T_val = (double)(M - 1) * t_max + L[N];

                // 17: if T < T* then
                if (T_val < T_star) {
                    // 18: T* <- T, S* <- S
                    T_star = T_val;
                    S_star = S_temp;
                }
            }
        }

        // Final Task Generation using S*
        std::vector<ChunkTask> tasks;
        int current_pos = 0;
        int seq_id = 0;

        if (S_star.empty()) {
            // Fallback (e.g., single chunk) if optimization fails unexpectedly
            S_star.push_back(N); 
        }

        for (int size : S_star) {
            ChunkTask task;
            task.seq_id = seq_id++;
            task.start_pos = current_pos;
            // Vector slicing
            task.tokens = std::vector<int>(prompt_tokens.begin() + current_pos, 
                                         prompt_tokens.begin() + current_pos + size);
            tasks.push_back(task);
            current_pos += size;
        }

        return tasks;
    }
};

#endif // HPIPE_POLICY_HPP
