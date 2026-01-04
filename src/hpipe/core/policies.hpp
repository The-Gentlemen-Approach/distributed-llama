#ifndef HPIPE_POLICY_HPP
#define HPIPE_POLICY_HPP

#include "hpipe/core/types.hpp"
#include <cmath>
#include <algorithm>

// ==================================================================================
// 레이어 할당 정책 구현체 (Partitioning Policy Implementations)
// ==================================================================================


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
    double estimateSegmentFlops(const SimpleLlmHeader& header, int segmentIdx) {
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
    double calculateCommTime(const SimpleLlmHeader& header, int startSegmentIdx, const DeviceProfile& device) {
        // startSegmentIdx가 0이면(첫 시작) 통신 비용 없음 (입력 데이터는 제외 가정)
        if (startSegmentIdx == 0) return 0.0;

        // Activation Tensor Size: Batch(1) * SeqLen * HiddenDim * sizeof(float)
        // FP16이면 2byte, FP32면 4byte. 여기선 FP32(4byte) 가정
        double dataSizeBytes = (double)header.seqLen * header.hiddenDim * sizeof(float);
        
        // Time = DataSize / Bandwidth (GB/s -> Bytes/s 변환 필요)
        double bandwidthBytesPerSec = device.bandwidth * 1e9; 
        return dataSizeBytes / bandwidthBytesPerSec;
    }

    // Eq(3) T(a, b, m): 디바이스 m에서 세그먼트 a부터 b까지 처리하는 총 시간
    double calculateExecutionTime(const SimpleLlmHeader& header, int startSeg, int endSeg, int deviceIdx) {
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

    std::vector<SegmentRange> assignSegments(const SimpleLlmHeader& header, int n_workers) override {
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

#endif // HPIPE_POLICY_HPP
