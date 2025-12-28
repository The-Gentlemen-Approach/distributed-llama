#ifndef HPIPE_POLICY_HPP
#define HPIPE_POLICY_HPP

#include "hpipe/core/types.hpp"
#include <cmath>
#include <algorithm>

// ==================================================================================
// 레이어 할당 정책 구현체 (Partitioning Policy Implementations)
// ==================================================================================

// 균등 할당 정책: 전체 세그먼트를 워커 수만큼 균등하게 나눕니다.
// 레이어가 아닌 세그먼트 단위로 분할하여 더 세밀한 부하 분산을 제공합니다.
// ⚠️ WARNING: This may split layers between workers, which currently has bugs!
class UniformSegmentPartitioningPolicy : public IPartitioningPolicy {
public:
    std::vector<SegmentRange> assignSegments(const SimpleLlmHeader& header, int n_workers) override {
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
    std::vector<SegmentRange> assignSegments(const SimpleLlmHeader& header, int n_workers) override {
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