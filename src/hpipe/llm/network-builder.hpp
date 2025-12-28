#ifndef HPIPE_LLM_NETWORK_BUILDER_HPP
#define HPIPE_LLM_NETWORK_BUILDER_HPP

#include "common/llm-types.hpp"
#include "nn/nn-core.hpp"

/**
 * Builds a partial network for HPipe execution.
 *
 * Segment Index Mapping:
 * 0: Embedding
 * 1: Layer 0 Attention
 * 2: Layer 0 FFN
 * 3: Layer 1 Attention
 * ...
 * 2*N: Layer N-1 FFN
 * 2*N+1: Classifier
 *
 * @param h Model header
 * @param nBatches Number of batches
 * @param startSegment Index of the first segment assigned to this node
 * @param endSegment Index of the last segment assigned to this node
 * @return Initialized SimpleLlmNet structure
 */
SimpleLlmNet buildHpipeLlmNet(SimpleLlmHeader *h, NnUint nBatches, int startSegment, int endSegment);

#endif
