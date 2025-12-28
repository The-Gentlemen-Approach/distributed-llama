#ifndef SIMPLE_LLM_NETWORK_BUILDER_HPP
#define SIMPLE_LLM_NETWORK_BUILDER_HPP

#include "common/llm-types.hpp"
#include "nn/nn-core.hpp"

/**
 * Builds the network for single-node execution.
 */
SimpleLlmNet buildSimpleLlmNet(SimpleLlmHeader *h, NnUint nBatches);

#endif
