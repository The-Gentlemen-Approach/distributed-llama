#ifndef SIMPLE_LLM_WEIGHT_LOADER_HPP
#define SIMPLE_LLM_WEIGHT_LOADER_HPP

#include "common/llm-types.hpp"
#include "nn/nn-executor.hpp"

/**
 * Loads weights into the network.
 */
void loadSimpleLlmNetWeight(const char* path, SimpleLlmNet *net, NnExecutor *executor);

#endif
