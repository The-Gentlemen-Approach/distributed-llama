#ifndef HPIPE_LLM_WEIGHT_LOADER_HPP
#define HPIPE_LLM_WEIGHT_LOADER_HPP

#include "common/llm-types.hpp"
#include "nn/nn-executor.hpp"

/**
 * Loads weights for assigned segments only.
 *
 * @param path Path to the model file
 * @param net SimpleLlmNet structure
 * @param executor NnExecutor instance
 * @param startSegment Index of the first segment assigned to this node
 * @param endSegment Index of the last segment assigned to this node
 */
void loadHpipeLlmNetWeight(const char *path, SimpleLlmNet *net, NnExecutor *executor, int startSegment, int endSegment);

#endif
