#ifndef SIMPLE_LLM_INFERENCE_HPP
#define SIMPLE_LLM_INFERENCE_HPP

#include "common/llm-types.hpp"
#include "nn/nn-executor.hpp"

/**
 * Simple Inference Control (Simplified RootLlmInference).
 * Assumes local execution or simplified network control.
 */
class SimpleLlmInference {
public:
    float *logitsPipe;

private:
    float *tokenPipe;
    float *positionPipe;
    SimpleLlmHeader *header;
    NnNetExecution *execution;
    NnExecutor *executor;
    
public:
    SimpleLlmInference(SimpleLlmNet *net, NnNetExecution *execution, NnExecutor *executor);
    void setBatchSize(NnUint batchSize);
    void setPosition(NnUint position);
    void setToken(NnUint batchIndex, NnUint token);
    void forward();
};

#endif
