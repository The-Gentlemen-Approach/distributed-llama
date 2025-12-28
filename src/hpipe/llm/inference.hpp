#ifndef HPIPE_LLM_INFERENCE_HPP
#define HPIPE_LLM_INFERENCE_HPP

#include "common/llm-types.hpp"
#include "nn/nn-core.hpp"
#include "nn/nn-executor.hpp"

/**
 * HPipe Inference Control.
 * Manages input/output tensors for pipeline segments.
 */
class HPipeLlmInference {
public:
    float *inputPipe;  // Pointer to input tensor (x or zq)
    float *outputPipe; // Pointer to output tensor (x, zq, or logits)
    float *logitsPipe; // Only valid for last node

private:
    float *tokenPipe;
    float *positionPipe;
    SimpleLlmHeader *header;
    NnNetExecution *execution;
    NnExecutor *executor;
    
    int startSegment;
    int endSegment;
    NnUint zqPipeIndex;

public:
    HPipeLlmInference(SimpleLlmNet *net, NnNetExecution *execution, NnExecutor *executor, int startSegment, int endSegment);
    void setBatchSize(NnUint batchSize);
    void setPosition(NnUint position);
    void setToken(NnUint batchIndex, NnUint token);
    void forward();
};

#endif
