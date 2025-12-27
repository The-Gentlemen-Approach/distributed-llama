#ifndef HPIPE_LLM_HPP
#define HPIPE_LLM_HPP

#include "nn/nn-core.hpp"
#include "nn/nn-executor.hpp"
#include "simple-llm-utils.hpp"
#include "simple-llm.hpp" // Use common structs from simple-llm

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

#endif // HPIPE_LLM_HPP
