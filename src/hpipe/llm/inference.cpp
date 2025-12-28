#include "hpipe/llm/inference.hpp"
#include <stdexcept>

// ==================================================================================
// HPipeLlmInference Implementation
// ==================================================================================

HPipeLlmInference::HPipeLlmInference(LlmNet *net, NnNetExecution *execution, NnExecutor *executor, int startSegment, int endSegment) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[net->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->execution = execution;
    this->executor = executor;
    this->startSegment = startSegment;
    this->endSegment = endSegment;

    int maxSegment = 2 * header->nLayers + 1;
    if (this->endSegment == -1) this->endSegment = maxSegment;

    // Determine Input/Output Pipes
    // zqPipeIndex is assumed to be 4 based on llm-builder logic (POS, TOK, X, LG, ZQ)
    // If not, we should look it up from netConfig if possible, but here we hardcode or pass it.
    // For now, assuming standard build order:
    this->zqPipeIndex = 4; 

    // Input Pipe Logic
    if (this->startSegment == 0) {
        // Embedding segment takes TOK pipe as input, but its output is X pipe.
        // If startSegment == 0, we don't need `inputPipe` for receiving tensors.
        this->inputPipe = nullptr; 
    } else if (this->startSegment == 1) {
        // Segment 1 (Layer 0 Attn) expects input from Embedding.
        // Embedding writes to X pipe.
        this->inputPipe = (float *)execution->pipes[net->xPipeIndex];
    } else {
        // Other segments (FFN, later Layers) expect input from ZQ pipe (previous layer/block output).
        this->inputPipe = (float *)execution->pipes[zqPipeIndex];
    }

    // Output Pipe Logic
    if (this->endSegment == maxSegment) {
        // Classifier writes to LG pipe (Logits).
        this->outputPipe = (float *)execution->pipes[net->logitsPipeIndex];
    } else if (this->endSegment == 0) {
        // Embedding writes to X pipe.
        this->outputPipe = (float *)execution->pipes[net->xPipeIndex];
    } else {
        // Layers write to ZQ pipe.
        this->outputPipe = (float *)execution->pipes[zqPipeIndex];
    }
}

void HPipeLlmInference::setBatchSize(NnUint batchSize) {
    execution->setBatchSize(batchSize);
}

void HPipeLlmInference::setPosition(NnUint position) {
    if (position + execution->batchSize - 1 >= header->seqLen) {
        throw std::runtime_error("Position exceeds sequence length");
    }
    for (NnUint i = 0; i < execution->batchSize; i++)
        positionPipe[i] = (float)(position + i);
}

void HPipeLlmInference::setToken(NnUint batchIndex, NnUint token) {
    if (batchIndex >= execution->batchSize) {
         throw std::runtime_error("Batch index out of bounds");
    }
    tokenPipe[batchIndex] = (float)token;
}

void HPipeLlmInference::forward() {
    executor->forward();
}
