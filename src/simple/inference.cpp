#include "simple/inference.hpp"
#include <stdexcept>

// ==================================================================================
// SimpleLlmInference Implementation
// ==================================================================================

SimpleLlmInference::SimpleLlmInference(SimpleLlmNet *net, NnNetExecution *execution, NnExecutor *executor) {
    this->header = net->header;
    this->tokenPipe = (float *)execution->pipes[net->tokenPipeIndex];
    this->positionPipe = (float *)execution->pipes[net->positionPipeIndex];
    this->logitsPipe = (float *)execution->pipes[net->logitsPipeIndex];
    this->execution = execution;
    this->executor = executor;
}

void SimpleLlmInference::setBatchSize(NnUint batchSize) {
    execution->setBatchSize(batchSize);
}

void SimpleLlmInference::setPosition(NnUint position) {
    // Basic safety checks
    // assert(position >= 0); // Unsigned is always >= 0
    if (position + execution->batchSize - 1 >= header->seqLen) {
        throw std::runtime_error("Position exceeds sequence length");
    }

    // Set position for each item in batch
    for (NnUint i = 0; i < execution->batchSize; i++)
        positionPipe[i] = (float)(position + i);
}

void SimpleLlmInference::setToken(NnUint batchIndex, NnUint token) {
    if (batchIndex >= execution->batchSize) {
         throw std::runtime_error("Batch index out of bounds");
    }
    tokenPipe[batchIndex] = (float)token;
}

void SimpleLlmInference::forward() {
    executor->forward();
}
