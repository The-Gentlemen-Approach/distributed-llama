#!/bin/bash

# Simple-DLLama Test Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Simple-DLLama Test ===${NC}\n"

# Clean old logs
rm -f simple.log

# Configuration
MODEL_PATH="models/llama3_2_1b_instruct_q40/dllama_model_llama3_2_1b_instruct_q40.m"
TOKENIZER_PATH="models/llama3_2_1b_instruct_q40/dllama_tokenizer_llama3_2_1b_instruct_q40.t"
PROMPT="${1:-Hello my name is Gyo. And I will tell you how to do gentlemen approach. Gentlemen Approach is a team name. Gentlemen Approach was decided as team name because our teammates love the album which name is A gentleman approach. As an team name, we will do approach for paper very gentle.}"
NTHREADS=4
STEPS=100
TEMPERATURE=0.8
TOPP=0.9
BUFFER_TYPE=q80

echo -e "${GREEN}📋 Configuration:${NC}"
echo "  Model: $MODEL_PATH"
echo "  Tokenizer: $TOKENIZER_PATH"
echo "  Prompt: \"${PROMPT:0:80}...\""
echo "  Threads: $NTHREADS"
echo "  Steps: $STEPS"
echo "  Temperature: $TEMPERATURE"
echo "  Top-p: $TOPP"
echo "  Buffer type: $BUFFER_TYPE"
echo ""

# Check if binaries exist
if [ ! -f "./simple-dllama" ]; then
    echo -e "${RED}❌ simple-dllama binary not found!${NC}"
    echo "Please run 'make simple-dllama' first"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Model file not found: $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -f "$TOKENIZER_PATH" ]; then
    echo -e "${RED}❌ Tokenizer file not found: $TOKENIZER_PATH${NC}"
    exit 1
fi

# Run inference
echo -e "${GREEN}🚀 Starting inference...${NC}\n"
echo "=========================================="
./simple-dllama \
    --model "$MODEL_PATH" \
    --tokenizer "$TOKENIZER_PATH" \
    --prompt "$PROMPT" \
    --nthreads $NTHREADS \
    --steps $STEPS \
    --temperature $TEMPERATURE \
    --topp $TOPP \
    --buffer-float-type $BUFFER_TYPE 2>&1 | tee simple.log

EXIT_CODE=$?
echo ""
echo "=========================================="
echo ""

# Check exit code
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Test completed successfully!${NC}"
    echo -e "\n${BLUE}📊 Log saved to: simple.log${NC}"
else
    echo -e "${RED}❌ Test failed with exit code $EXIT_CODE${NC}"
    exit $EXIT_CODE
fi
