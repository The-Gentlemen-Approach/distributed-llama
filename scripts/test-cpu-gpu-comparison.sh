#!/bin/bash

# CPU vs GPU Operation Comparison Test Script
# This script compares CPU and GPU results for all operations to debug GPU issues

set -e

# Ensure we are running from the project root
cd "$(dirname "$0")/.."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         CPU vs GPU Operation Comparison Test             ║"
echo "║                                                           ║"
echo "║  This test compares CPU and GPU results for each op      ║"
echo "║  to identify discrepancies in GPU implementation.        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}\n"

# Check if Vulkan is compiled
if [ ! -f "./nn-cpu-gpu-comparison-test" ]; then
    echo -e "${YELLOW}⚠️  Binary not found. Building with Vulkan support...${NC}"
    echo ""

    # Clean and build with Vulkan
    make clean
    echo ""
    echo -e "${GREEN}🔨 Building nn-cpu-gpu-comparison-test with DLLAMA_VULKAN=1...${NC}"
    make nn-cpu-gpu-comparison-test DLLAMA_VULKAN=1

    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Build failed!${NC}"
        echo "   Make sure you have Vulkan SDK installed."
        echo "   On macOS: Install MoltenVK"
        echo "   On Linux: sudo apt install vulkan-sdk"
        exit 1
    fi
    echo ""
fi

# Run the comparison test
echo -e "${GREEN}🚀 Running CPU vs GPU comparison test...${NC}\n"
echo "=========================================="
./nn-cpu-gpu-comparison-test 2>&1 | tee cpu-gpu-comparison.log

EXIT_CODE=$?
echo ""
echo "=========================================="
echo ""

# Check exit code
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Test completed!${NC}"
    echo -e "\n${BLUE}📊 Full log saved to: cpu-gpu-comparison.log${NC}"
    echo -e "${BLUE}📊 Check the log for detailed comparison results${NC}"
else
    echo -e "${RED}❌ Test failed with exit code $EXIT_CODE${NC}"
    echo -e "\n${BLUE}📊 Log saved to: cpu-gpu-comparison.log${NC}"
    exit $EXIT_CODE
fi

echo ""
echo -e "${CYAN}💡 Tips:${NC}"
echo "   - Look for ❌ marks in the output to find failing operations"
echo "   - Check 'max diff' values to see the magnitude of errors"
echo "   - GPU errors often indicate issues in Vulkan shader code"
echo "   - Compare CPU and GPU array values to debug specific operations"
echo ""
