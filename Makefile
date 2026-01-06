CXX = g++
CXXFLAGS = -std=c++11 -Werror -Wformat -Werror=format-security -I src

UNAME_M := $(shell uname -m)
ifndef TERMUX_VERSION
	ifeq ($(UNAME_M),x86_64)
		CXXFLAGS += -mavx -mavx2 -mfma -mf16c
	else
		CXXFLAGS += -march=native -mtune=native
	endif
endif

ifdef DEBUG
	CXXFLAGS += -g -fsanitize=address
else
	CXXFLAGS += -O3
endif

ifdef WVLA
	CXXFLAGS += -Wvla-extension
endif

ifdef DLLAMA_VULKAN
	CGLSLC = glslc

ifeq ($(OS),Windows_NT)
	LIBS += -L$(VK_SDK_PATH)\lib -lvulkan-1
	CXXFLAGS += -DDLLAMA_VULKAN -I$(VK_SDK_PATH)\include
else
	LIBS += -lvulkan
	CXXFLAGS += -DDLLAMA_VULKAN
endif

	DEPS += nn-vulkan.o
endif

ifeq ($(OS),Windows_NT)
    LIBS += -lws2_32
	DELETE_CMD = del /f
else
    LIBS += -lpthread
    DELETE_CMD = rm -fv
endif

.PHONY: clean hpipe-root hpipe-worker simple-dllama

clean:
	$(DELETE_CMD) *.o hpipe-root hpipe-worker simple-dllama *-test *.exe

# ==========================================
# NN Core
# ==========================================
nn-quants.o: src/nn/nn-quants.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-core.o: src/nn/nn-core.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-executor.o: src/nn/nn-executor.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-network.o: src/nn/nn-network.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
llamafile-sgemm.o: src/nn/llamafile/sgemm.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-cpu-ops.o: src/nn/nn-cpu-ops.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
nn-cpu.o: src/nn/nn-cpu.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# Tests
nn-cpu-test: src/nn/nn-cpu-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-cpu-ops-test: src/nn/nn-cpu-ops-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

# Vulkan
nn-vulkan.o: src/nn/nn-vulkan.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

ifdef DLLAMA_VULKAN
VULKAN_SHADER_SRCS := $(wildcard src/nn/vulkan/*.comp)
VULKAN_SHADER_BINS := $(VULKAN_SHADER_SRCS:.comp=.spv)
DEPS += $(VULKAN_SHADER_BINS)

%.spv: %.comp
	$(CGLSLC) -c $< -o $@ --target-env=vulkan1.2
nn-vulkan-test: src/nn/nn-vulkan-test.cpp nn-quants.o nn-core.o nn-executor.o nn-vulkan.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)
endif

# ==========================================
# Common
# ==========================================
llm-types.o: src/common/llm-types.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
llm-builder.o: src/common/llm-builder.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
tokenizer.o: src/common/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# ==========================================
# Simple LLM
# ==========================================
simple-inference.o: src/simple/inference.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
simple-network-builder.o: src/simple/network-builder.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
simple-weight-loader.o: src/simple/weight-loader.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

SIMPLE_OBJS = simple-inference.o simple-network-builder.o simple-weight-loader.o
COMMON_OBJS = llm-types.o llm-builder.o tokenizer.o
NN_OBJS = nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o

simple-dllama: src/simple/main.cpp $(SIMPLE_OBJS) $(COMMON_OBJS) $(NN_OBJS) ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)

# ==========================================
# H-Pipe
# ==========================================
hpipe-network-base.o: src/hpipe/network/base.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
hpipe-network-root.o: src/hpipe/network/root.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
hpipe-network-worker.o: src/hpipe/network/worker.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

hpipe-llm-network-builder.o: src/hpipe/llm/network-builder.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
hpipe-llm-weight-loader.o: src/hpipe/llm/weight-loader.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
hpipe-llm-inference.o: src/hpipe/llm/inference.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

HPIPE_NET_OBJS = hpipe-network-base.o hpipe-network-root.o hpipe-network-worker.o
HPIPE_LLM_OBJS = hpipe-llm-network-builder.o hpipe-llm-weight-loader.o hpipe-llm-inference.o

hpipe-root: src/hpipe/main/root.cpp $(HPIPE_NET_OBJS) $(HPIPE_LLM_OBJS) $(COMMON_OBJS) $(NN_OBJS) nn-network.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)

hpipe-worker: src/hpipe/main/worker.cpp $(HPIPE_NET_OBJS) $(HPIPE_LLM_OBJS) $(COMMON_OBJS) $(NN_OBJS) nn-network.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)

# Tests
hpipe-network-test: src/hpipe/tests/network-test.cpp $(HPIPE_NET_OBJS) $(COMMON_OBJS) $(NN_OBJS) nn-network.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

hpipe-pipeline-test: src/hpipe/tests/pipeline-test.cpp $(HPIPE_NET_OBJS) $(HPIPE_LLM_OBJS) $(COMMON_OBJS) $(NN_OBJS) nn-network.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)