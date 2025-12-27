CXX = g++
CXXFLAGS = -std=c++11 -Werror -Wformat -Werror=format-security 

ifndef TERMUX_VERSION
	CXXFLAGS += -march=native -mtune=native
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

.PHONY: clean dllama

clean:
	$(DELETE_CMD) *.o dllama dllama-* socket-benchmark mmap-buffer-* *-test *.exe

# nn
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
nn-cpu-test: src/nn/nn-cpu-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
nn-cpu-ops-test: src/nn/nn-cpu-ops-test.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
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

# llm
tokenizer.o: src/tokenizer.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
llm.o: src/llm.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
app.o: src/app.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
tokenizer-test: src/tokenizer-test.cpp nn-quants.o nn-core.o llamafile-sgemm.o nn-cpu-ops.o tokenizer.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)
dllama: src/dllama.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)
dllama-api: src/dllama-api.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o llm.o app.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)

# Simple LLM (단일 디바이스, 분산 처리 없음)
# - 워커 분산 로직 제거: loadSimpleLlmWeights로 슬라이싱 없이 전체 가중치 로드
# - 네트워크 통신 없음: SimpleLlmInference에서 제어 패킷 전송 제거
simple-llm.o: src/simple-llm.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
llm-builder.o: src/llm-builder.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
simple-dllama: src/simple-dllama.cpp nn-quants.o nn-core.o nn-executor.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o tokenizer.o simple-llm.o llm-builder.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)

# H-Pipe (파이프라인 병렬화)
# - 선형 파이프라인 토폴로지: Root → Worker1 → Worker2 → ... → Root
# - ACK 기반 역방향 흐름 제어
# - 청크 기반 스트리밍 처리
hpipe-network.o: src/hpipe-network.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@
hpipe-network-test: src/hpipe-network-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o hpipe-network.o simple-llm.o llm-builder.o tokenizer.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

hpipe-pipeline-test: src/hpipe-pipeline-test.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o hpipe-network.o simple-llm.o llm-builder.o tokenizer.o
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LIBS)

# H-Pipe LLM execution
hpipe-llm.o: src/hpipe-llm.cpp
	$(CXX) $(CXXFLAGS) -c $^ -o $@

# H-Pipe Root and Worker binaries
hpipe-worker: src/hpipe-worker.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o hpipe-network.o hpipe-llm.o simple-llm.o llm-builder.o tokenizer.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)

hpipe-root: src/hpipe-root.cpp nn-quants.o nn-core.o nn-executor.o nn-network.o llamafile-sgemm.o nn-cpu-ops.o nn-cpu.o hpipe-network.o simple-llm.o llm-builder.o tokenizer.o ${DEPS}
	$(CXX) $(CXXFLAGS) $(filter-out %.spv, $^) -o $@ $(LIBS)
