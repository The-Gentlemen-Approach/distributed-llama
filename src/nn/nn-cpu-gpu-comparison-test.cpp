#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include "nn-config-builder.hpp"
#include "nn-quants.hpp"
#include "nn-cpu.hpp"

#ifdef DLLAMA_VULKAN
#include "nn-vulkan.hpp"
#endif

#define N_BATCHES 4
#define SEED 42

// Helper to print array
void printArray(const char *name, float *arr, int size, int limit = 10) {
    printf("%s (first %d of %d): ", name, std::min(limit, size), size);
    for (int i = 0; i < std::min(limit, size); i++) {
        printf("%.6f ", arr[i]);
    }
    printf("\n");
}

// Helper to compare two float arrays
bool compareArrays(const char *testName, float *cpu, float *gpu, int size, float tolerance, bool verbose = false) {
    int errors = 0;
    float maxDiff = 0.0f;
    int maxDiffIdx = -1;

    for (int i = 0; i < size; i++) {
        float diff = fabs(cpu[i] - gpu[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = i;
        }
        if (diff > tolerance) {
            errors++;
            if (errors <= 10 || verbose) { // Print first 10 errors
                printf("  ❌ [%d] CPU=%.6f, GPU=%.6f, diff=%.6f\n", i, cpu[i], gpu[i], diff);
            }
        }
    }

    if (errors == 0) {
        printf("✅ %s PASSED (max diff: %.6e at idx %d)\n", testName, maxDiff, maxDiffIdx);
        return true;
    } else {
        printf("❌ %s FAILED: %d/%d errors, max diff: %.6e at idx %d (CPU=%.6f, GPU=%.6f)\n",
               testName, errors, size, maxDiff, maxDiffIdx, cpu[maxDiffIdx], gpu[maxDiffIdx]);
        return false;
    }
}

#ifdef DLLAMA_VULKAN

// Generic test executor for CPU vs GPU comparison
void executeComparison(
    const char *testName,
    void (*build)(NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder),
    void (*setupInputs)(NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice),
    void (*verify)(NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice),
    NnUint nBatches = N_BATCHES  // Allow override for tests with nZ > 1
) {
    printf("\n========================================\n");
    printf("Testing: %s\n", testName);
    printf("========================================\n");

    // Build network configuration
    NnUint nNodes = 1;
    NnNetConfigBuilder netBuilder(nNodes, nBatches);
    NnNodeConfigBuilder nodeBuilder(0);
    NnSegmentConfigBuilder segmentBuilder;
    build(&netBuilder, &nodeBuilder, &segmentBuilder);
    nodeBuilder.addSegment(segmentBuilder.build());

    NnNetConfig netConfig = netBuilder.build();
    NnNodeConfig nodeConfig = nodeBuilder.build();
    std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
    std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

    // CPU execution
    NnNetExecution cpuExecution(1, &netConfig);
    NnCpuDevice *cpuDevice = new NnCpuDevice(&netConfig, &nodeConfig, &cpuExecution);
    std::vector<NnExecutorDevice> cpuDevices;
    cpuDevices.push_back(NnExecutorDevice(cpuDevice, -1, -1));
    NnFakeNodeSynchronizer cpuSync;
    NnExecutor cpuExecutor(&netConfig, &nodeConfig, &cpuDevices, &cpuExecution, &cpuSync, false);

    // GPU execution
    NnNetExecution gpuExecution(1, &netConfig);
    NnUint gpuIndex = 0;
    NnVulkanDevice *gpuDevice = new NnVulkanDevice(gpuIndex, &netConfig, &nodeConfig, &gpuExecution);
    std::vector<NnExecutorDevice> gpuDevices;
    gpuDevices.push_back(NnExecutorDevice(gpuDevice, -1, -1));
    NnFakeNodeSynchronizer gpuSync;
    NnExecutor gpuExecutor(&netConfig, &nodeConfig, &gpuDevices, &gpuExecution, &gpuSync, false);

    // Setup inputs (same for both CPU and GPU)
    setupInputs(&cpuExecution, &cpuExecutor, cpuDevice, gpuDevice);
    setupInputs(&gpuExecution, &gpuExecutor, cpuDevice, gpuDevice);

    // Execute both
    printf("⚙️  Executing on CPU...\n");
    cpuExecutor.forward();
    printf("⚙️  Executing on GPU...\n");
    gpuExecutor.forward();

    // Verify results
    verify(&cpuExecution, &gpuExecution, cpuDevice, gpuDevice);
}

// Test 1: RMS Norm
template <NnUint dim>
void testRmsNorm_F32_F32_F32() {
    #define TEST_RMS_NORM_EPS 1e-5f
    char testName[256];
    snprintf(testName, sizeof(testName), "RmsNorm_F32_F32_F32 (dim=%u)", dim);

    executeComparison(
        testName,
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            NnUint invRmsBufferIndex = nodeBuilder->addBuffer("inv_rms", size2D(F_32, N_BATCHES, 1));
            segmentBuilder->addOp(OP_INV_RMS, "inv_rms", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{TEST_RMS_NORM_EPS, 1});
            segmentBuilder->addOp(OP_RMS_NORM, "rms_norm", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size1D(F_32, dim),
                NnRmsNormOpConfig{invRmsBufferIndex, 1});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            const NnUint batchSize = 2;
            execution->setBatchSize(batchSize);

            // Create deterministic weights
            std::vector<float> normWeight(dim);
            for (NnUint i = 0; i < dim; i++)
                normWeight[i] = (0.25f + (float)i) / (float)dim;
            executor->loadWeight("rms_norm", 0u, 0u, normWeight.size() * sizeof(float), (NnByte *)normWeight.data());

            // Create deterministic input
            float *xPipe = (float *)execution->pipes[0];
            for (NnUint b = 0; b < batchSize; b++) {
                float *xBatchPipe = &xPipe[b * dim];
                for (NnUint i = 0; i < dim; i++) {
                    float u = (float)(dim - i + b) / (float)(dim / 2);
                    xBatchPipe[i] = u;
                }
            }
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            const NnUint batchSize = 2;

            // Compare inv_rms buffer
            std::vector<float> cpuInvRms(N_BATCHES);
            std::vector<float> gpuInvRms(N_BATCHES);
            std::memcpy(cpuInvRms.data(), cpuDevice->buffers[0], batchSize * sizeof(float));
            gpuDevice->data.buffers[0].get()->read((NnByte *)gpuInvRms.data());

            printf("Comparing inv_rms buffer:\n");
            printArray("  CPU inv_rms", cpuInvRms.data(), batchSize);
            printArray("  GPU inv_rms", gpuInvRms.data(), batchSize);
            compareArrays("  inv_rms", cpuInvRms.data(), gpuInvRms.data(), batchSize, 0.000002f);

            // Compare output pipe
            float *cpuOutput = (float *)cpuExecution->pipes[0];
            std::vector<float> gpuOutput(N_BATCHES * dim);
            gpuDevice->data.pipes[0].get()->read((NnByte *)gpuOutput.data());

            printf("Comparing output pipe:\n");
            printArray("  CPU output", cpuOutput, batchSize * dim);
            printArray("  GPU output", gpuOutput.data(), batchSize * dim);
            compareArrays("  output", cpuOutput, gpuOutput.data(), batchSize * dim, 0.000002f);
        }
    );
}

// Test 2: SILU
template <NnUint dim>
void testSilu_F32_F32() {
    char testName[256];
    snprintf(testName, sizeof(testName), "Silu_F32_F32 (dim=%u)", dim);

    executeComparison(
        testName,
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            segmentBuilder->addOp(OP_SILU, "silu", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnSiluOpCodeConfig{});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];

            for (NnUint b = 0; b < N_BATCHES; b++) {
                const NnUint offset = b * dim;
                for (NnUint i = 0; i < dim; i++) {
                    const float v = i / (float)dim + (float)(b + 1);
                    xPipe[offset + i] = v;
                }
            }
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            float *cpuOutput = (float *)cpuExecution->pipes[0];
            std::vector<float> gpuOutput(N_BATCHES * dim);
            gpuDevice->data.pipes[0].get()->read((NnByte *)gpuOutput.data());

            printArray("  CPU output", cpuOutput, N_BATCHES * dim);
            printArray("  GPU output", gpuOutput.data(), N_BATCHES * dim);
            compareArrays("  output", cpuOutput, gpuOutput.data(), N_BATCHES * dim, 0.00001f);
        }
    );
}

// Test 3: Matmul F32 x F32 -> F32
template <NnUint N, NnUint D>
void testMatmul_F32_F32_F32() {
    char testName[256];
    snprintf(testName, sizeof(testName), "Matmul_F32_F32_F32 (N=%u, D=%u)", N, D);

    executeComparison(
        testName,
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, N));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size2D(F_32, N_BATCHES, D));
            NnUint nullBufferIndex = nodeBuilder->addBuffer("null", size1D(F_32, 1u));
            segmentBuilder->addOp(
                OP_MATMUL, "matmul", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, yPipeIndex),
                size2D(F_32, N, D),
                NnMatmulOpConfig{0u, 0u, nullBufferIndex});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];

            float weight[N * D];
            for (NnUint i = 0; i < N_BATCHES * N; i++)
                xPipe[i] = i * 0.0001f;
            for (NnUint i = 0; i < N * D; i++)
                weight[i] = i * 0.000001f;
            executor->loadWeight("matmul", 0u, 0u, N * D * sizeof(float), (NnByte *)weight);
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            float *cpuOutput = (float *)cpuExecution->pipes[1];
            std::vector<float> gpuOutput(N_BATCHES * D);
            gpuDevice->data.pipes[1].get()->read((NnByte *)gpuOutput.data());

            printArray("  CPU output", cpuOutput, N_BATCHES * D);
            printArray("  GPU output", gpuOutput.data(), N_BATCHES * D);
            compareArrays("  output", cpuOutput, gpuOutput.data(), N_BATCHES * D, 0.0002f, true);
        }
    );
}

// Test 4: Embedding
void testEmbedding_F32_F32() {
    #define EMBEDDING_DIM 16
    #define EMBEDDING_LEN 8

    executeComparison(
        "Embedding_F32_F32",
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, EMBEDDING_DIM));
            segmentBuilder->addOp(OP_EMBEDDING, "embedding", 0,
                pointerBatchConfig(SRC_PIPE, posPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size2D(F_32, EMBEDDING_LEN, EMBEDDING_DIM),
                NnEmbeddingOpConfig{});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            execution->setBatchSize(N_BATCHES);

            float embedding[EMBEDDING_DIM * EMBEDDING_LEN];
            for (NnUint l = 0; l < EMBEDDING_LEN; l++) {
                for (NnUint i = 0; i < EMBEDDING_DIM; i++)
                    embedding[l * EMBEDDING_DIM + i] = (float)(l + 4);
            }
            float *posPipe = (float *)execution->pipes[0];
            for (NnUint b = 0; b < N_BATCHES; b++)
                posPipe[b] = (float)b;

            executor->loadWeight("embedding", 0u, 0u, EMBEDDING_DIM * EMBEDDING_LEN * sizeof(float), (NnByte *)embedding);
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            float *cpuOutput = (float *)cpuExecution->pipes[1];
            std::vector<float> gpuOutput(N_BATCHES * EMBEDDING_DIM);
            gpuDevice->data.pipes[1].get()->read((NnByte *)gpuOutput.data());

            printArray("  CPU output", cpuOutput, N_BATCHES * EMBEDDING_DIM);
            printArray("  GPU output", gpuOutput.data(), N_BATCHES * EMBEDDING_DIM);
            compareArrays("  output", cpuOutput, gpuOutput.data(), N_BATCHES * EMBEDDING_DIM, 0.00001f);
        }
    );
}

// Test 5: Softmax
template <NnUint dim, NnUint nZ>
void testSoftmax_F32_F32() {
    char testName[256];
    snprintf(testName, sizeof(testName), "Softmax_F32_F32 (dim=%u, nZ=%u)", dim, nZ);

    executeComparison(
        testName,
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            // Flatten Z into batch dimension: size2D(F_32, nZ * N_BATCHES, dim)
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, nZ * N_BATCHES, dim));
            segmentBuilder->addOp(OP_SOFTMAX, "softmax", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnSoftmaxOpCodeConfig{});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            // Softmax processes all Z*batch elements as flattened batches
            execution->setBatchSize(nZ * N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];

            // Flatten: treat nZ * N_BATCHES as total batch count
            for (NnUint b = 0; b < nZ * N_BATCHES; b++) {
                const NnUint offset = b * dim;
                for (NnUint i = 0u; i < dim; i++)
                    xPipe[offset + i] = i / (float)dim + (float)(b % N_BATCHES);
            }
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            float *cpuOutput = (float *)cpuExecution->pipes[0];
            std::vector<float> gpuOutput(nZ * N_BATCHES * dim);
            gpuDevice->data.pipes[0].get()->read((NnByte *)gpuOutput.data());

            printArray("  CPU output", cpuOutput, nZ * N_BATCHES * dim);
            printArray("  GPU output", gpuOutput.data(), nZ * N_BATCHES * dim);
            compareArrays("  output", cpuOutput, gpuOutput.data(), nZ * N_BATCHES * dim, 0.00001f);
        },
        nZ * N_BATCHES  // Pass total batch count for nZ > 1 tests
    );
}

// Test 6: ROPE (Rotary Position Embedding)
void testRope_F32_F32() {
    #define ROPE_DIM 64
    #define ROPE_KV_DIM 16

    const char *testName = "Rope_F32_F32 (Llama)";

    executeComparison(
        testName,
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint nHeads = 4;
            const NnUint seqLen = 128;
            const NnRopeSlice slice = sliceRope(ROPE_LLAMA, ROPE_DIM, ROPE_KV_DIM, 8, 1, seqLen, ROPE_DIM / nHeads, 500000.0f, 0);

            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, ROPE_DIM));
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint ropeCacheBufferIndex = nodeBuilder->addBuffer("ropeCache", slice.cacheSize);
            NnUint isQ = 1;

            segmentBuilder->addOp(
                OP_ROPE, "rope_llama", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnRopeOpConfig{ROPE_LLAMA, isQ, posPipeIndex, ropeCacheBufferIndex, 32.0f, 1.0f, 4.0f, 8192, slice});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            execution->setBatchSize(2);

            float *xPipe = (float *)execution->pipes[0];
            float *posPipe = (float *)execution->pipes[1];

            posPipe[0] = 6.0f;
            posPipe[1] = 31.0f;

            for (NnUint b = 0; b < 2; b++) {
                for (NnUint i = 0; i < ROPE_DIM; i++)
                    xPipe[b * ROPE_DIM + i] = 1.0f;
            }

            // Note: RopeCache is automatically initialized:
            // - CPU: in NnCpuDevice::createSegment() via opInit
            // - GPU: in NnVulkanDeviceData constructor
            // We only need to set input pipes here

            // GPU needs explicit write for position pipe
            float pos[N_BATCHES];
            pos[0] = 6.0f;
            pos[1] = 31.0f;
            gpuDevice->data.pipes[1].get()->write((NnByte *)pos);
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            float *cpuOutput = (float *)cpuExecution->pipes[0];
            std::vector<float> gpuOutput(2 * ROPE_DIM);
            gpuDevice->data.pipes[0].get()->read((NnByte *)gpuOutput.data());

            printArray("  CPU output", cpuOutput, 2 * ROPE_DIM, 20);
            printArray("  GPU output", gpuOutput.data(), 2 * ROPE_DIM, 20);
            compareArrays("  output", cpuOutput, gpuOutput.data(), 2 * ROPE_DIM, 0.00001f);
        }
    );
}

// Test 7: MUL (Element-wise multiplication)
template <NnUint dim>
void testMul_F32_F32() {
    char testName[256];
    snprintf(testName, sizeof(testName), "Mul_F32_F32 (dim=%u)", dim);

    executeComparison(
        testName,
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            NnUint sBufferIndex = nodeBuilder->addBuffer("s", size2D(F_32, N_BATCHES, dim));
            segmentBuilder->addOp(OP_MUL, "mul", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnMulOpCodeConfig{sBufferIndex});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            execution->setBatchSize(N_BATCHES);

            float *xPipe = (float *)execution->pipes[0];
            float sBuffer[N_BATCHES * dim];
            for (NnUint i = 0; i < N_BATCHES * dim; i++) {
                xPipe[i] = (float)i;
                sBuffer[i] = (i % 8) / 10.0f;
            }

            // CPU: direct memory copy
            std::memcpy(cpuDevice->buffers[0], sBuffer, N_BATCHES * dim * sizeof(float));
            // GPU: use write method
            gpuDevice->data.buffers[0].get()->write((NnByte *)sBuffer);
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            float *cpuOutput = (float *)cpuExecution->pipes[0];
            std::vector<float> gpuOutput(N_BATCHES * dim);
            gpuDevice->data.pipes[0].get()->read((NnByte *)gpuOutput.data());

            printArray("  CPU output", cpuOutput, N_BATCHES * dim);
            printArray("  GPU output", gpuOutput.data(), N_BATCHES * dim);
            compareArrays("  output", cpuOutput, gpuOutput.data(), N_BATCHES * dim, 0.00001f);
        }
    );
}

// Test 8: MERGE_ADD (Residual connection)
void testMergeAdd_F32_F32() {
    #define MERGE_ADD_F32_NODES 2
    #define MERGE_ADD_F32_DIM 64

    const char *testName = "MergeAdd_F32_F32";

    executeComparison(
        testName,
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint zPipeIndex = netBuilder->addPipe("Z", size2D(F_32, N_BATCHES, MERGE_ADD_F32_DIM * MERGE_ADD_F32_NODES));
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, MERGE_ADD_F32_DIM));
            segmentBuilder->addOp(OP_MERGE_ADD, "mergeAdd", 0,
                pointerBatchConfig(SRC_PIPE, zPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnMergeAddOpCodeConfig{});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            execution->setBatchSize(N_BATCHES);

            float *zPipe = (float *)execution->pipes[0];
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint n = 0; n < MERGE_ADD_F32_NODES; n++) {
                    for (NnUint i = 0; i < MERGE_ADD_F32_DIM; i++)
                        zPipe[b * MERGE_ADD_F32_NODES * MERGE_ADD_F32_DIM + n * MERGE_ADD_F32_DIM + i] = (float)(b + 1);
                }
            }
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            float *cpuOutput = (float *)cpuExecution->pipes[1];
            std::vector<float> gpuOutput(N_BATCHES * MERGE_ADD_F32_DIM);
            gpuDevice->data.pipes[1].get()->read((NnByte *)gpuOutput.data());

            printArray("  CPU output", cpuOutput, N_BATCHES * MERGE_ADD_F32_DIM);
            printArray("  GPU output", gpuOutput.data(), N_BATCHES * MERGE_ADD_F32_DIM);
            compareArrays("  output", cpuOutput, gpuOutput.data(), N_BATCHES * MERGE_ADD_F32_DIM, 0.00001f);
        }
    );
}

// Test 9: MultiheadAttention
void testMultiheadAtt_F32_F32() {
    #define MULTIHEAD_ATT_DIM 256

    const char *testName = "MultiheadAtt_F32_F32";

    executeComparison(
        testName,
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint nHeads = 8;
            const NnUint nKvHeads = 4;
            const NnUint headDim = MULTIHEAD_ATT_DIM / nHeads;
            const NnUint seqLen = 1024;
            const NnUint qSliceD0 = 256;
            const NnUint kvDim0 = 128;
            const NnKvCacheSlice kvCacheSlice = sliceKvCache(kvDim0, seqLen, 1);
            const NnMultiHeadAttSlice multiHeadAttSlice = sliceMultiHeadAtt(nHeads, seqLen, 1, N_BATCHES);

            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, MULTIHEAD_ATT_DIM));
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint qBufferIndex = nodeBuilder->addBuffer("q", size2D(F_32, N_BATCHES, qSliceD0));
            NnUint kCacheBufferIndex = nodeBuilder->addBuffer("kCache", kvCacheSlice.keySize);
            NnUint vCacheBufferIndex = nodeBuilder->addBuffer("vCache", kvCacheSlice.valueSize);
            NnUint attCacheBufferIndex = nodeBuilder->addBuffer("attCache", multiHeadAttSlice.attSize);

            segmentBuilder->addOp(
                OP_MULTIHEAD_ATT, "multihead_att", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnMultiHeadAttOpConfig{nHeads, nHeads, nKvHeads, headDim, seqLen, qSliceD0, kvDim0,
                    posPipeIndex, qBufferIndex, kCacheBufferIndex, vCacheBufferIndex, attCacheBufferIndex});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            execution->setBatchSize(N_BATCHES);

            float *xPipe = (float *)execution->pipes[0];
            float *posPipe = (float *)execution->pipes[1];

            for (NnUint b = 0; b < N_BATCHES; b++) {
                posPipe[b] = (float)b;
                for (NnUint i = 0; i < MULTIHEAD_ATT_DIM; i++)
                    xPipe[b * MULTIHEAD_ATT_DIM + i] = (float)(b * 10 + i % 10) * 0.1f;
            }

            // GPU needs explicit write
            float pos[N_BATCHES];
            for (NnUint b = 0; b < N_BATCHES; b++)
                pos[b] = (float)b;
            gpuDevice->data.pipes[1].get()->write((NnByte *)pos);
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            float *cpuOutput = (float *)cpuExecution->pipes[0];
            std::vector<float> gpuOutput(N_BATCHES * MULTIHEAD_ATT_DIM);
            gpuDevice->data.pipes[0].get()->read((NnByte *)gpuOutput.data());

            printArray("  CPU output", cpuOutput, N_BATCHES * MULTIHEAD_ATT_DIM, 20);
            printArray("  GPU output", gpuOutput.data(), N_BATCHES * MULTIHEAD_ATT_DIM, 20);
            compareArrays("  output", cpuOutput, gpuOutput.data(), N_BATCHES * MULTIHEAD_ATT_DIM, 0.0001f);
        }
    );
}

// Test 10: SHIFT (KV cache shift)
template <NnUint dim>
void testShift_F32_F32() {
    char testName[256];
    snprintf(testName, sizeof(testName), "Shift_F32_F32 (dim=%u)", dim);

    executeComparison(
        testName,
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size2D(F_32, 1, N_BATCHES * dim));
            segmentBuilder->addOp(
                OP_SHIFT, "shift", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerRawConfig(SRC_PIPE, yPipeIndex),
                size0(),
                NnShiftOpCodeConfig{posPipeIndex});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            execution->setBatchSize(N_BATCHES);

            float *posPipe = (float *)execution->pipes[0];
            float *xPipe = (float *)execution->pipes[1];

            for (NnUint b = 0; b < N_BATCHES; b++) {
                posPipe[b] = (float)b;
                for (NnUint i = 0; i < dim; i++)
                    xPipe[b * dim + i] = (float)(b * 100 + i);
            }

            // GPU needs explicit write
            float pos[N_BATCHES];
            for (NnUint b = 0; b < N_BATCHES; b++)
                pos[b] = (float)b;
            gpuDevice->data.pipes[0].get()->write((NnByte *)pos);
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            float *cpuOutput = (float *)cpuExecution->pipes[2];
            std::vector<float> gpuOutput(N_BATCHES * dim);
            gpuDevice->data.pipes[2].get()->read((NnByte *)gpuOutput.data());

            printArray("  CPU output", cpuOutput, N_BATCHES * dim);
            printArray("  GPU output", gpuOutput.data(), N_BATCHES * dim);
            compareArrays("  output", cpuOutput, gpuOutput.data(), N_BATCHES * dim, 0.00001f);
        }
    );
}

// Test 11: CAST F32 -> F32 (simple copy)
template <NnUint dim, NnUint nZ>
void testCast_F32_F32() {
    char testName[256];
    snprintf(testName, sizeof(testName), "Cast_F32_F32 (dim=%u, nZ=%u)", dim, nZ);

    executeComparison(
        testName,
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size3D(F_32, nZ, N_BATCHES, dim));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size3D(F_32, nZ, N_BATCHES, dim));
            segmentBuilder->addOp(
                OP_CAST, "cast", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, yPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];

            for (NnUint i = 0; i < nZ * N_BATCHES * dim; i++)
                xPipe[i] = (float)(i + 1) * 0.1f;
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            float *cpuOutput = (float *)cpuExecution->pipes[1];
            std::vector<float> gpuOutput(nZ * N_BATCHES * dim);
            gpuDevice->data.pipes[1].get()->read((NnByte *)gpuOutput.data());

            printArray("  CPU output", cpuOutput, nZ * N_BATCHES * dim);
            printArray("  GPU output", gpuOutput.data(), nZ * N_BATCHES * dim);
            compareArrays("  output", cpuOutput, gpuOutput.data(), nZ * N_BATCHES * dim, 0.00001f);
        }
    );
}

// Test 12: CAST F32 -> Q80 (quantization)
template <NnUint dim, NnUint nZ>
void testCast_F32_Q80() {
    char testName[256];
    snprintf(testName, sizeof(testName), "Cast_F32_Q80 (dim=%u, nZ=%u)", dim, nZ);

    executeComparison(
        testName,
        // Build
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size3D(F_32, nZ, N_BATCHES, dim));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size3D(F_Q80, nZ, N_BATCHES, dim));
            segmentBuilder->addOp(
                OP_CAST, "cast", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, yPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
        },
        // Setup inputs
        [](NnNetExecution *execution, NnExecutor *executor, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];

            for (NnUint i = 0; i < nZ * N_BATCHES * dim; i++)
                xPipe[i] = (float)(i + 1) * 0.5f;
        },
        // Verify
        [](NnNetExecution *cpuExecution, NnNetExecution *gpuExecution, NnCpuDevice *cpuDevice, NnVulkanDevice *gpuDevice) {
            // Dequantize CPU output
            NnBlockQ80 *cpuQ80 = (NnBlockQ80 *)cpuExecution->pipes[1];
            std::vector<float> cpuOutput(nZ * N_BATCHES * dim);
            dequantizeQ80toF32(cpuQ80, cpuOutput.data(), nZ * N_BATCHES * dim, 1, 0);

            // Dequantize GPU output
            std::vector<NnBlockQ80> gpuQ80Raw(nZ * N_BATCHES * dim / Q80_BLOCK_SIZE);
            gpuDevice->data.pipes[1].get()->read((NnByte *)gpuQ80Raw.data());
            std::vector<float> gpuOutput(nZ * N_BATCHES * dim);
            dequantizeQ80toF32(gpuQ80Raw.data(), gpuOutput.data(), nZ * N_BATCHES * dim, 1, 0);

            printArray("  CPU output (dequantized)", cpuOutput.data(), nZ * N_BATCHES * dim);
            printArray("  GPU output (dequantized)", gpuOutput.data(), nZ * N_BATCHES * dim);
            compareArrays("  output", cpuOutput.data(), gpuOutput.data(), nZ * N_BATCHES * dim, 0.01f);  // Higher tolerance for Q80
        }
    );
}

#endif // DLLAMA_VULKAN

void printUsage(const char *programName) {
    printf("Usage: %s [test_name]\n\n", programName);
    printf("Available tests:\n");
    printf("  all           - Run all tests (default)\n");
    printf("  rmsnorm       - RMS Normalization tests\n");
    printf("  silu          - SILU activation tests\n");
    printf("  matmul        - Matrix multiplication tests\n");
    printf("  embedding     - Embedding lookup test\n");
    printf("  softmax       - Softmax tests\n");
    printf("  rope          - Rotary Position Embedding test\n");
    printf("  mul           - Element-wise multiplication tests\n");
    printf("  mergeadd      - Residual connection test\n");
    printf("  multiheadatt  - Multi-head Attention test\n");
    printf("  shift         - KV cache shift tests\n");
    printf("  cast          - Type casting tests (F32<->Q80)\n");
    printf("\nExample:\n");
    printf("  %s rope         # Run only ROPE test\n", programName);
    printf("  %s              # Run all tests\n", programName);
}

void runTest(const char *testName) {
    if (strcmp(testName, "all") == 0 || strcmp(testName, "rmsnorm") == 0) {
        testRmsNorm_F32_F32_F32<32>();
        testRmsNorm_F32_F32_F32<128>();
        testRmsNorm_F32_F32_F32<1024>();
    }

    if (strcmp(testName, "all") == 0 || strcmp(testName, "silu") == 0) {
        testSilu_F32_F32<32>();
        testSilu_F32_F32<128>();
    }

    if (strcmp(testName, "all") == 0 || strcmp(testName, "matmul") == 0) {
        testMatmul_F32_F32_F32<64, 64>();
        testMatmul_F32_F32_F32<128, 64>();
        testMatmul_F32_F32_F32<64, 128>();
    }

    if (strcmp(testName, "all") == 0 || strcmp(testName, "embedding") == 0) {
        testEmbedding_F32_F32();
    }

    if (strcmp(testName, "all") == 0 || strcmp(testName, "softmax") == 0) {
        testSoftmax_F32_F32<64, 1>();
        testSoftmax_F32_F32<128, 2>();
        testSoftmax_F32_F32<64, 4>();
        testSoftmax_F32_F32<256, 8>();
    }

    if (strcmp(testName, "all") == 0 || strcmp(testName, "rope") == 0) {
        testRope_F32_F32();
    }

    if (strcmp(testName, "all") == 0 || strcmp(testName, "mul") == 0) {
        testMul_F32_F32<64>();
        testMul_F32_F32<128>();
    }

    if (strcmp(testName, "all") == 0 || strcmp(testName, "mergeadd") == 0) {
        testMergeAdd_F32_F32();
    }

    if (strcmp(testName, "all") == 0 || strcmp(testName, "multiheadatt") == 0) {
        testMultiheadAtt_F32_F32();
    }

    if (strcmp(testName, "all") == 0 || strcmp(testName, "shift") == 0) {
        testShift_F32_F32<32>();
        testShift_F32_F32<64>();
    }

    if (strcmp(testName, "all") == 0 || strcmp(testName, "cast") == 0) {
        testCast_F32_F32<32, 1>();
        testCast_F32_F32<64, 1>();
        testCast_F32_F32<128, 1>();
        testCast_F32_Q80<64, 1>();   // Q80는 64 이상만 (메모리 정렬)
        testCast_F32_Q80<128, 1>();
    }
}

int main(int argc, char *argv[]) {
    initQuants();

#ifdef DLLAMA_VULKAN
    const char *testName = "all";

    if (argc > 1) {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
        testName = argv[1];
    }

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║         CPU vs GPU Operation Comparison Test             ║\n");
    printf("║                                                           ║\n");
    printf("║  This test compares CPU and GPU results for each op      ║\n");
    printf("║  to identify discrepancies in GPU implementation.        ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("\n");

    if (strcmp(testName, "all") != 0) {
        printf("🎯 Running test: %s\n\n", testName);
    } else {
        printf("🎯 Running all tests\n\n");
    }

    runTest(testName);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║                   Tests Complete!                        ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("\n");

#else
    printf("❌ Error: This test requires Vulkan support.\n");
    printf("   Please compile with DLLAMA_VULKAN=1\n");
    return 1;
#endif

    return 0;
}
