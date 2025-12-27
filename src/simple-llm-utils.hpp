#ifndef SIMPLE_LLM_UTILS_HPP
#define SIMPLE_LLM_UTILS_HPP

#include "nn/nn-core.hpp"

/**
 * Single-node utility functions for SimpleLlm.
 * These helpers initialize matmul slices without distribution (nNodes=1).
 */

/**
 * Initialize row matmul slice for single-node (no slicing).
 */
inline NnRowMatmulSlice initSingleNodeRowMatmulSlice(NnFloatType weightType, NnUint n, NnUint d) {
    NnRowMatmulSlice slice;
    slice.type = weightType;
    slice.nNodes = 1;
    slice.size = size2D(weightType, n, d);
    slice.sliceSize = slice.size;
    slice.n = n;
    slice.d0 = d;
    return slice;
}

/**
 * Initialize col matmul slice for single-node (no slicing).
 */
inline NnColMatmulSlice initSingleNodeColMatmulSlice(NnFloatType weightType, NnUint n, NnUint d) {
    NnColMatmulSlice slice;
    slice.type = weightType;
    slice.nNodes = 1;
    slice.size = size2D(weightType, n, d);
    slice.sliceSize = size2D(weightType, n, d);  // Single-node: same as size
    slice.n = n;
    slice.n0 = n;  // Single-node: processes all n rows
    slice.d = d;
    return slice;
}

#endif // SIMPLE_LLM_UTILS_HPP
