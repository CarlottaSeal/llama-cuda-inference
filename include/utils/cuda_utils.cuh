#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

// ============================================================
//  CUDA error checking
// ============================================================

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

// ============================================================
//  FP16 conversion helpers
// ============================================================

__device__ __forceinline__ float half_to_float(half h) {
    return __half2float(h);
}

__device__ __forceinline__ half float_to_half(float f) {
    return __float2half(f);
}

// ============================================================
//  Integer division, round up
// ============================================================

inline int cdiv(int a, int b) {
    return (a + b - 1) / b;
}

// ============================================================
//  Warp-level reductions (warp size = 32)
//  Uses butterfly shuffle — no shared memory needed.
// ============================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        if (other > val) {
            val = other;
        }
    }
    return val;
}

// ============================================================
//  Block-level reductions
//  Two-phase: intra-warp shuffle → inter-warp via shared memory.
//  Assumes blockDim.x <= 1024 (max 32 warps).
// ============================================================

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // one slot per warp

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces across warps
    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        if (lane < num_warps) {
            val = shared[lane];
        } else {
            val = 0.0f;
        }
        val = warp_reduce_sum(val);
    }

    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float shared[32];

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + 31) / 32;
    if (warp_id == 0) {
        if (lane < num_warps) {
            val = shared[lane];
        } else {
            val = -INFINITY;
        }
        val = warp_reduce_max(val);
    }

    return val;
}
