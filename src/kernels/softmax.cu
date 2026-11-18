#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"
#include <cfloat>

// ============================================================
//  Softmax (in-place, FP32)
//
//  For the final logits layer: input is FP32 [rows, cols]
//  where cols = vocab_size (128256).
//
//  Three-pass approach with block reductions:
//    1. Find row max (block_reduce_max)
//    2. Compute exp(x - max) and accumulate sum (block_reduce_sum)
//    3. Divide by sum
//
//  One block per row. Using FP32 throughout because logits
//  are already in FP32 at sampling time.
//
//  For attention-internal softmax, Flash Attention uses
//  online softmax instead — this kernel is only for the
//  final vocab-size softmax.
// ============================================================

__global__ void softmax_kernel(
    float* x,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float* x_row = x + (long long)row * cols;

    // Pass 1: find max
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x_row[i]);
    }
    float row_max = block_reduce_max(local_max);

    // Broadcast max via shared memory
    __shared__ float s_max;
    if (threadIdx.x == 0) {
        s_max = row_max;
    }
    __syncthreads();
    row_max = s_max;

    // Pass 2: exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = expf(x_row[i] - row_max);
        x_row[i] = val;  // store exp result in-place
        local_sum += val;
    }
    float row_sum = block_reduce_sum(local_sum);

    // Broadcast sum
    __shared__ float s_sum;
    if (threadIdx.x == 0) {
        s_sum = row_sum;
    }
    __syncthreads();
    row_sum = s_sum;

    // Pass 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        x_row[i] *= inv_sum;
    }
}

void launch_softmax(float* x, int rows, int cols) {
    // vocab_size=128256 needs many iterations per thread.
    // Use 1024 threads to maximize parallelism within a block.
    // 128256 / 1024 ≈ 125 elements per thread — fine for stride loop.
    constexpr int BLOCK = 1024;
    softmax_kernel<<<rows, BLOCK>>>(x, rows, cols);
}
