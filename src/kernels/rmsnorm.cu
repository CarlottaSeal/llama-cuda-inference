#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"

// ============================================================
//  RMSNorm
//
//  out[i] = x[i] / rms(x) * weight[i]
//  where rms(x) = sqrt(mean(x^2) + eps)
//
//  One block processes one row (one token's hidden vector).
//  Steps:
//    1. Each thread accumulates partial sum-of-squares (FP32)
//    2. block_reduce_sum to get total sum-of-squares
//    3. Thread 0 computes rms_inv = rsqrt(ss/cols + eps),
//       writes to shared memory
//    4. All threads normalize: out[i] = x[i] * rms_inv * w[i]
//
//  Memory-bound: hidden_size=2048 → 4KB per row (FP16)
// ============================================================

__global__ void rmsnorm_kernel(
    half* out,
    const half* x,
    const half* weight,
    int rows,
    int cols,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const half* x_row = x + (long long)row * cols;
    half* out_row = out + (long long)row * cols;

    // Step 1: partial sum of squares
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        ss += val * val;
    }

    // Step 2: block-level reduction
    ss = block_reduce_sum(ss);

    // Step 3: compute rms_inv, broadcast via shared memory
    __shared__ float s_rms_inv;
    if (threadIdx.x == 0) {
        s_rms_inv = rsqrtf(ss / (float)cols + eps);
    }
    __syncthreads();

    float rms_inv = s_rms_inv;

    // Step 4: normalize and scale
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        float w = __half2float(weight[i]);
        out_row[i] = __float2half(val * rms_inv * w);
    }
}

void launch_rmsnorm(half* out, const half* x, const half* weight,
                     int rows, int cols, float eps) {
    // Use 256 threads per block.
    // For hidden_size=2048, each thread handles 2048/256 = 8 elements.
    constexpr int BLOCK = 256;
    rmsnorm_kernel<<<rows, BLOCK>>>(out, x, weight, rows, cols, eps);
}
