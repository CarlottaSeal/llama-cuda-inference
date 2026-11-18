#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"

// ============================================================
//  Residual Add — out[i] = a[i] + b[i]
//
//  Pure element-wise, memory-bound.
//  Used after attention output projection and after FFN down projection.
//
//  Launch: 1D grid, 256 threads/block.
//  Each thread handles one element.
// ============================================================

__global__ void residual_add_kernel(half* out, const half* a, const half* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = half_to_float(a[idx]);
        float vb = half_to_float(b[idx]);
        out[idx] = float_to_half(va + vb);
    }
}

void launch_residual_add(half* out, const half* a, const half* b, int n) {
    constexpr int BLOCK = 256;
    int grid = cdiv(n, BLOCK);
    residual_add_kernel<<<grid, BLOCK>>>(out, a, b, n);
}
