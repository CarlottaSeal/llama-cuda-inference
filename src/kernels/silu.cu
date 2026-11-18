#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"

// ============================================================
//  SiLU + Multiply (SwiGLU activation)
//
//  out[i] = silu(gate[i]) * up[i]
//  where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
//  Used in FFN block: gate_proj and up_proj outputs are combined
//  before feeding into down_proj.
//
//  Memory-bound, element-wise.
//  Launch: 1D grid, 256 threads/block.
// ============================================================

__global__ void silu_mul_kernel(half* out, const half* gate, const half* up, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = half_to_float(gate[idx]);
        float u = half_to_float(up[idx]);

        // silu(g) = g * sigmoid(g)
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        float silu_g = g * sigmoid_g;

        out[idx] = float_to_half(silu_g * u);
    }
}

void launch_silu_mul(half* out, const half* gate, const half* up, int n) {
    constexpr int BLOCK = 256;
    int grid = cdiv(n, BLOCK);
    silu_mul_kernel<<<grid, BLOCK>>>(out, gate, up, n);
}
