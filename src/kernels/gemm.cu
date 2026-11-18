#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"

// ============================================================
//  GEMM: C[M,N] = A[M,K] × B[K,N]
//
//  All matrices are row-major, FP16 storage.
//  Accumulation in FP32 for numerical stability.
//
//  Three variants showing the optimization journey:
//    4a. Naive:  1 thread → 1 output element
//    4b. Tiled:  shared memory blocking
//    4c. wmma:   Tensor Core via nvcuda::wmma
// ============================================================

// ============================================================
//  4a: Naive GEMM
//
//  Each thread computes one element of C:
//    C[row][col] = Σ_k A[row][k] * B[k][col]
//
//  Grid:  (cdiv(N, 32), cdiv(M, 32))
//  Block: (32, 32)
//
//  Performance: ~2% of peak — memory-bound due to no reuse.
//  Every thread reads an entire row of A and column of B
//  from global memory for each output element.
// ============================================================

__global__ void gemm_naive_kernel(
    half*       C,
    const half* A,
    const half* B,
    int M, int N, int K
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = __half2float(A[row * K + k]);
        float b = __half2float(B[k * N + col]);
        acc += a * b;
    }

    C[row * N + col] = __float2half(acc);
}

void launch_gemm_naive(half* C, const half* A, const half* B,
                        int M, int N, int K) {
    dim3 block(32, 32);
    dim3 grid(cdiv(N, 32), cdiv(M, 32));
    gemm_naive_kernel<<<grid, block>>>(C, A, B, M, N, K);
}

