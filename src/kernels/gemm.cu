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

// ============================================================
//  4b: Tiled shared memory GEMM
//
//  Load TILE×TILE sub-blocks of A and B into shared memory
//  to exploit data reuse. Each element of A and B is loaded
//  once per tile instead of once per output element.
//
//  Data reuse: TILE = 32 → 32× fewer global loads vs naive.
//  Still FP16 load → FP32 accumulate → FP16 store.
//
//  Grid:  (cdiv(N, TILE), cdiv(M, TILE))
//  Block: (TILE, TILE) = (32, 32)
// ============================================================

constexpr int TILE = 32;

__global__ void gemm_tiled_kernel(
    half*       C,
    const half* A,
    const half* B,
    int M, int N, int K
) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    // Loop over tiles along the K dimension
    for (int t = 0; t < cdiv(K, TILE); t++) {
        // Collaborative load: each thread loads one element of A and B
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;

        if (row < M && a_col < K) {
            sA[threadIdx.y][threadIdx.x] = __half2float(A[row * K + a_col]);
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (b_row < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = __half2float(B[b_row * N + col]);
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the tile
        for (int k = 0; k < TILE; k++) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}

void launch_gemm_tiled(half* C, const half* A, const half* B,
                        int M, int N, int K) {
    dim3 block(TILE, TILE);
    dim3 grid(cdiv(N, TILE), cdiv(M, TILE));
    gemm_tiled_kernel<<<grid, block>>>(C, A, B, M, N, K);
}

