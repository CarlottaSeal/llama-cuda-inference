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

// ============================================================
//  4c: wmma Tensor Core GEMM
//
//  Uses nvcuda::wmma API for 16×16×16 matrix multiply-accumulate.
//  A warp (32 threads) cooperates on one 16×16 output tile.
//
//  Each warp:
//    - Loads 16×16 fragment of A (row_major, FP16)
//    - Loads 16×16 fragment of B (row_major, FP16)
//    - MMA accumulates into 16×16 fragment of C (FP32)
//    - Stores result as FP16
//
//  Block layout: (WMMA_BLOCK_N/16, WMMA_BLOCK_M/16) warps per block
//  where each warp handles one 16×16 output tile.
//
//  Grid: covers the full M×N output.
// ============================================================

#include <mma.h>
#include <cstdint>
using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block-level tiling: how many wmma tiles per block
constexpr int WMMA_BLOCK_M = 64;  // 4 wmma tiles in M
constexpr int WMMA_BLOCK_N = 64;  // 4 wmma tiles in N
// Warps per block: (64/16) × (64/16) = 4×4 = 16 warps = 512 threads

__global__ void gemm_wmma_kernel(
    half*       C,
    const half* A,
    const half* B,
    int M, int N, int K
) {
    // Which wmma tile this warp is responsible for
    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warps_per_row = WMMA_BLOCK_N / WMMA_N;  // 4
    int warp_row = warp_id / warps_per_row;      // 0..3
    int warp_col = warp_id % warps_per_row;      // 0..3

    int lane = threadIdx.x % 32;

    // Global output tile position
    int tile_row = blockIdx.y * WMMA_BLOCK_M + warp_row * WMMA_M;
    int tile_col = blockIdx.x * WMMA_BLOCK_N + warp_col * WMMA_N;

    // Warp-uniform exit; the kernel has no block-level barriers
    if (tile_row >= M || tile_col >= N) return;

    // Edge tiles (M=1 in decode, ragged prefill lengths, K/N not a multiple
    // of 16, unaligned base pointers) go through per-warp shared memory,
    // zero-padded on load and masked on store. Full tiles read/write global
    // directly.
    constexpr int TILE_ELEMS = WMMA_M * WMMA_K;
    constexpr int WARPS = (WMMA_BLOCK_M / WMMA_M) * (WMMA_BLOCK_N / WMMA_N);
    __shared__ half  sA[WARPS][TILE_ELEMS];
    __shared__ half  sB[WARPS][TILE_ELEMS];
    __shared__ float sC[WARPS][WMMA_M * WMMA_N];

    bool rows_full = tile_row + WMMA_M <= M;
    bool cols_full = tile_col + WMMA_N <= N;
    bool a_direct = rows_full && K % 16 == 0 && (uintptr_t)A % 32 == 0;
    bool b_direct = cols_full && K % 16 == 0 && N % 16 == 0 && (uintptr_t)B % 32 == 0;
    bool c_direct = rows_full && cols_full && N % 16 == 0 && (uintptr_t)C % 32 == 0;

    // Declare wmma fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K in steps of WMMA_K=16
    for (int k = 0; k < K; k += WMMA_K) {
        // A fragment: A[tile_row : tile_row+16, k : k+16], leading dim K
        if (a_direct) {
            wmma::load_matrix_sync(a_frag, A + tile_row * K + k, K);
        } else {
            for (int i = lane; i < TILE_ELEMS; i += 32) {
                int r = tile_row + i / WMMA_K, c = k + i % WMMA_K;
                sA[warp_id][i] = (r < M && c < K) ? A[r * K + c] : __float2half(0.0f);
            }
            __syncwarp();
            wmma::load_matrix_sync(a_frag, sA[warp_id], WMMA_K);
        }

        // B fragment: B[k : k+16, tile_col : tile_col+16], leading dim N
        if (b_direct) {
            wmma::load_matrix_sync(b_frag, B + k * N + tile_col, N);
        } else {
            for (int i = lane; i < TILE_ELEMS; i += 32) {
                int r = k + i / WMMA_N, c = tile_col + i % WMMA_N;
                sB[warp_id][i] = (r < K && c < N) ? B[r * N + c] : __float2half(0.0f);
            }
            __syncwarp();
            wmma::load_matrix_sync(b_frag, sB[warp_id], WMMA_N);
        }

        // Matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncwarp();  // staging buffers are rewritten next iteration
    }

    // Store result as FP16
    if (c_direct) {
        // Convert FP32 accumulator to FP16 output
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_half_frag;
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_half_frag.x[i] = __float2half(c_frag.x[i]);
        }
        wmma::store_matrix_sync(C + tile_row * N + tile_col, c_half_frag, N, wmma::mem_row_major);
    } else {
        wmma::store_matrix_sync(sC[warp_id], c_frag, WMMA_N, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < WMMA_M * WMMA_N; i += 32) {
            int r = tile_row + i / WMMA_N, c = tile_col + i % WMMA_N;
            if (r < M && c < N) C[r * N + c] = __float2half(sC[warp_id][i]);
        }
    }
}

void launch_gemm_wmma(half* C, const half* A, const half* B,
                       int M, int N, int K) {
    // Block: 16 warps (512 threads), arranged as a flat 1D block
    // Each warp handles one 16×16 tile
    dim3 block(512);
    dim3 grid(cdiv(N, WMMA_BLOCK_N), cdiv(M, WMMA_BLOCK_M));
    gemm_wmma_kernel<<<grid, block>>>(C, A, B, M, N, K);
}

// ============================================================
//  4e: Decode GEMV — y[N] = x[K] × W[K,N]   (the M=1 case)
//
//  At M=1 every weight element is used exactly once, so the only
//  thing that matters is streaming W at DRAM bandwidth. The wmma
//  kernel pads the single row to a 16-row tile and stages it
//  through shared memory; this kernel does neither.
//
//  Block: 256 threads = 8 warps, owns 64 consecutive columns.
//    lane → (row group rg = lane/8, column group cg = lane%8)
//    each thread loads one uint4 = 8 halves of W[k, n0+cg*8 ..],
//    so 8 lanes read one full 128-byte line of a row and a warp
//    covers 4 rows; the block covers 32 rows per step.
//  Split-K: narrow outputs (k/v proj: N=512 → 8 column blocks)
//    can't fill the GPU, so gridDim.y splits K into chunks. Each
//    block writes its 64 partial sums to a workspace; the last
//    block to finish a column range (atomic counter) adds the
//    partials in split order and writes FP16. One launch, and the
//    summation order is fixed, so results are deterministic.
//  Requires N % 8 == 0 and 16-byte aligned W (cudaMalloc is).
// ============================================================

constexpr int GEMV_COLS    = 64;                 // columns per block
constexpr int GEMV_THREADS = 256;
constexpr int GEMV_ROWS    = GEMV_THREADS / 8;   // rows per block step (32)
constexpr int GEMV_MAX_PARTIAL_BLOCKS = 1024;    // cap on split * col_blocks when split > 1

__device__ float    g_gemv_ws[GEMV_MAX_PARTIAL_BLOCKS * GEMV_COLS];
__device__ unsigned g_gemv_count[GEMV_MAX_PARTIAL_BLOCKS];

__global__ void __launch_bounds__(GEMV_THREADS)
gemv_kernel(half* __restrict__ y, const half* __restrict__ x,
            const half* __restrict__ W, int N, int K, int chunk) {
    const int tid  = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int rg   = warp * 4 + (lane >> 3);         // 0..31
    const int cg   = lane & 7;                       // 0..7
    const int n0   = blockIdx.x * GEMV_COLS;
    const int col  = n0 + cg * 8;
    const int k0   = blockIdx.y * chunk;
    const int k1   = min(K, k0 + chunk);

    float acc[8] = {};
    if (col < N) {
        #pragma unroll 4
        for (int k = k0 + rg; k < k1; k += GEMV_ROWS) {
            uint4 w = __ldg(reinterpret_cast<const uint4*>(W + (size_t)k * N + col));
            float xk = __half2float(x[k]);
            const half2* w2 = reinterpret_cast<const half2*>(&w);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                float2 f = __half22float2(w2[j]);
                acc[2 * j]     = fmaf(xk, f.x, acc[2 * j]);
                acc[2 * j + 1] = fmaf(xk, f.y, acc[2 * j + 1]);
            }
        }
    }

    // Reduce the 32 row groups: red[rg][col_in_block]
    __shared__ float red[GEMV_ROWS][GEMV_COLS];
    #pragma unroll
    for (int j = 0; j < 8; j++) red[rg][cg * 8 + j] = acc[j];
    __syncthreads();

    float sum = 0.0f;
    if (tid < GEMV_COLS) {
        #pragma unroll 8
        for (int r = 0; r < GEMV_ROWS; r++) sum += red[r][tid];
    }

    const int n = n0 + tid;
    if (gridDim.y == 1) {
        if (tid < GEMV_COLS && n < N) y[n] = __float2half(sum);
        return;
    }

    // Split-K: publish this block's partials, last block finishes
    const int cols_in_grid = gridDim.x * GEMV_COLS;
    if (tid < GEMV_COLS) g_gemv_ws[blockIdx.y * cols_in_grid + n] = sum;
    __threadfence();
    __syncthreads();

    __shared__ bool is_last;
    if (tid == 0) {
        unsigned done = atomicAdd(&g_gemv_count[blockIdx.x], 1u);
        is_last = (done == gridDim.y - 1);
    }
    __syncthreads();
    if (!is_last) return;

    if (tid < GEMV_COLS && n < N) {
        float total = 0.0f;
        for (int s = 0; s < (int)gridDim.y; s++)
            total += __ldcg(&g_gemv_ws[s * cols_in_grid + n]);  // bypass L1: other SMs wrote these
        y[n] = __float2half(total);
    }
    if (tid == 0) g_gemv_count[blockIdx.x] = 0;  // ready for the next launch
}

bool gemv_supported(int N) { return N % 8 == 0; }

void launch_gemv(half* y, const half* x, const half* W, int N, int K) {
    static int target_blocks = 0;
    if (!target_blocks) {
        int dev = 0, sms = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        target_blocks = 4 * sms;
    }
    const int col_blocks = cdiv(N, GEMV_COLS);
    int split = 1;
    if (col_blocks < target_blocks) {
        split = cdiv(target_blocks, col_blocks);
        split = min(split, cdiv(K, GEMV_ROWS));                       // >= one block step each
        split = min(split, GEMV_MAX_PARTIAL_BLOCKS / col_blocks);     // workspace capacity
        split = max(split, 1);
    }
    // Round chunks to whole block steps, then drop empty trailing splits
    int chunk = cdiv(cdiv(K, split), GEMV_ROWS) * GEMV_ROWS;
    split = cdiv(K, chunk);
    dim3 grid(col_blocks, split);
    gemv_kernel<<<grid, GEMV_THREADS>>>(y, x, W, N, K, chunk);
}
