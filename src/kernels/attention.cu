#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"
#include <cfloat>

// ============================================================
//  Attention Kernels with GQA support
//
//  Llama 3.2 1B: 32 Q heads, 8 KV heads → GQA ratio 4:1
//  Q head i uses KV head (i / gqa_ratio) = (i / 4)
//
//  Tensor layouts (all FP16, row-major):
//    Q: [seq_len,  num_q_heads,  head_dim]   — query tokens
//    K: [kv_len,   num_kv_heads, head_dim]   — key (from cache)
//    V: [kv_len,   num_kv_heads, head_dim]   — value (from cache)
//    O: [seq_len,  num_q_heads,  head_dim]   — output
//
//  scale = 1 / sqrt(head_dim)
//
//  Causal mask: position q_pos can attend to k_pos <= q_pos
//  During decode with KV cache: q_pos = start_pos + q_idx
//  For simplicity, we pass kv_len and seq_len and apply:
//    mask: k_idx <= (kv_len - seq_len) + q_idx
//    i.e., the last seq_len positions of the kv sequence
// ============================================================


// ============================================================
//  5a: Naive Attention (materializes S matrix)
//
//  One block per (q_position, q_head) pair.
//  Threads collaborate on the kv_len dimension.
//
//  Steps:
//    1. Compute S[kv_len]: dot(Q[q], K[k]) * scale for each k
//    2. Apply causal mask (set masked positions to -inf)
//    3. Softmax over S
//    4. Compute O = S @ V (weighted sum of V rows)
//
//  This is O(seq * kv_len * d) memory for S — fine for testing,
//  but FlashAttention avoids materializing S.
// ============================================================

__global__ void attention_naive_kernel(
    half*       O,          // [seq_len, num_q_heads, head_dim]
    const half* Q,          // [seq_len, num_q_heads, head_dim]
    const half* K,          // [kv_len,  num_kv_heads, head_dim]
    const half* V,          // [kv_len,  num_kv_heads, head_dim]
    int seq_len,
    int kv_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int q_idx = blockIdx.x;   // which query position (0..seq_len-1)
    int q_head = blockIdx.y;  // which query head (0..num_q_heads-1)

    if (q_idx >= seq_len || q_head >= num_q_heads) return;

    // GQA mapping: which KV head does this Q head use?
    int kv_head = q_head / (num_q_heads / num_kv_heads);

    // Pointers
    const half* q_vec = Q + (long long)q_idx * num_q_heads * head_dim
                          + (long long)q_head * head_dim;
    const half* k_base = K + (long long)kv_head * head_dim;
    const half* v_base = V + (long long)kv_head * head_dim;
    half* o_vec = O + (long long)q_idx * num_q_heads * head_dim
                    + (long long)q_head * head_dim;

    // Strides for K and V along the kv_len dimension
    int kv_stride = num_kv_heads * head_dim;

    // Causal mask boundary: this q can attend to k positions [0, mask_end]
    // In prefill: q_idx maps to absolute position (kv_len - seq_len + q_idx)
    int q_abs_pos = kv_len - seq_len + q_idx;

    // Step 1 & 2: Compute attention scores with mask
    // Use shared memory for scores (one value per kv position)
    extern __shared__ float s_scores[];

    // Each thread handles a subset of kv positions
    for (int k_idx = threadIdx.x; k_idx < kv_len; k_idx += blockDim.x) {
        if (k_idx <= q_abs_pos) {
            // Dot product: Q[q] · K[k]
            float dot = 0.0f;
            const half* k_vec = k_base + (long long)k_idx * kv_stride;
            for (int d = 0; d < head_dim; d++) {
                dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
            }
            s_scores[k_idx] = dot * scale;
        } else {
            s_scores[k_idx] = -FLT_MAX;  // masked
        }
    }
    __syncthreads();

    // Step 3: Softmax over s_scores[0..kv_len-1]
    // Find max
    float local_max = -FLT_MAX;
    for (int i = threadIdx.x; i < kv_len; i += blockDim.x) {
        local_max = fmaxf(local_max, s_scores[i]);
    }
    float row_max = block_reduce_max(local_max);

    __shared__ float s_max;
    if (threadIdx.x == 0) s_max = row_max;
    __syncthreads();
    row_max = s_max;

    // Exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < kv_len; i += blockDim.x) {
        float val = expf(s_scores[i] - row_max);
        s_scores[i] = val;
        local_sum += val;
    }
    float row_sum = block_reduce_sum(local_sum);

    __shared__ float s_sum;
    if (threadIdx.x == 0) s_sum = row_sum;
    __syncthreads();
    row_sum = s_sum;

    // Normalize
    float inv_sum = 1.0f / row_sum;
    for (int i = threadIdx.x; i < kv_len; i += blockDim.x) {
        s_scores[i] *= inv_sum;
    }
    __syncthreads();

    // Step 4: O = S @ V — each thread computes a subset of output dimensions
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < kv_len; k_idx++) {
            const half* v_vec = v_base + (long long)k_idx * kv_stride;
            acc += s_scores[k_idx] * __half2float(v_vec[d]);
        }
        o_vec[d] = __float2half(acc);
    }
}

void launch_attention_naive(
    half* out, const half* Q, const half* K, const half* V,
    int seq_len, int kv_len,
    int num_q_heads, int num_kv_heads,
    int head_dim, float scale
) {
    dim3 grid(seq_len, num_q_heads);
    int block = 256;
    // Dynamic shared memory for attention scores
    size_t shmem = kv_len * sizeof(float);
    attention_naive_kernel<<<grid, block, shmem>>>(
        out, Q, K, V, seq_len, kv_len,
        num_q_heads, num_kv_heads, head_dim, scale);
}


// ============================================================
//  5b: Flash Attention with GQA
//
//  Online softmax — never materializes the full S matrix.
//  Processes K/V in tiles of KV_BLOCK_SIZE along the kv_len
//  dimension, maintaining running max and sum for numerically
//  stable softmax.
//
//  One block per (q_position, q_head) pair.
//  Threads in the block cooperate on head_dim dimensions.
//
//  Algorithm (per query vector):
//    Initialize: O = 0, m = -inf, l = 0
//    For each KV tile:
//      1. Compute partial scores s[j] = Q · K[j] * scale
//      2. m_new = max(m, max(s))
//      3. Correction factor: alpha = exp(m - m_new)
//      4. O = O * alpha + Σ exp(s[j] - m_new) * V[j]
//      5. l = l * alpha + Σ exp(s[j] - m_new)
//      6. m = m_new
//    Final: O = O / l
//
//  Memory: O(head_dim) per block — no O(kv_len) allocation.
// ============================================================

constexpr int KV_BLOCK_SIZE = 32;  // process 32 KV positions per tile

__global__ void flash_attention_gqa_kernel(
    half*       O,          // [seq_len, num_q_heads, head_dim]
    const half* Q,          // [seq_len, num_q_heads, head_dim]
    const half* K,          // [kv_len,  num_kv_heads, head_dim]
    const half* V,          // [kv_len,  num_kv_heads, head_dim]
    int seq_len,
    int kv_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int q_idx = blockIdx.x;
    int q_head = blockIdx.y;

    if (q_idx >= seq_len || q_head >= num_q_heads) return;

    int kv_head = q_head / (num_q_heads / num_kv_heads);

    const half* q_vec = Q + (long long)q_idx * num_q_heads * head_dim
                          + (long long)q_head * head_dim;
    half* o_vec = O + (long long)q_idx * num_q_heads * head_dim
                    + (long long)q_head * head_dim;

    int kv_stride = num_kv_heads * head_dim;
    int q_abs_pos = kv_len - seq_len + q_idx;

    // Load Q vector into registers (each thread loads a subset of dims)
    // With blockDim.x threads, each thread handles ceil(head_dim / blockDim.x) dims
    // For head_dim=64 and blockDim=64, each thread handles 1 dim
    float q_reg[4];  // max head_dim / min_block_size = 64 / 16 = 4
    int dims_per_thread = cdiv(head_dim, blockDim.x);
    for (int i = 0; i < dims_per_thread; i++) {
        int d = threadIdx.x + i * blockDim.x;
        if (d < head_dim) {
            q_reg[i] = __half2float(q_vec[d]);
        }
    }

    // Online softmax state per thread (for its output dimensions)
    float o_acc[4] = {0.0f};  // running output accumulator
    float m = -FLT_MAX;       // running max (shared across block)
    float l = 0.0f;           // running sum of exp (shared across block)

    // Shared memory for KV tile scores
    __shared__ float s_scores[KV_BLOCK_SIZE];
    __shared__ float s_new_max;
    __shared__ float s_new_sum;

    // Process KV in tiles
    for (int kv_start = 0; kv_start < kv_len; kv_start += KV_BLOCK_SIZE) {
        int kv_end = min(kv_start + KV_BLOCK_SIZE, kv_len);
        int tile_size = kv_end - kv_start;

        // Step 1: Compute scores for this tile
        // Each thread computes one or more scores via dot product
        for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
            int k_idx = kv_start + j;
            if (k_idx <= q_abs_pos) {
                const half* k_vec = K + (long long)k_idx * kv_stride
                                      + (long long)kv_head * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
                }
                s_scores[j] = dot * scale;
            } else {
                s_scores[j] = -FLT_MAX;
            }
        }
        __syncthreads();

        // Step 2: Find tile max
        float tile_max = -FLT_MAX;
        for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
            tile_max = fmaxf(tile_max, s_scores[j]);
        }
        tile_max = block_reduce_max(tile_max);

        // Broadcast new max
        float m_new = fmaxf(m, tile_max);
        if (threadIdx.x == 0) s_new_max = m_new;
        __syncthreads();
        m_new = s_new_max;

        // Step 3: Correction factor for previous accumulator
        float alpha = expf(m - m_new);

        // Step 4 & 5: Compute exp(s - m_new) and accumulate into output
        // First compute tile sum of exp
        float tile_sum = 0.0f;
        for (int j = threadIdx.x; j < tile_size; j += blockDim.x) {
            float val = expf(s_scores[j] - m_new);
            s_scores[j] = val;  // store exp'd value for V accumulation
            tile_sum += val;
        }
        tile_sum = block_reduce_sum(tile_sum);
        if (threadIdx.x == 0) s_new_sum = tile_sum;
        __syncthreads();
        tile_sum = s_new_sum;

        // Update running output: O = O * alpha + sum(exp(s) * V)
        for (int i = 0; i < dims_per_thread; i++) {
            int d = threadIdx.x + i * blockDim.x;
            if (d < head_dim) {
                o_acc[i] *= alpha;

                for (int j = 0; j < tile_size; j++) {
                    const half* v_vec = V + (long long)(kv_start + j) * kv_stride
                                          + (long long)kv_head * head_dim;
                    o_acc[i] += s_scores[j] * __half2float(v_vec[d]);
                }
            }
        }

        // Update running statistics
        l = l * alpha + tile_sum;
        m = m_new;

        __syncthreads();
    }

    // Final: normalize output by l
    float inv_l = 1.0f / l;
    for (int i = 0; i < dims_per_thread; i++) {
        int d = threadIdx.x + i * blockDim.x;
        if (d < head_dim) {
            o_vec[d] = __float2half(o_acc[i] * inv_l);
        }
    }
}

// ============================================================
//  5d: Split-KV decode attention (flash-decoding), seq_len = 1
//
//  The kernel above gives decode 32 blocks (one per Q head) of 64
//  threads, each walking the whole KV range serially: at kv_len 2048
//  a layer's 4.2 MB of K/V took ~465 us (~9 GB/s). Here:
//
//  Grid (kv_head, split), one warp per block. A block owns one KV
//  head and all FD_G = 4 Q heads that share it (GQA), so every K/V
//  row is read once, and a chunk of the KV range; chunks are sized
//  so that long contexts get up to FD_MAX_SPLITS blocks per head.
//    scores: lane l takes KV position base+l, reads its 128 B K row
//            (8 × uint4) and dots it with the 4 Q heads (q·scale in smem)
//    softmax: per-head running (m, l), warp max/sum via shuffles
//    P·V:   lane owns dims 2l, 2l+1; for each position reads a half2
//            of V (one 128 B row per warp) and gets p by shuffle
//  Splits are merged like the GEMV's split-K: each block stores its
//  (m, l, unnormalized O) per Q head, the last block of a KV head
//  (atomic counter) rescales and sums them in split order. One
//  launch, deterministic.
// ============================================================

constexpr int FD_D = 64;            // head_dim
constexpr int FD_G = 4;             // Q heads per KV head
constexpr int FD_TILE = 32;         // KV positions per warp step
constexpr int FD_MAX_SPLITS = 64;
constexpr int FD_MAX_KV_HEADS = 8;

__device__ float    g_fd_o[FD_MAX_SPLITS * FD_MAX_KV_HEADS * FD_G * FD_D];
__device__ float    g_fd_ml[FD_MAX_SPLITS * FD_MAX_KV_HEADS * FD_G * 2];
__device__ unsigned g_fd_count[FD_MAX_KV_HEADS];

__global__ void __launch_bounds__(32)
flash_decode_kernel(half* __restrict__ O, const half* __restrict__ Q,
                    const half* __restrict__ K, const half* __restrict__ V,
                    int kv_len, int num_kv_heads, int chunk, float scale) {
    const int kvh = blockIdx.x, split = blockIdx.y, nsplit = gridDim.y;
    const int lane = threadIdx.x;
    const int kv_stride = num_kv_heads * FD_D;
    const int nqh = num_kv_heads * FD_G;
    const int qh0 = kvh * FD_G;

    __shared__ float q_s[FD_G][FD_D];
    for (int i = lane; i < FD_G * FD_D; i += 32)
        q_s[i / FD_D][i % FD_D] = __half2float(Q[qh0 * FD_D + i]) * scale;
    __syncwarp();

    float m[FD_G], l[FD_G], o[FD_G][2];
    #pragma unroll
    for (int h = 0; h < FD_G; h++) { m[h] = -FLT_MAX; l[h] = 0.0f; o[h][0] = o[h][1] = 0.0f; }

    const half* Kh = K + kvh * FD_D;
    const half* Vh = V + kvh * FD_D;
    const int k0 = split * chunk;
    const int k1 = min(kv_len, k0 + chunk);

    for (int base = k0; base < k1; base += FD_TILE) {
        const int j = base + lane;
        const bool valid = j < k1;

        // ---- scores for this lane's position, all 4 heads ----
        float s[FD_G] = {};
        if (valid) {
            const uint4* krow = reinterpret_cast<const uint4*>(Kh + (size_t)j * kv_stride);
            #pragma unroll
            for (int c = 0; c < FD_D / 8; c++) {
                uint4 kk = __ldg(krow + c);
                const half2* k2 = reinterpret_cast<const half2*>(&kk);
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    float2 kf = __half22float2(k2[e]);
                    const int d = c * 8 + e * 2;
                    #pragma unroll
                    for (int h = 0; h < FD_G; h++)
                        s[h] = fmaf(q_s[h][d], kf.x, fmaf(q_s[h][d + 1], kf.y, s[h]));
                }
            }
        }

        // ---- online softmax update ----
        float p[FD_G];
        #pragma unroll
        for (int h = 0; h < FD_G; h++) {
            float tile_max = warp_reduce_max(valid ? s[h] : -FLT_MAX);
            float m_new = fmaxf(m[h], tile_max);
            float alpha = __expf(m[h] - m_new);
            p[h] = valid ? __expf(s[h] - m_new) : 0.0f;
            l[h] = l[h] * alpha + warp_reduce_sum(p[h]);
            o[h][0] *= alpha;
            o[h][1] *= alpha;
            m[h] = m_new;
        }

        // ---- O += P · V over the tile's positions ----
        const int n = min(FD_TILE, k1 - base);
        const half* vrow = Vh + (size_t)base * kv_stride + lane * 2;
        #pragma unroll 8
        for (int jj = 0; jj < n; jj++) {
            float2 v = __half22float2(*reinterpret_cast<const half2*>(vrow + (size_t)jj * kv_stride));
            #pragma unroll
            for (int h = 0; h < FD_G; h++) {
                float pj = __shfl_sync(0xffffffffu, p[h], jj);
                o[h][0] = fmaf(pj, v.x, o[h][0]);
                o[h][1] = fmaf(pj, v.y, o[h][1]);
            }
        }
    }

    if (nsplit == 1) {
        #pragma unroll
        for (int h = 0; h < FD_G; h++) {
            float inv = 1.0f / l[h];
            *reinterpret_cast<half2*>(O + (qh0 + h) * FD_D + lane * 2) =
                __floats2half2_rn(o[h][0] * inv, o[h][1] * inv);
        }
        return;
    }

    // ---- publish partials, last block of this KV head merges ----
    #pragma unroll
    for (int h = 0; h < FD_G; h++) {
        const int idx = split * nqh + qh0 + h;
        g_fd_o[idx * FD_D + lane * 2]     = o[h][0];
        g_fd_o[idx * FD_D + lane * 2 + 1] = o[h][1];
        if (lane == 0) { g_fd_ml[idx * 2] = m[h]; g_fd_ml[idx * 2 + 1] = l[h]; }
    }
    __threadfence();
    __syncwarp();
    unsigned done = 0;
    if (lane == 0) done = atomicAdd(&g_fd_count[kvh], 1u);
    done = __shfl_sync(0xffffffffu, done, 0);
    if (done != (unsigned)nsplit - 1) return;

    #pragma unroll
    for (int h = 0; h < FD_G; h++) {
        const int qh = qh0 + h;
        float M = -FLT_MAX;
        for (int sp = 0; sp < nsplit; sp++) M = fmaxf(M, __ldcg(&g_fd_ml[(sp * nqh + qh) * 2]));
        float L = 0.0f, o0 = 0.0f, o1 = 0.0f;
        for (int sp = 0; sp < nsplit; sp++) {
            const int idx = sp * nqh + qh;
            float w = __expf(__ldcg(&g_fd_ml[idx * 2]) - M);
            L  += __ldcg(&g_fd_ml[idx * 2 + 1]) * w;
            o0 += __ldcg(&g_fd_o[idx * FD_D + lane * 2]) * w;
            o1 += __ldcg(&g_fd_o[idx * FD_D + lane * 2 + 1]) * w;
        }
        float inv = 1.0f / L;
        *reinterpret_cast<half2*>(O + qh * FD_D + lane * 2) = __floats2half2_rn(o0 * inv, o1 * inv);
    }
    if (lane == 0) g_fd_count[kvh] = 0;  // ready for the next launch
}

// ============================================================
//  5e: Prefill attention on Tensor Cores (seq_len > 1)
//
//  Ported from the author's FlashAttention-lite v5 (FA-2 structure
//  on raw mma.sync): each warp owns 16 query rows of a 64-row block;
//  S, P and O stay in registers; Q/K/V tiles go through XOR-swizzled
//  shared memory read with ldmatrix; K/V tiles are double-buffered
//  with cp.async. Changes for this engine:
//    - layouts: Q/O are [seq, num_q_heads·D], K/V the KV cache
//      [kv_len, num_kv_heads·D]; one block per (64 query rows, Q head)
//    - GQA: the block reads KV head q_head / (num_q_heads/num_kv_heads)
//    - causal mask with a query offset: row i sits at position
//      kv_len - seq_len + i and sees columns <= that position
//    - tile skip: a block stops at its last row's position, so tiles
//      entirely above the diagonal are never loaded
//    - FP16 output
// ============================================================

namespace fa {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, const half* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, const half* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
}
// c(16x8 f32) += a(16x16 f16, row) * b(16x8 f16, col)
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack_half2(float x, float y) {
    half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&h);
}
// 16 B global -> shared, bypassing registers and L1. src-size 0 writes zeros.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool in) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(smem_u32(dst)), "l"(src), "r"(in ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int n>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(n) : "memory"); }

// Row r of a [rows, D] half tile is D/8 chunks of 16 B; chunk c is stored at c ^ (r % 8).
template <int D>
__device__ __forceinline__ int swz(int r, int d) {
    return r * D + ((((d >> 3) ^ r) & 7) | ((d >> 3) & ~7)) * 8 + (d & 7);
}

constexpr int BR = 64, BC = 64, D = 64, T = 128;
constexpr int SMEM_HALVES = 2 * BC * D + (BR > 2 * BC ? BR : 2 * BC) * D;  // 2 K/V stages; Q borrows stage 1
// -inf from its bit pattern: MSVC's INFINITY macro expands to 1e300*1e300, which nvcc flags with #221
__device__ __forceinline__ float neg_inf() { return __int_as_float(0xff800000); }

}  // namespace fa

__global__ void __launch_bounds__(fa::T)
flash_prefill_kernel(half* __restrict__ O, const half* __restrict__ Q,
                     const half* __restrict__ K, const half* __restrict__ V,
                     int seq_len, int kv_len, int num_q_heads, int num_kv_heads, float scale) {
    using namespace fa;
    constexpr int KD = D / 16, NS = BC / 8, NO = D / 8;

    const int qh  = blockIdx.y;
    const int kvh = qh / (num_q_heads / num_kv_heads);
    const int q_stride  = num_q_heads * D;
    const int kv_stride = num_kv_heads * D;
    const int start = kv_len - seq_len;          // position of query row 0
    Q += qh * D;  O += qh * D;  K += kvh * D;  V += kvh * D;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* sKV = reinterpret_cast<half*>(smem_raw);
    half* sQ  = sKV + 2 * BC * D;

    const int tid = threadIdx.x, lane = tid & 31;
    const int wr = (tid >> 5) * 16;               // this warp's first row in the tile
    const int g = lane >> 2, t = lane & 3;        // accumulator row (and +8), column pair
    const int i0 = blockIdx.x * BR;
    // Causal tile skip: no row of this block sees past its last row's position
    const int kv_end = min(kv_len, start + min(i0 + BR, seq_len));
    const int pos0 = start + i0 + wr + g;         // positions of this thread's two rows
    const int pos1 = pos0 + 8;

    auto load_kv = [&](int j0, int st) {
        half* sK = sKV + st * 2 * BC * D;
        half* sV = sK + BC * D;
        for (int c = tid; c < BC * D / 8; c += T) {
            int b = c / (D / 8), d = c % (D / 8) * 8;
            bool in = j0 + b < kv_end;
            size_t src = (size_t)(in ? j0 + b : 0) * kv_stride + d;
            cp_async16(sK + swz<D>(b, d), K + src, in);
            cp_async16(sV + swz<D>(b, d), V + src, in);
        }
        cp_async_commit();
    };

    for (int c = tid; c < BR * D / 8; c += T) {
        int r = c / (D / 8), d = c % (D / 8) * 8;
        bool in = i0 + r < seq_len;
        cp_async16(sQ + swz<D>(r, d), Q + (size_t)(in ? i0 + r : 0) * q_stride + d, in);
    }
    cp_async_commit();
    load_kv(0, 0);
    cp_async_wait<1>();   // Q is in; tile 0 may still be in flight
    __syncthreads();

    uint32_t qf[KD][4];
    #pragma unroll
    for (int kk = 0; kk < KD; ++kk)
        ldsm_x4(qf[kk], sQ + swz<D>(wr + (lane & 15), kk * 16 + (lane >> 4) * 8));

    float o[NO][4] = {};
    float m[2] = { neg_inf(), neg_inf() };
    float l[2] = { 0.0f, 0.0f };   // per-thread partial row sums, reduced at the end

    for (int j0 = 0, st = 0; j0 < kv_end; j0 += BC, st ^= 1) {
        __syncthreads();
        if (j0 + BC < kv_end) load_kv(j0 + BC, st ^ 1);
        else                  cp_async_commit();   // empty group keeps the wait count uniform
        cp_async_wait<1>();
        __syncthreads();
        const half* sK = sKV + st * 2 * BC * D;
        const half* sV = sK + BC * D;

        // S = Q K^T
        float s[NS][4] = {};
        #pragma unroll
        for (int kk = 0; kk < KD; ++kk) {
            #pragma unroll
            for (int nb = 0; nb < NS / 2; ++nb) {
                uint32_t b[4];
                ldsm_x4(b, sK + swz<D>(nb * 16 + (lane & 7) + (lane >> 4) * 8,
                                       kk * 16 + ((lane >> 3) & 1) * 8));
                mma16816(s[2 * nb],     qf[kk], b[0], b[1]);
                mma16816(s[2 * nb + 1], qf[kk], b[2], b[3]);
            }
        }

        // Causal mask + online softmax. s[n][0..1] is row g, s[n][2..3] row g+8.
        // Column 0 is visible to every row, so m is finite after the first tile.
        float mx[2] = { m[0], m[1] };
        #pragma unroll
        for (int n = 0; n < NS; ++n)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                int col = j0 + n * 8 + 2 * t + (e & 1);
                int pos = (e >> 1) ? pos1 : pos0;
                s[n][e] = (col <= pos && col < kv_len) ? s[n][e] * scale : neg_inf();
                mx[e >> 1] = fmaxf(mx[e >> 1], s[n][e]);
            }
        float alpha[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffff, mx[i], 1));
            mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffff, mx[i], 2));
            alpha[i] = __expf(m[i] - mx[i]);
            m[i] = mx[i];
            l[i] *= alpha[i];
        }
        #pragma unroll
        for (int n = 0; n < NS; ++n)
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                s[n][e] = __expf(s[n][e] - m[e >> 1]);
                l[e >> 1] += s[n][e];
            }
        #pragma unroll
        for (int n = 0; n < NO; ++n)
            #pragma unroll
            for (int e = 0; e < 4; ++e) o[n][e] *= alpha[e >> 1];

        // O += P V; the S accumulators of n-tiles 2k, 2k+1 are the A fragment for k-step k
        #pragma unroll
        for (int kk = 0; kk < BC / 16; ++kk) {
            uint32_t a[4] = { pack_half2(s[2 * kk][0],     s[2 * kk][1]),
                              pack_half2(s[2 * kk][2],     s[2 * kk][3]),
                              pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                              pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3]) };
            #pragma unroll
            for (int nb = 0; nb < NO / 2; ++nb) {
                uint32_t b[4];
                ldsm_x4_t(b, sV + swz<D>(kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                                         nb * 16 + (lane >> 4) * 8));
                mma16816(o[2 * nb],     a, b[0], b[1]);
                mma16816(o[2 * nb + 1], a, b[2], b[3]);
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        l[i] += __shfl_xor_sync(0xffffffff, l[i], 1);
        l[i] += __shfl_xor_sync(0xffffffff, l[i], 2);
    }
    const int r0 = i0 + wr + g, r1 = r0 + 8;
    const float inv0 = 1.0f / l[0], inv1 = 1.0f / l[1];
    #pragma unroll
    for (int n = 0; n < NO; ++n) {
        int c = n * 8 + 2 * t;
        if (r0 < seq_len) *reinterpret_cast<half2*>(O + (size_t)r0 * q_stride + c) = __floats2half2_rn(o[n][0] * inv0, o[n][1] * inv0);
        if (r1 < seq_len) *reinterpret_cast<half2*>(O + (size_t)r1 * q_stride + c) = __floats2half2_rn(o[n][2] * inv1, o[n][3] * inv1);
    }
}

bool flash_prefill_supported(int num_q_heads, int num_kv_heads, int head_dim) {
    return head_dim == fa::D && num_q_heads % num_kv_heads == 0;
}

void launch_flash_prefill(half* out, const half* Q, const half* K, const half* V,
                          int seq_len, int kv_len, int num_q_heads, int num_kv_heads,
                          int head_dim, float scale) {
    (void)head_dim;  // fixed by flash_prefill_supported
    constexpr size_t smem = (size_t)fa::SMEM_HALVES * sizeof(half);  // 32 KB
    dim3 grid(cdiv(seq_len, fa::BR), num_q_heads);
    flash_prefill_kernel<<<grid, fa::T, smem>>>(out, Q, K, V, seq_len, kv_len,
                                                num_q_heads, num_kv_heads, scale);
}

bool flash_decode_supported(int num_q_heads, int num_kv_heads, int head_dim) {
    return head_dim == FD_D && num_q_heads == FD_G * num_kv_heads && num_kv_heads <= FD_MAX_KV_HEADS;
}

void launch_flash_decode(half* out, const half* Q, const half* K, const half* V,
                         int kv_len, int num_q_heads, int num_kv_heads, int head_dim, float scale) {
    (void)num_q_heads; (void)head_dim;  // fixed by flash_decode_supported
    // One 32-position tile per block until the split cap; longer contexts get longer chunks
    int chunk = max(FD_TILE, cdiv(cdiv(kv_len, FD_MAX_SPLITS), FD_TILE) * FD_TILE);
    dim3 grid(num_kv_heads, cdiv(kv_len, chunk));
    flash_decode_kernel<<<grid, 32>>>(out, Q, K, V, kv_len, num_kv_heads, chunk, scale);
}

void launch_flash_attention_gqa(
    half* out, const half* Q, const half* K, const half* V,
    int seq_len, int kv_len,
    int num_q_heads, int num_kv_heads,
    int head_dim, float scale
) {
    dim3 grid(seq_len, num_q_heads);
    // Use head_dim threads so each thread owns one output dimension
    int block = head_dim;  // 64 for Llama 3.2
    size_t shmem = KV_BLOCK_SIZE * sizeof(float);
    flash_attention_gqa_kernel<<<grid, block, shmem>>>(
        out, Q, K, V, seq_len, kv_len,
        num_q_heads, num_kv_heads, head_dim, scale);
}
