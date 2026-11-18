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
