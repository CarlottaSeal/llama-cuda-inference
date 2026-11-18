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

