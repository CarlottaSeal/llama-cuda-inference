#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"
#include <cmath>

// ============================================================
//  RoPE (Rotary Position Embedding)
//
//  In-place rotation of Q and K tensors.
//  Layout: [seq_len, num_heads, head_dim]
//
//  For each position pos and dimension pair (2i, 2i+1):
//    freq  = 1.0 / pow(theta, 2*i / head_dim)
//    angle = pos * freq
//    x'[2i]   = x[2i]   * cos(angle) - x[2i+1] * sin(angle)
//    x'[2i+1] = x[2i]   * sin(angle) + x[2i+1] * cos(angle)
//
//  Llama 3.2: theta = 500000, head_dim = 64 → 32 pairs per head
//
//  Grid:  (seq_len, num_heads)
//  Block: (head_dim / 2)   — 32 threads for head_dim=64
//
//  start_pos: position offset for decode phase.
//             During prefill: start_pos=0, pos = token_index
//             During decode:  start_pos=past_kv_len, pos = start_pos + token_index
// ============================================================

__global__ void rope_kernel(
    half* x,           // [seq_len, num_heads, head_dim], modified in-place
    int   seq_len,
    int   num_heads,
    int   head_dim,
    int   start_pos,
    float theta
) {
    int token = blockIdx.x;   // which token in the sequence
    int head  = blockIdx.y;   // which head
    int pair  = threadIdx.x;  // which dimension pair (0 .. head_dim/2 - 1)

    if (token >= seq_len || head >= num_heads || pair >= head_dim / 2) return;

    // Position in the full sequence (accounting for KV cache offset)
    int pos = start_pos + token;

    // Compute rotation angle
    // freq = 1 / theta^(2*pair / head_dim)
    float freq = 1.0f / powf(theta, (2.0f * pair) / (float)head_dim);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // Pointer to the pair of elements: x[token, head, 2*pair] and x[token, head, 2*pair+1]
    long long offset = (long long)token * num_heads * head_dim
                     + (long long)head * head_dim
                     + 2 * pair;

    float x0 = __half2float(x[offset]);
    float x1 = __half2float(x[offset + 1]);

    // Rotation
    float out0 = x0 * cos_a - x1 * sin_a;
    float out1 = x0 * sin_a + x1 * cos_a;

    x[offset]     = __float2half(out0);
    x[offset + 1] = __float2half(out1);
}

void launch_rope(
    half* q, half* k,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    float theta
) {
    int half_dim = head_dim / 2;  // threads per block = 32 for head_dim=64

    // Apply RoPE to Q
    dim3 grid_q(seq_len, num_q_heads);
    rope_kernel<<<grid_q, half_dim>>>(q, seq_len, num_q_heads, head_dim, start_pos, theta);

    // Apply RoPE to K
    dim3 grid_k(seq_len, num_kv_heads);
    rope_kernel<<<grid_k, half_dim>>>(k, seq_len, num_kv_heads, head_dim, start_pos, theta);
}
