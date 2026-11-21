#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"
#include <cmath>

// ============================================================
//  RoPE (Rotary Position Embedding)
//
//  In-place rotation of Q and K tensors.
//  Layout: [seq_len, num_heads, head_dim]
//
//  Pairs follow the HF Llama convention (rotate_half): dimension i is
//  rotated together with i + head_dim/2, not with its neighbour 2i+1.
//  The HF checkpoints' q_proj/k_proj rows are laid out for this pairing.
//
//  For each position pos and pair i in [0, head_dim/2):
//    freq  = 1.0 / pow(theta, 2*i / head_dim)
//    angle = pos * freq
//    x'[i]       = x[i] * cos(angle) - x[i + d/2] * sin(angle)
//    x'[i + d/2] = x[i] * sin(angle) + x[i + d/2] * cos(angle)
//
//  Llama 3.2: theta = 500000, head_dim = 64 → 32 pairs per head
//
//  Llama 3 scaling (HF _compute_llama3_parameters), with
//  wavelen = 2π / freq, L = original_max_pos:
//    wavelen < L / high_freq_factor  → freq unchanged
//    wavelen > L / low_freq_factor   → freq / factor
//    otherwise s = (L / wavelen - low) / (high - low),
//              freq = (1 - s) * freq / factor + s * freq
//
//  Grid:  (seq_len, num_heads)
//  Block: (head_dim / 2)   — 32 threads for head_dim=64
//
//  start_pos: position offset for decode phase.
//             During prefill: start_pos=0, pos = token_index
//             During decode:  start_pos=past_kv_len, pos = start_pos + token_index
// ============================================================

__device__ float llama3_scaled_freq(float freq, RopeScaling s) {
    float wavelen = 2.0f * 3.14159265358979f / freq;
    float low_wavelen  = s.original_max_pos / s.low_freq_factor;
    float high_wavelen = s.original_max_pos / s.high_freq_factor;
    if (wavelen < high_wavelen) return freq;
    if (wavelen > low_wavelen)  return freq / s.factor;
    float smooth = (s.original_max_pos / wavelen - s.low_freq_factor)
                 / (s.high_freq_factor - s.low_freq_factor);
    return (1.0f - smooth) * freq / s.factor + smooth * freq;
}

__global__ void rope_kernel(
    half* x,           // [seq_len, num_heads, head_dim], modified in-place
    int   seq_len,
    int   num_heads,
    int   head_dim,
    int   start_pos,
    float theta,
    RopeScaling scaling,
    const int* d_start_pos,
    bool rows_at_pos   // x is indexed by absolute position (KV cache base) instead of token
) {
    int token = blockIdx.x;   // which token in the sequence
    int head  = blockIdx.y;   // which head
    int pair  = threadIdx.x;  // which dimension pair (0 .. head_dim/2 - 1)

    if (token >= seq_len || head >= num_heads || pair >= head_dim / 2) return;

    // Position in the full sequence (accounting for KV cache offset)
    int pos = (d_start_pos ? *d_start_pos : start_pos) + token;
    int row = rows_at_pos ? pos : token;

    // Compute rotation angle
    // freq = 1 / theta^(2*pair / head_dim)
    float freq = 1.0f / powf(theta, (2.0f * pair) / (float)head_dim);
    freq = llama3_scaled_freq(freq, scaling);
    float angle = (float)pos * freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    // The pair: x[token, head, pair] and x[token, head, pair + head_dim/2]
    long long offset = (long long)row * num_heads * head_dim
                     + (long long)head * head_dim
                     + pair;
    int half_dim = head_dim / 2;

    float x0 = __half2float(x[offset]);
    float x1 = __half2float(x[offset + half_dim]);

    // Rotation
    float out0 = x0 * cos_a - x1 * sin_a;
    float out1 = x0 * sin_a + x1 * cos_a;

    x[offset]            = __float2half(out0);
    x[offset + half_dim] = __float2half(out1);
}

void launch_rope(
    half* q, half* k,
    int seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    float theta,
    RopeScaling scaling,
    const int* d_start_pos
) {
    int half_dim = head_dim / 2;  // threads per block = 32 for head_dim=64

    // Apply RoPE to Q
    dim3 grid_q(seq_len, num_q_heads);
    rope_kernel<<<grid_q, half_dim>>>(q, seq_len, num_q_heads, head_dim, start_pos, theta, scaling,
                                      d_start_pos, false);

    // Apply RoPE to K. With a device position, k is the cache base and the rows
    // are found at that position on the device.
    dim3 grid_k(seq_len, num_kv_heads);
    rope_kernel<<<grid_k, half_dim>>>(k, seq_len, num_kv_heads, head_dim, start_pos, theta, scaling,
                                      d_start_pos, d_start_pos != nullptr);
}
