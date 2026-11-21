#pragma once

#include <cuda_fp16.h>

// ============================================================
//  Kernel launch function declarations
//  Added incrementally as each kernel is implemented.
// ============================================================

// --- M1: Element-wise kernels ---
void launch_residual_add(half* out, const half* a, const half* b, int n);
void launch_silu_mul(half* out, const half* gate, const half* up, int n);
// d_offset (optional, device): token i is read from token_ids[*d_offset + i].
void launch_embedding(half* out, const half* table, const int* token_ids, int seq_len, int hidden_size, const int* d_offset = nullptr);

// --- M2: Reduction kernels ---
void launch_rmsnorm(half* out, const half* x, const half* weight, int rows, int cols, float eps);
void launch_softmax(float* x, int rows, int cols);

// --- M3: RoPE ---
// Llama 3 frequency scaling (HF rope_type "llama3"). factor = 1 leaves the
// frequencies unchanged.
struct RopeScaling {
    float factor            = 1.0f;
    float low_freq_factor   = 1.0f;
    float high_freq_factor  = 4.0f;
    int   original_max_pos  = 8192;
};
// d_start_pos (optional, device): overrides start_pos, read when the kernel runs. Then k
// is the KV cache base ([pos, kv_heads·head_dim]) and the rows rotated are those at the
// device position; q is still indexed from row 0.
void launch_rope(half* q, half* k, int seq_len, int num_q_heads, int num_kv_heads, int head_dim, int start_pos, float theta, RopeScaling scaling = {}, const int* d_start_pos = nullptr);

// --- M4: GEMM ---
void launch_gemm_naive(half* C, const half* A, const half* B, int M, int N, int K);
void launch_gemm_tiled(half* C, const half* A, const half* B, int M, int N, int K);
void launch_gemm_wmma(half* C, const half* A, const half* B, int M, int N, int K);
// Decode GEMV: y[N] = x[K] × W[K,N] (row-major W). Needs N % 8 == 0.
bool gemv_supported(int N);
void launch_gemv(half* y, const half* x, const half* W, int N, int K);

// Fused gate/up: W_gate_up is [K, 2*I] with gate and up columns interleaved in
// groups of GATE_UP_GROUP: g[0:32] u[0:32] g[32:64] u[32:64] ... One GEMV block
// (64 columns) then holds matching gate and up outputs.
constexpr int GATE_UP_GROUP = 32;
// Decode: out[I] = silu(x·W_gate) * (x·W_up) in one launch. Needs I % GATE_UP_GROUP == 0.
void launch_gemv_swiglu(half* out, const half* x, const half* W_gate_up, int I, int K);
// Prefill: out[rows, I] = silu(gate) * up, read from the interleaved [rows, 2*I] GEMM output.
void launch_silu_mul_interleaved(half* out, const half* gate_up, int rows, int I);

// --- M5: Attention ---
void launch_attention_naive(half* out, const half* Q, const half* K, const half* V, int seq_len, int kv_len, int num_q_heads, int num_kv_heads, int head_dim, float scale);
void launch_flash_attention_gqa(half* out, const half* Q, const half* K, const half* V, int seq_len, int kv_len, int num_q_heads, int num_kv_heads, int head_dim, float scale);
// Tensor Core prefill attention (FlashAttention-lite v5 port), causal, GQA, head_dim 64.
bool flash_prefill_supported(int num_q_heads, int num_kv_heads, int head_dim);
void launch_flash_prefill(half* out, const half* Q, const half* K, const half* V, int seq_len, int kv_len, int num_q_heads, int num_kv_heads, int head_dim, float scale);
// Split-KV decode attention (seq_len = 1): head_dim 64, 4 Q heads per KV head, <= 8 KV heads.
bool flash_decode_supported(int num_q_heads, int num_kv_heads, int head_dim);
void launch_flash_decode(half* out, const half* Q, const half* K, const half* V, int kv_len, int num_q_heads, int num_kv_heads, int head_dim, float scale);
// Same, with kv_len = *d_pos + 1 read on the device: the grid is sized for max_kv_len,
// so one launch configuration serves every position (for CUDA Graph replay). Splits past
// kv_len contribute nothing; for max_kv_len <= 2048 the result is bit-identical to the
// host-kv_len launch.
void launch_flash_decode_dev(half* out, const half* Q, const half* K, const half* V, const int* d_pos, int max_kv_len, int num_q_heads, int num_kv_heads, int head_dim, float scale);
