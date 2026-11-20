#pragma once

#include <cuda_fp16.h>

// ============================================================
//  Kernel launch function declarations
//  Added incrementally as each kernel is implemented.
// ============================================================

// --- M1: Element-wise kernels ---
void launch_residual_add(half* out, const half* a, const half* b, int n);
void launch_silu_mul(half* out, const half* gate, const half* up, int n);
void launch_embedding(half* out, const half* table, const int* token_ids, int seq_len, int hidden_size);

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
void launch_rope(half* q, half* k, int seq_len, int num_q_heads, int num_kv_heads, int head_dim, int start_pos, float theta, RopeScaling scaling = {});

// --- M4: GEMM ---
void launch_gemm_naive(half* C, const half* A, const half* B, int M, int N, int K);
void launch_gemm_tiled(half* C, const half* A, const half* B, int M, int N, int K);
void launch_gemm_wmma(half* C, const half* A, const half* B, int M, int N, int K);
// Decode GEMV: y[N] = x[K] × W[K,N] (row-major W). Needs N % 8 == 0.
bool gemv_supported(int N);
void launch_gemv(half* y, const half* x, const half* W, int N, int K);

// --- M5: Attention ---
void launch_attention_naive(half* out, const half* Q, const half* K, const half* V, int seq_len, int kv_len, int num_q_heads, int num_kv_heads, int head_dim, float scale);
void launch_flash_attention_gqa(half* out, const half* Q, const half* K, const half* V, int seq_len, int kv_len, int num_q_heads, int num_kv_heads, int head_dim, float scale);
// Split-KV decode attention (seq_len = 1): head_dim 64, 4 Q heads per KV head, <= 8 KV heads.
bool flash_decode_supported(int num_q_heads, int num_kv_heads, int head_dim);
void launch_flash_decode(half* out, const half* Q, const half* K, const half* V, int kv_len, int num_q_heads, int num_kv_heads, int head_dim, float scale);
