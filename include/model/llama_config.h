#pragma once

#include <cuda_fp16.h>

// ============================================================
//  Llama 3.2 1B — Model Configuration
//
//  Source: meta-llama/Llama-3.2-1B config.json
//  All values hardcoded for this specific model.
// ============================================================

struct LlamaConfig {
    // Architecture
    static constexpr int num_layers    = 16;
    static constexpr int hidden_size   = 2048;
    static constexpr int num_q_heads   = 32;     // query heads
    static constexpr int num_kv_heads  = 8;      // key/value heads (GQA 4:1)
    static constexpr int head_dim      = 64;     // hidden_size / num_q_heads
    static constexpr int intermediate_size = 8192; // FFN intermediate
    static constexpr int vocab_size    = 128256;
    static constexpr int max_seq_len   = 2048;   // conservative default

    // RoPE
    static constexpr float rope_theta  = 500000.0f;
    // rope_scaling (rope_type "llama3") from config.json
    static constexpr float rope_factor           = 32.0f;
    static constexpr float rope_low_freq_factor  = 1.0f;
    static constexpr float rope_high_freq_factor = 4.0f;
    static constexpr int   rope_original_max_pos = 8192;

    // Normalization
    static constexpr float rms_norm_eps = 1e-5f;

    // Derived constants
    static constexpr int gqa_ratio = num_q_heads / num_kv_heads;  // 4
    static constexpr int q_dim     = num_q_heads * head_dim;      // 2048
    static constexpr int kv_dim    = num_kv_heads * head_dim;     // 512
    static constexpr int qkv_dim   = q_dim + 2 * kv_dim;          // 3072, fused Q|K|V columns
};

// ============================================================
//  Weight pointers for one transformer layer (all FP16 on GPU)
// ============================================================

// Linear weights are transposed at load time from HF's [out, in] to
// [in, out], so each projection is a plain C[M,N] = A[M,K] × B[K,N].
struct LlamaLayerWeights {
    // Attention
    half* qkv_proj;     // [hidden_size, qkv_dim]: columns are W_q | W_k | W_v
    half* o_proj;       // [q_dim, hidden_size]

    // FFN (SwiGLU)
    half* gate_up_proj; // [hidden_size, 2*intermediate_size]: W_gate and W_up columns
                        // interleaved in groups of GATE_UP_GROUP (32), see kernels.cuh
    half* down_proj;    // [intermediate_size, hidden_size]

    // Norms
    half* attn_norm;    // [hidden_size]
    half* ffn_norm;     // [hidden_size]
};

// ============================================================
//  All model weights
// ============================================================

struct LlamaWeights {
    half* embedding;                    // [vocab_size, hidden_size]
    LlamaLayerWeights layers[LlamaConfig::num_layers];
    half* final_norm;                   // [hidden_size]
    half* lm_head;                      // [hidden_size, vocab_size] (transposed)
};
