#pragma once

#include "model/llama_config.h"
#include <vector>

// ============================================================
//  Llama Inference Engine
//
//  Manages GPU scratch buffers, KV cache, and runs the
//  full forward pass (prefill + decode).
//
//  Buffer layout (all FP16 unless noted):
//    hidden:     [max_seq, hidden_size]     — current hidden states
//    hidden2:    [max_seq, hidden_size]     — scratch for RMSNorm output
//    qkv:        [max_seq, qkv_dim]         — fused Q|K|V projection output
//    q:          [max_seq, q_dim]           — Q split out of qkv (K/V go
//                                             straight into the KV cache)
//    attn_out:   [max_seq, q_dim]           — attention output
//    gate:       [max_seq, intermediate]    — FFN gate output
//    up:         [max_seq, intermediate]    — FFN up output
//    ffn_out:    [max_seq, hidden_size]     — FFN down output
//    logits:     [vocab_size]               — FP32 logits, last token
//
//  KV Cache: [num_layers, 2, num_kv_heads, max_seq, head_dim]
//    Stored in FP16. Key and Value are separate for each layer.
//    Grows as tokens are generated.
// ============================================================

struct InferenceBuffers {
    // Activation buffers
    half* hidden;       // [max_seq, hidden_size]
    half* hidden2;      // [max_seq, hidden_size]
    half* q;            // [max_seq, q_dim]
    half* qkv;          // [max_seq, qkv_dim] — fused Q|K|V projection output
    half* attn_out;     // [max_seq, q_dim]
    half* gate;         // [max_seq, intermediate_size]
    half* up;           // [max_seq, intermediate_size]
    half* ffn_out;      // [max_seq, hidden_size]
    half* logits_half;  // [vocab_size] — LM head GEMM output
    float* logits;      // [vocab_size] — FP32, last token only

    // KV Cache
    // Layout: kv_cache[layer][0=K,1=V] points to [max_seq, num_kv_heads, head_dim]
    half* kv_cache_k[LlamaConfig::num_layers];  // each [max_seq, kv_dim]
    half* kv_cache_v[LlamaConfig::num_layers];  // each [max_seq, kv_dim]

    // Token IDs on GPU
    int* token_ids;     // [max_seq]
};

class LlamaInference {
public:
    // Allocate all GPU buffers
    bool init(int max_seq_len = LlamaConfig::max_seq_len);

    // Free all GPU buffers
    void cleanup();

    // Run forward pass on token sequence
    // Returns logits for the last token (on GPU)
    // seq_len: number of tokens in this pass
    // start_pos: position offset (0 for prefill, past_len for decode)
    void forward(const LlamaWeights& weights, int seq_len, int start_pos);

    // Greedy sampling: argmax of logits for last token
    int sample_greedy(int seq_len);

    // Top-p (nucleus) sampling
    int sample_top_p(int seq_len, float temperature = 0.8f, float top_p = 0.9f);

    // Copy token IDs to GPU
    void set_tokens(const int* ids, int count, int offset = 0);

    // Get current KV cache length
    int kv_len() const { return kv_len_; }

    // Use flash attention instead of naive
    void set_use_flash(bool use_flash) { use_flash_ = use_flash; }

    // Use wmma GEMM instead of tiled
    void set_use_wmma(bool use_wmma) { use_wmma_ = use_wmma; }

private:
    InferenceBuffers buf_{};
    int max_seq_ = 0;
    int kv_len_ = 0;
    bool use_flash_ = true;
    bool use_wmma_ = true;

    // Internal: GEMM dispatch (picks wmma or tiled based on config)
    void gemm(half* C, const half* A, const half* B, int M, int N, int K);
};
