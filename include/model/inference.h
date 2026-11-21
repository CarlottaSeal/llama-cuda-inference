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
//    gate:       [max_seq, intermediate]    — FFN activation silu(gate)·up
//    gate_up:    [max_seq, 2·intermediate]  — fused gate|up GEMM output (prefill only)
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
    half* gate;         // [max_seq, intermediate_size] — silu(gate)·up, input to down_proj
    half* gate_up;      // [max_seq, 2*intermediate_size], interleaved gate/up groups
    half* ffn_out;      // [max_seq, hidden_size]
    half* logits_half;  // [vocab_size] — LM head GEMM output
    float* logits;      // [vocab_size] — FP32, last token only

    // KV Cache
    // Layout: kv_cache[layer][0=K,1=V] points to [max_seq, num_kv_heads, head_dim]
    half* kv_cache_k[LlamaConfig::num_layers];  // each [max_seq, kv_dim]
    half* kv_cache_v[LlamaConfig::num_layers];  // each [max_seq, kv_dim]

    // Token IDs on GPU
    int* token_ids;     // [max_seq]

    // Decode state on the GPU, so one captured decode step serves every position
    int* d_pos;         // [1] position of the token the next decode step consumes
    int* d_argmax;      // [1] sample_greedy result
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

    // Replay decode steps from a CUDA Graph (default) instead of launching each kernel
    void set_use_graph(bool use_graph) { use_graph_ = use_graph; }

    // Capture the decode-step graph now (otherwise the first decode_greedy does it).
    // Returns false when graphs are off or this configuration can't use them.
    bool prepare_decode_graph(const LlamaWeights& weights) { return use_graph_ && graph_decode_ready(weights); }

    // Greedy decode. first_token sits at position pos (already sampled from the prefill
    // logits). Runs max_steps single-token forwards; forward k consumes the token at
    // pos + k and produces the token at pos + k + 1, which is passed to on_token for
    // k + 1 < max_steps. Stops early when on_token returns false. With graphs, the host
    // runs one step behind the GPU, so one extra forward may run after a stop.
    // Returns the number of forwards run.
    template <typename OnToken>
    int decode_greedy(const LlamaWeights& weights, int first_token, int pos, int max_steps, OnToken on_token);

private:
    InferenceBuffers buf_{};
    int max_seq_ = 0;
    int kv_len_ = 0;
    bool use_flash_ = true;
    bool use_wmma_ = true;
    bool use_graph_ = true;

    // Decode-step graph (captured on first use) and the host side of its pipeline
    void* graph_exec_ = nullptr;   // cudaGraphExec_t
    const LlamaWeights* graph_weights_ = nullptr;
    int* h_tokens_ = nullptr;      // pinned [2]
    void* events_[2] = {};         // cudaEvent_t

    // Internal: GEMM dispatch (picks wmma or tiled based on config)
    void gemm(half* C, const half* A, const half* B, int M, int N, int K);

    // Forward pass body. With d_pos set (decode, seq_len 1) every position-dependent
    // argument is read from the device, so the launches can be captured once.
    void forward_impl(const LlamaWeights& weights, int seq_len, int start_pos, const int* d_pos);

    // Graph-mode decode helpers (non-template parts of decode_greedy)
    bool graph_decode_ready(const LlamaWeights& weights);
    void graph_decode_start(int first_token, int pos);
    void graph_decode_launch(int k, int pos, bool copy_out);
    int  graph_decode_wait(int k);
    void graph_decode_finish(int pos, int steps);
};

template <typename OnToken>
int LlamaInference::decode_greedy(const LlamaWeights& weights, int first_token, int pos,
                                  int max_steps, OnToken on_token) {
    if (max_steps <= 0) return 0;
    if (!use_graph_ || !graph_decode_ready(weights)) {
        // Eager path: one launch per kernel, a host round trip per token
        int token = first_token;
        for (int k = 0; k < max_steps; k++) {
            set_tokens(&token, 1, pos + k);
            forward(weights, 1, pos + k);
            token = sample_greedy(1);
            if (k + 1 < max_steps && !on_token(token)) return k + 1;
        }
        return max_steps;
    }

    // Graph path, two deep: step k is enqueued before waiting for step k-1's token,
    // so the GPU never idles on the host. Step k's token exists only if pos + k + 1 is
    // still inside the token buffer.
    graph_decode_start(first_token, pos);
    auto wanted = [&](int k) { return k + 1 < max_steps && pos + k + 1 < max_seq_; };
    // The last step's token is never wanted (k + 1 == max_steps), so every wanted token
    // is collected inside the loop, one iteration after its step was enqueued.
    int k = 0;
    for (; k < max_steps; k++) {
        graph_decode_launch(k, pos, wanted(k));
        if (k > 0 && wanted(k - 1) && !on_token(graph_decode_wait(k - 1))) { k++; break; }
    }
    graph_decode_finish(pos, k);
    return k;
}
