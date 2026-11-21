// ============================================================
//  Llama Inference Engine — Forward Pass Implementation
//
//  Orchestrates all kernels into a complete Llama forward pass:
//    Embedding → 16× { Attn Block, FFN Block } → LM Head
//
//  Attention Block:
//    RMSNorm → Q/K/V Projection (GEMM) → RoPE → KV Cache Update
//    → Attention → O Projection (GEMM) → Residual Add
//
//  FFN Block:
//    RMSNorm → Gate+Up Projection (GEMM) → SiLU×Mul
//    → Down Projection (GEMM) → Residual Add
//
//  Weight layout: HF stores linear weights as [out, in]; the loader
//  transposes them once to [in, out], so every projection here is a
//  plain C[M,N] = A[M,K] × B[K,N] with no per-forward transposes.
// ============================================================

#include "model/inference.h"
#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"

#include <cstdio>
#include <cstring>
#include <cfloat>
#include <cmath>

// ============================================================
//  Argmax kernel for greedy sampling
// ============================================================

__global__ void argmax_kernel(const float* logits, int* result, int n) {
    // Single block, find argmax
    extern __shared__ char smem[];
    float* s_max = (float*)smem;
    int* s_idx = (int*)(s_max + blockDim.x);

    int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    int local_idx = 0;

    for (int i = tid; i < n; i += blockDim.x) {
        if (logits[i] > local_max) {
            local_max = logits[i];
            local_idx = i;
        }
    }

    s_max[tid] = local_max;
    s_idx[tid] = local_idx;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) {
                s_max[tid] = s_max[tid + s];
                s_idx[tid] = s_idx[tid + s];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = s_idx[0];
    }
}

// ============================================================
//  Half-to-float and float-to-half conversion kernels
//  For logits computation (FP32 accumulation)
// ============================================================

__global__ void half_to_float_kernel(float* out, const half* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __half2float(in[i]);
    }
}

static void launch_half_to_float(float* out, const half* in, int n) {
    int block = 256;
    int grid = cdiv(n, block);
    half_to_float_kernel<<<grid, block>>>(out, in, n);
}

// ============================================================
//  Split the fused QKV projection output
//
//  qkv[seq, q_dim | kv_dim | kv_dim] → q[seq, q_dim], and K/V
//  written straight into this layer's KV cache rows. Replaces the
//  two cache cudaMemcpys the separate-projection path needed.
// ============================================================

__global__ void qkv_split_kernel(half* q, half* k_dst, half* v_dst,
                                 const half* qkv, int seq_len) {
    using C = LlamaConfig;
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // index in half2 units
    constexpr int row2 = C::qkv_dim / 2;
    if (i >= seq_len * row2) return;
    int r = i / row2, c = (i % row2) * 2;
    half2 val = reinterpret_cast<const half2*>(qkv)[i];
    half* dst;
    if (c < C::q_dim)                  dst = q     + (size_t)r * C::q_dim  + c;
    else if (c < C::q_dim + C::kv_dim) dst = k_dst + (size_t)r * C::kv_dim + (c - C::q_dim);
    else                               dst = v_dst + (size_t)r * C::kv_dim + (c - C::q_dim - C::kv_dim);
    *reinterpret_cast<half2*>(dst) = val;
}

static void launch_qkv_split(half* q, half* k_dst, half* v_dst, const half* qkv, int seq_len) {
    int n2 = seq_len * LlamaConfig::qkv_dim / 2;
    qkv_split_kernel<<<cdiv(n2, 256), 256>>>(q, k_dst, v_dst, qkv, seq_len);
}

// ============================================================
//  Buffer allocation / cleanup
// ============================================================

bool LlamaInference::init(int max_seq_len) {
    using C = LlamaConfig;
    max_seq_ = max_seq_len;

    size_t total = 0;
    auto alloc = [&](half** ptr, size_t count, const char* name) {
        size_t bytes = count * sizeof(half);
        CUDA_CHECK(cudaMalloc(ptr, bytes));
        CUDA_CHECK(cudaMemset(*ptr, 0, bytes));
        total += bytes;
    };

    auto alloc_f = [&](float** ptr, size_t count, const char* name) {
        size_t bytes = count * sizeof(float);
        CUDA_CHECK(cudaMalloc(ptr, bytes));
        CUDA_CHECK(cudaMemset(*ptr, 0, bytes));
        total += bytes;
    };

    auto alloc_i = [&](int** ptr, size_t count, const char* name) {
        size_t bytes = count * sizeof(int);
        CUDA_CHECK(cudaMalloc(ptr, bytes));
        total += bytes;
    };

    // Activation buffers
    alloc(&buf_.hidden,  (size_t)max_seq_ * C::hidden_size, "hidden");
    alloc(&buf_.hidden2, (size_t)max_seq_ * C::hidden_size, "hidden2");
    alloc(&buf_.q,       (size_t)max_seq_ * C::q_dim, "q");
    alloc(&buf_.qkv,     (size_t)max_seq_ * C::qkv_dim, "qkv");
    alloc(&buf_.attn_out,(size_t)max_seq_ * C::q_dim, "attn_out");
    alloc(&buf_.gate,    (size_t)max_seq_ * C::intermediate_size, "gate");
    alloc(&buf_.up,      (size_t)max_seq_ * C::intermediate_size, "up");
    alloc(&buf_.ffn_out, (size_t)max_seq_ * C::hidden_size, "ffn_out");

    // Logits are only computed for the last token
    alloc(&buf_.logits_half, (size_t)C::vocab_size, "logits_half");
    alloc_f(&buf_.logits, (size_t)C::vocab_size, "logits");

    // Token IDs
    alloc_i(&buf_.token_ids, max_seq_, "token_ids");

    // KV Cache: per-layer K and V tensors
    for (int l = 0; l < C::num_layers; l++) {
        alloc(&buf_.kv_cache_k[l], (size_t)max_seq_ * C::kv_dim, "kv_cache_k");
        alloc(&buf_.kv_cache_v[l], (size_t)max_seq_ * C::kv_dim, "kv_cache_v");
    }

    printf("Inference buffers allocated: %.1f MB\n", total / (1024.0 * 1024.0));
    printf("  KV cache: %.1f MB (for %d tokens)\n",
           (size_t)C::num_layers * 2 * max_seq_ * C::kv_dim * sizeof(half) / (1024.0 * 1024.0),
           max_seq_);

    return true;
}

void LlamaInference::cleanup() {
    auto free_ptr = [](auto& ptr) { if (ptr) { cudaFree(ptr); ptr = nullptr; } };

    free_ptr(buf_.hidden);
    free_ptr(buf_.hidden2);
    free_ptr(buf_.q);
    free_ptr(buf_.qkv);
    free_ptr(buf_.attn_out);
    free_ptr(buf_.gate);
    free_ptr(buf_.up);
    free_ptr(buf_.ffn_out);
    free_ptr(buf_.logits_half);
    free_ptr(buf_.logits);
    free_ptr(buf_.token_ids);

    for (int l = 0; l < LlamaConfig::num_layers; l++) {
        free_ptr(buf_.kv_cache_k[l]);
        free_ptr(buf_.kv_cache_v[l]);
    }
}

// ============================================================
//  Token input
// ============================================================

void LlamaInference::set_tokens(const int* ids, int count, int offset) {
    CUDA_CHECK(cudaMemcpy(buf_.token_ids + offset, ids,
                           count * sizeof(int), cudaMemcpyHostToDevice));
}

// ============================================================
//  GEMM dispatch — GEMV for a single row (decode, LM head),
//  otherwise wmma or tiled
// ============================================================

void LlamaInference::gemm(half* C, const half* A, const half* B, int M, int N, int K) {
    if (M == 1 && gemv_supported(N)) {
        launch_gemv(C, A, B, N, K);
    } else if (use_wmma_) {
        launch_gemm_wmma(C, A, B, M, N, K);
    } else {
        launch_gemm_tiled(C, A, B, M, N, K);
    }
}

// ============================================================
//  Forward pass — single call through all layers
//
//  seq_len: number of NEW tokens in this pass
//  start_pos: position of first new token (= past KV length)
//
//  On entry: token_ids[start_pos .. start_pos+seq_len-1] are set
//  On exit:  logits[] has logits for the LAST token
// ============================================================

void LlamaInference::forward(const LlamaWeights& weights, int seq_len, int start_pos) {
    using C = LlamaConfig;

    int kv_len = start_pos + seq_len;  // total KV length after this pass

    // ---- Step 1: Embedding lookup ----
    // hidden[seq_len, hidden_size] = embedding_table[token_ids[start_pos:]]
    launch_embedding(buf_.hidden, weights.embedding,
                     buf_.token_ids + start_pos, seq_len, C::hidden_size);

    // ---- Step 2: Transformer layers ----
    for (int l = 0; l < C::num_layers; l++) {
        const auto& layer = weights.layers[l];

        // ========== Attention Block ==========

        // 2a. RMSNorm(hidden) → hidden2
        launch_rmsnorm(buf_.hidden2, buf_.hidden, layer.attn_norm,
                       seq_len, C::hidden_size, C::rms_norm_eps);

        // 2b. Fused QKV projection: qkv[seq, q|k|v] = hidden2[seq, hidden] × W_qkv
        gemm(buf_.qkv, buf_.hidden2, layer.qkv_proj, seq_len, C::qkv_dim, C::hidden_size);

        // 2c. Split: Q → q buffer, K/V → KV cache rows [start_pos .. start_pos+seq_len)
        half* k_new = buf_.kv_cache_k[l] + (size_t)start_pos * C::kv_dim;
        half* v_new = buf_.kv_cache_v[l] + (size_t)start_pos * C::kv_dim;
        launch_qkv_split(buf_.q, k_new, v_new, buf_.qkv, seq_len);

        // 2d. RoPE on Q and on the new K rows (in place, K inside the cache)
        launch_rope(buf_.q, k_new, seq_len,
                    C::num_q_heads, C::num_kv_heads, C::head_dim,
                    start_pos, C::rope_theta,
                    RopeScaling{C::rope_factor, C::rope_low_freq_factor,
                                C::rope_high_freq_factor, C::rope_original_max_pos});

        // 2h. Attention: Q against full KV cache [0..kv_len)
        float scale = 1.0f / sqrtf((float)C::head_dim);

        if (use_flash_ && seq_len == 1 &&
            flash_decode_supported(C::num_q_heads, C::num_kv_heads, C::head_dim)) {
            launch_flash_decode(
                buf_.attn_out, buf_.q,
                buf_.kv_cache_k[l], buf_.kv_cache_v[l],
                kv_len, C::num_q_heads, C::num_kv_heads, C::head_dim, scale);
        } else if (use_flash_ && flash_prefill_supported(C::num_q_heads, C::num_kv_heads, C::head_dim)) {
            launch_flash_prefill(
                buf_.attn_out, buf_.q,
                buf_.kv_cache_k[l], buf_.kv_cache_v[l],
                seq_len, kv_len, C::num_q_heads, C::num_kv_heads, C::head_dim, scale);
        } else if (use_flash_) {
            launch_flash_attention_gqa(
                buf_.attn_out, buf_.q,
                buf_.kv_cache_k[l], buf_.kv_cache_v[l],
                seq_len, kv_len,
                C::num_q_heads, C::num_kv_heads, C::head_dim, scale);
        } else {
            launch_attention_naive(
                buf_.attn_out, buf_.q,
                buf_.kv_cache_k[l], buf_.kv_cache_v[l],
                seq_len, kv_len,
                C::num_q_heads, C::num_kv_heads, C::head_dim, scale);
        }

        // 2i. O projection: ffn_out[seq, hidden] = attn_out[seq, q_dim] × W_o^T
        gemm(buf_.ffn_out, buf_.attn_out, layer.o_proj, seq_len, C::hidden_size, C::q_dim);

        // 2j. Residual: hidden += ffn_out
        launch_residual_add(buf_.hidden, buf_.hidden, buf_.ffn_out,
                            seq_len * C::hidden_size);

        // ========== FFN Block ==========

        // 2k. RMSNorm(hidden) → hidden2
        launch_rmsnorm(buf_.hidden2, buf_.hidden, layer.ffn_norm,
                       seq_len, C::hidden_size, C::rms_norm_eps);

        // 2l. Gate projection: gate[seq, intermediate] = hidden2 × W_gate^T
        gemm(buf_.gate, buf_.hidden2, layer.gate_proj, seq_len, C::intermediate_size, C::hidden_size);

        // 2m. Up projection: up[seq, intermediate] = hidden2 × W_up^T
        gemm(buf_.up, buf_.hidden2, layer.up_proj, seq_len, C::intermediate_size, C::hidden_size);

        // 2n. SiLU(gate) × up → gate  (in-place)
        launch_silu_mul(buf_.gate, buf_.gate, buf_.up,
                        seq_len * C::intermediate_size);

        // 2o. Down projection: ffn_out[seq, hidden] = gate[seq, intermediate] × W_down^T
        gemm(buf_.ffn_out, buf_.gate, layer.down_proj, seq_len, C::hidden_size, C::intermediate_size);

        // 2p. Residual: hidden += ffn_out
        launch_residual_add(buf_.hidden, buf_.hidden, buf_.ffn_out,
                            seq_len * C::hidden_size);
    }

    // ---- Step 3: Final RMSNorm ----
    launch_rmsnorm(buf_.hidden2, buf_.hidden, weights.final_norm,
                   seq_len, C::hidden_size, C::rms_norm_eps);

    // ---- Step 4: LM Head (only for last token) ----
    // hidden2[last_token, hidden_size] × lm_head^T → logits[vocab_size]
    // lm_head is stored as [hidden_size, vocab_size]
    half* last_hidden = buf_.hidden2 + (size_t)(seq_len - 1) * C::hidden_size;


    // GEMM: [1, hidden] × [hidden, vocab_size] → [1, vocab_size]
    gemm(buf_.logits_half, last_hidden, weights.lm_head, 1, C::vocab_size, C::hidden_size);

    // Convert to float for sampling
    launch_half_to_float(buf_.logits, buf_.logits_half, C::vocab_size);

    // Update KV length
    kv_len_ = kv_len;
}

// ============================================================
//  Greedy sampling: argmax over logits
// ============================================================

int LlamaInference::sample_greedy(int seq_len) {
    int* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    int block = 256;
    size_t smem = block * (sizeof(float) + sizeof(int));
    argmax_kernel<<<1, block, smem>>>(buf_.logits, d_result, LlamaConfig::vocab_size);

    int result;
    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    cudaFree(d_result);

    return result;
}

// ============================================================
//  Top-p sampling (placeholder — greedy first)
// ============================================================

int LlamaInference::sample_top_p(int seq_len, float temperature, float top_p) {
    // TODO: implement proper top-p sampling
    // For now, just use greedy
    return sample_greedy(seq_len);
}
