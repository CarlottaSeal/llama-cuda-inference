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
//  GEMM note on weight layout:
//    HuggingFace stores weights as [out_features, in_features].
//    y = x @ W^T means: for input x[M, K] and weight W[N, K],
//    output y[M, N] = x * W^T.
//    Our GEMM computes C[M,N] = A[M,K] * B[K,N], so we need to
//    transpose: pass weight as B with K=in_features, N=out_features
//    BUT the weight is stored row-major as [N,K], which when read
//    as [K,N] column-major IS the transpose. So we pass:
//      A = input[M, K=hidden], B = weight[K=hidden, N=out] (logical)
//    Actually, since weight is [N,K] in memory (row-major),
//    we do: C[M,N] = A[M,K] × B^T where B is [N,K]
//    Our GEMM assumes row-major: C[M,N] = A[M,K] × B[K,N]
//    So we need to logically transpose the weight.
//    Simplest approach: launch_gemm(out, input, weight^T, M, N, K)
//    where weight^T[K,N] is the weight read as-is since it's [N,K]
//    stored row-major... actually:
//
//    weight stored as [N, K] row-major = [K, N] column-major
//    Our GEMM does C[M,N] = A[M,K] * B[K,N] (row-major)
//    So B should be [K,N] row-major.
//    weight[N,K] row-major → need transpose to get [K,N].
//
//    Alternative: since weight is [out, in] = [N, K] row-major,
//    we can do y = x * W^T by computing:
//      C^T = W * x^T → but this is awkward for batch.
//
//    Cleanest: We'll treat weight as [N,K] and compute
//    C = A * B^T by swapping dimensions in the GEMM call:
//      gemm(C, A, W, M, N, K) with the understanding that
//      internally: C[i,j] = sum_k A[i,k] * W[j,k]
//    This is exactly what happens if we call:
//      gemm(C, A, W, M=seq, N=out_dim, K=in_dim)
//    and our GEMM kernel does C[m,n] = dot(A[m,:], B[:,n])
//    But W is [N,K] so B[:,n] would be a column of W,
//    which is W[n,:] = the n-th row of [N,K] matrix.
//
//    Wait — let me just be precise:
//    A[M,K] row-major: element (i,j) at A[i*K+j]
//    B[K,N] row-major: element (i,j) at B[i*N+j]
//    C[i,j] = sum_k A[i*K+k] * B[k*N+j]
//
//    W is [N,K] row-major: element (i,j) at W[i*K+j]
//    We want C[m,n] = sum_k input[m,k] * W[n,k]
//    = sum_k input[m*K+k] * W[n*K+k]
//
//    But our GEMM gives: C[m,n] = sum_k A[m*K+k] * B[k*N+n]
//    where B is [K,N]. So B[k,n] = B[k*N+n].
//    We need B[k,n] = W[n,k] = W[n*K+k].
//    This means B is W transposed, which we don't have.
//
//    Solution: Add a "transposed B" GEMM variant, OR just
//    handle it in the kernel. For simplicity, since we control
//    the GEMM, let's add a flag. BUT that changes all kernels.
//
//    Simpler approach: just use the existing GEMM and note that
//    for the linear layer y = x W^T where W is [out, in]:
//    We need GEMM(x[M,in], W^T[in,out]) = y[M,out]
//    Since we can't easily transpose, we'll add a simple
//    transpose kernel, or better yet, just write a version
//    of GEMM that does C = A * B^T.
//
//    Actually, the SIMPLEST solution: our GEMM multiplies
//    C[M,N] = A[M,K] * B[K,N]. If we want to compute
//    x * W^T where W is [N,K]:
//    We call gemm with A=x[M,K], and reinterpret W[N,K]
//    as B[K,N] by swapping N and K indices... no, that doesn't work.
//
//    Let's just add launch_gemm_transB() that computes
//    C[M,N] = A[M,K] * B^T[K,N] where B is [N,K].
// ============================================================

#include "model/inference.h"
#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"

#include <cstdio>
#include <cstring>
#include <cfloat>
#include <cmath>

// ============================================================
//  Simple GPU transpose kernel: B_out[K,N] = B_in[N,K]^T
//  Used to prepare weight matrices for our GEMM
// ============================================================

__global__ void transpose_kernel(half* out, const half* in, int rows, int cols) {
    // in[rows, cols] → out[cols, rows]
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

static void launch_transpose(half* out, const half* in, int rows, int cols) {
    dim3 block(32, 32);
    dim3 grid(cdiv(cols, 32), cdiv(rows, 32));
    transpose_kernel<<<grid, block>>>(out, in, rows, cols);
}

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
    alloc(&buf_.kv_proj, (size_t)max_seq_ * C::kv_dim, "kv_proj");
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
    free_ptr(buf_.kv_proj);
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
//  GEMM dispatch — picks wmma or tiled
// ============================================================

void LlamaInference::gemm(half* C, const half* A, const half* B, int M, int N, int K) {
    if (use_wmma_) {
        launch_gemm_wmma(C, A, B, M, N, K);
    } else {
        launch_gemm_tiled(C, A, B, M, N, K);
    }
}

// ============================================================
//  Linear layer: out[M, N] = input[M, K] × weight^T[K, N]
//  where weight is stored as [N, K] (HuggingFace convention).
//
//  We need a scratch buffer for the transposed weight [K, N].
//  To avoid allocating per-call, we use a static persistent buffer.
// ============================================================

// Scratch buffer for weight transpose
static half* s_weight_T = nullptr;
static size_t s_weight_T_size = 0;

static void ensure_weight_T_buffer(size_t elements) {
    if (elements > s_weight_T_size) {
        if (s_weight_T) cudaFree(s_weight_T);
        CUDA_CHECK(cudaMalloc(&s_weight_T, elements * sizeof(half)));
        s_weight_T_size = elements;
    }
}

void LlamaInference::update_kv_cache(int layer_idx, int seq_len, int start_pos) {
    using C = LlamaConfig;
    // Copy K from kv_proj buffer to KV cache at position start_pos
    // kv_proj layout after K projection: [seq_len, kv_dim]
    // KV cache layout: [max_seq, kv_dim], we write at row start_pos

    // K is already in kv_proj, copy to cache
    // We'll call this with the right data pointer set up externally
    size_t row_bytes = C::kv_dim * sizeof(half);
    CUDA_CHECK(cudaMemcpy(
        buf_.kv_cache_k[layer_idx] + (size_t)start_pos * C::kv_dim,
        buf_.kv_proj,
        (size_t)seq_len * row_bytes,
        cudaMemcpyDeviceToDevice));
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

    // Ensure transpose buffer is large enough for largest weight
    // Largest weight: gate/up/down projections = hidden_size × intermediate_size
    size_t max_weight = (size_t)C::hidden_size * C::intermediate_size;
    // lm_head: vocab_size × hidden_size — even bigger
    size_t lm_head_size = (size_t)C::vocab_size * C::hidden_size;
    if (lm_head_size > max_weight) max_weight = lm_head_size;
    ensure_weight_T_buffer(max_weight);

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

        // 2b. Q projection: q[seq, q_dim] = hidden2[seq, hidden] × W_q^T
        //     W_q is [q_dim, hidden_size], we need [hidden_size, q_dim]
        launch_transpose(s_weight_T, layer.q_proj, C::q_dim, C::hidden_size);
        gemm(buf_.q, buf_.hidden2, s_weight_T, seq_len, C::q_dim, C::hidden_size);

        // 2c. K projection: kv_proj[seq, kv_dim] = hidden2[seq, hidden] × W_k^T
        launch_transpose(s_weight_T, layer.k_proj, C::kv_dim, C::hidden_size);
        gemm(buf_.kv_proj, buf_.hidden2, s_weight_T, seq_len, C::kv_dim, C::hidden_size);

        // 2d. RoPE on Q and K (in-place)
        launch_rope(buf_.q, buf_.kv_proj, seq_len,
                    C::num_q_heads, C::num_kv_heads, C::head_dim,
                    start_pos, C::rope_theta,
                    RopeScaling{C::rope_factor, C::rope_low_freq_factor,
                                C::rope_high_freq_factor, C::rope_original_max_pos});

        // 2e. Copy K to cache at [start_pos .. start_pos+seq_len)
        CUDA_CHECK(cudaMemcpy(
            buf_.kv_cache_k[l] + (size_t)start_pos * C::kv_dim,
            buf_.kv_proj,
            (size_t)seq_len * C::kv_dim * sizeof(half),
            cudaMemcpyDeviceToDevice));

        // 2f. V projection: kv_proj[seq, kv_dim] = hidden2[seq, hidden] × W_v^T
        launch_transpose(s_weight_T, layer.v_proj, C::kv_dim, C::hidden_size);
        gemm(buf_.kv_proj, buf_.hidden2, s_weight_T, seq_len, C::kv_dim, C::hidden_size);

        // 2g. Copy V to cache
        CUDA_CHECK(cudaMemcpy(
            buf_.kv_cache_v[l] + (size_t)start_pos * C::kv_dim,
            buf_.kv_proj,
            (size_t)seq_len * C::kv_dim * sizeof(half),
            cudaMemcpyDeviceToDevice));

        // 2h. Attention: Q against full KV cache [0..kv_len)
        float scale = 1.0f / sqrtf((float)C::head_dim);

        if (use_flash_) {
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
        //     W_o is [hidden_size, q_dim]
        launch_transpose(s_weight_T, layer.o_proj, C::hidden_size, C::q_dim);
        gemm(buf_.ffn_out, buf_.attn_out, s_weight_T, seq_len, C::hidden_size, C::q_dim);

        // 2j. Residual: hidden += ffn_out
        launch_residual_add(buf_.hidden, buf_.hidden, buf_.ffn_out,
                            seq_len * C::hidden_size);

        // ========== FFN Block ==========

        // 2k. RMSNorm(hidden) → hidden2
        launch_rmsnorm(buf_.hidden2, buf_.hidden, layer.ffn_norm,
                       seq_len, C::hidden_size, C::rms_norm_eps);

        // 2l. Gate projection: gate[seq, intermediate] = hidden2 × W_gate^T
        launch_transpose(s_weight_T, layer.gate_proj, C::intermediate_size, C::hidden_size);
        gemm(buf_.gate, buf_.hidden2, s_weight_T, seq_len, C::intermediate_size, C::hidden_size);

        // 2m. Up projection: up[seq, intermediate] = hidden2 × W_up^T
        launch_transpose(s_weight_T, layer.up_proj, C::intermediate_size, C::hidden_size);
        gemm(buf_.up, buf_.hidden2, s_weight_T, seq_len, C::intermediate_size, C::hidden_size);

        // 2n. SiLU(gate) × up → gate  (in-place)
        launch_silu_mul(buf_.gate, buf_.gate, buf_.up,
                        seq_len * C::intermediate_size);

        // 2o. Down projection: ffn_out[seq, hidden] = gate[seq, intermediate] × W_down^T
        launch_transpose(s_weight_T, layer.down_proj, C::hidden_size, C::intermediate_size);
        gemm(buf_.ffn_out, buf_.gate, s_weight_T, seq_len, C::hidden_size, C::intermediate_size);

        // 2p. Residual: hidden += ffn_out
        launch_residual_add(buf_.hidden, buf_.hidden, buf_.ffn_out,
                            seq_len * C::hidden_size);
    }

    // ---- Step 3: Final RMSNorm ----
    launch_rmsnorm(buf_.hidden2, buf_.hidden, weights.final_norm,
                   seq_len, C::hidden_size, C::rms_norm_eps);

    // ---- Step 4: LM Head (only for last token) ----
    // hidden2[last_token, hidden_size] × lm_head^T → logits[vocab_size]
    // lm_head is [vocab_size, hidden_size]
    half* last_hidden = buf_.hidden2 + (size_t)(seq_len - 1) * C::hidden_size;

    // Transpose lm_head: [vocab_size, hidden] → [hidden, vocab_size]
    launch_transpose(s_weight_T, weights.lm_head, C::vocab_size, C::hidden_size);

    // GEMM: [1, hidden] × [hidden, vocab_size] → [1, vocab_size]
    gemm(buf_.logits_half, last_hidden, s_weight_T, 1, C::vocab_size, C::hidden_size);

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

// ============================================================
//  Free static transpose buffer on program exit
// ============================================================

static struct WeightTCleanup {
    ~WeightTCleanup() {
        if (s_weight_T) {
            cudaFree(s_weight_T);
            s_weight_T = nullptr;
        }
    }
} s_cleanup;
