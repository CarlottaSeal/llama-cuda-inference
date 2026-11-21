#include "model/weight_loader.h"
#include "utils/cuda_utils.cuh"
#include "kernels/kernels.cuh"  // GATE_UP_GROUP

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================
//  Load a single tensor: mmap data → GPU memory
//  Supports F16 (direct copy) and BF16/F32 (convert to FP16).
//  transpose=true stores a 2-D [rows, cols] tensor as [cols, rows],
//  which turns HF's [out, in] linear weights into the [K, N] layout
//  our GEMMs read, once at load time instead of every forward.
// ============================================================

static void transpose_host(half* dst, const half* src, size_t rows, size_t cols) {
    const size_t T = 64;
    for (size_t r0 = 0; r0 < rows; r0 += T)
        for (size_t c0 = 0; c0 < cols; c0 += T)
            for (size_t r = r0; r < std::min(r0 + T, rows); r++)
                for (size_t c = c0; c < std::min(c0 + T, cols); c++)
                    dst[c * rows + r] = src[r * cols + c];
}

half* WeightLoader::load_tensor(SafetensorsFile& file,
                                 const std::string& name,
                                 size_t expected_elements,
                                 bool transpose) {
    std::vector<half> host;
    if (!read_tensor(file, name, expected_elements, transpose, host)) return nullptr;
    return upload(host);
}

half* WeightLoader::upload(const std::vector<half>& host) {
    size_t gpu_size = host.size() * sizeof(half);
    half* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, gpu_size));
    allocations_.push_back(d_ptr);
    gpu_bytes_ += gpu_size;
    CUDA_CHECK(cudaMemcpy(d_ptr, host.data(), gpu_size, cudaMemcpyHostToDevice));
    return d_ptr;
}

// Column-concatenate several transposed [K, N_i] weights into one [K, ΣN_i],
// so projections that read the same input run as one GEMM/GEMV.
half* WeightLoader::load_fused(SafetensorsFile& file,
                                const std::vector<std::string>& names,
                                const std::vector<size_t>& cols, size_t K) {
    size_t total_cols = 0;
    for (size_t c : cols) total_cols += c;
    std::vector<half> fused(K * total_cols);
    std::vector<half> part;
    size_t col0 = 0;
    for (size_t i = 0; i < names.size(); i++) {
        if (!read_tensor(file, names[i], K * cols[i], /*transpose=*/true, part)) return nullptr;
        for (size_t k = 0; k < K; k++)
            std::memcpy(&fused[k * total_cols + col0], &part[k * cols[i]], cols[i] * sizeof(half));
        col0 += cols[i];
    }
    return upload(fused);
}

half* WeightLoader::load_interleaved(SafetensorsFile& file, const std::string& name_a,
                                      const std::string& name_b, size_t N, size_t K, size_t group) {
    if (N % group != 0) {
        error_ = "Interleave group does not divide " + name_a;
        return nullptr;
    }
    std::vector<half> a, b;
    if (!read_tensor(file, name_a, K * N, /*transpose=*/true, a)) return nullptr;
    if (!read_tensor(file, name_b, K * N, /*transpose=*/true, b)) return nullptr;
    std::vector<half> fused(K * 2 * N);
    for (size_t k = 0; k < K; k++) {
        half* row = &fused[k * 2 * N];
        for (size_t c = 0; c < N; c += group) {
            std::memcpy(row + 2 * c,         &a[k * N + c], group * sizeof(half));
            std::memcpy(row + 2 * c + group, &b[k * N + c], group * sizeof(half));
        }
    }
    return upload(fused);
}

bool WeightLoader::read_tensor(SafetensorsFile& file,
                                const std::string& name,
                                size_t expected_elements,
                                bool transpose,
                                std::vector<half>& out) {
    const TensorInfo* info = file.tensor(name);
    if (!info) {
        error_ = "Tensor not found: " + name;
        return false;
    }

    // Compute number of elements from shape
    size_t num_elements = 1;
    for (auto d : info->shape) num_elements *= d;

    if (num_elements != expected_elements) {
        error_ = "Shape mismatch for " + name +
                 ": expected " + std::to_string(expected_elements) +
                 " elements, got " + std::to_string(num_elements);
        return false;
    }

    if (transpose && info->shape.size() != 2) {
        error_ = "Cannot transpose non-2D tensor: " + name;
        return false;
    }

    const void* src = file.tensor_data(*info);
    if (!src) {
        error_ = "Failed to get data for: " + name;
        return false;
    }

    // Convert to FP16 on the host
    std::vector<half> host_fp16(num_elements);
    if (info->dtype == "F16") {
        // safetensors F16 is IEEE 754 half, same as CUDA half
        std::memcpy(host_fp16.data(), src, num_elements * sizeof(half));
    } else if (info->dtype == "BF16") {
        // BF16 → FP32 is a 16-bit left shift, then round to FP16
        const uint16_t* bf16_data = (const uint16_t*)src;
        for (size_t i = 0; i < num_elements; i++) {
            uint32_t fp32_bits = (uint32_t)bf16_data[i] << 16;
            float fp32_val;
            std::memcpy(&fp32_val, &fp32_bits, sizeof(float));
            host_fp16[i] = __float2half(fp32_val);
        }
    } else if (info->dtype == "F32") {
        const float* fp32_data = (const float*)src;
        for (size_t i = 0; i < num_elements; i++) {
            host_fp16[i] = __float2half(fp32_data[i]);
        }
    } else {
        error_ = "Unsupported dtype: " + info->dtype + " for " + name;
        return false;
    }

    if (transpose) {
        out.resize(num_elements);
        transpose_host(out.data(), host_fp16.data(), info->shape[0], info->shape[1]);
    } else {
        out = std::move(host_fp16);
    }
    return true;
}

// ============================================================
//  Load all Llama weights
// ============================================================

bool WeightLoader::load(const std::string& model_dir, LlamaWeights& weights) {
    // Find safetensors file(s)
    std::string st_path;
    if (fs::exists(model_dir + "/model.safetensors")) {
        st_path = model_dir + "/model.safetensors";
    } else {
        // Multi-file: model-00001-of-00002.safetensors, etc.
        // For Llama 3.2 1B, it's typically a single file
        error_ = "model.safetensors not found in: " + model_dir;
        return false;
    }

    SafetensorsFile file;
    if (!file.open(st_path)) {
        error_ = "Failed to open safetensors: " + file.error();
        return false;
    }

    printf("Loading weights from %s\n", st_path.c_str());
    printf("  Found %zu tensors\n", file.tensor_names().size());

    using C = LlamaConfig;

    // Global weights
    weights.embedding = load_tensor(file, "model.embed_tokens.weight",
                                     C::vocab_size * C::hidden_size);
    if (!weights.embedding) return false;

    weights.final_norm = load_tensor(file, "model.norm.weight", C::hidden_size);
    if (!weights.final_norm) return false;

    // lm_head: some models tie embeddings, some have separate lm_head
    // lm_head is stored transposed ([hidden, vocab]) for the GEMM. With tied
    // embeddings that layout differs from the embedding table's, so it gets
    // its own transposed copy of embed_tokens.
    const bool tied = file.tensor("lm_head.weight") == nullptr;
    weights.lm_head = load_tensor(file, tied ? "model.embed_tokens.weight" : "lm_head.weight",
                                   C::vocab_size * C::hidden_size, /*transpose=*/true);
    if (!weights.lm_head) return false;
    if (tied) printf("  lm_head tied to embedding (transposed copy)\n");

    // Per-layer weights
    for (int i = 0; i < C::num_layers; i++) {
        auto& layer = weights.layers[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";

        layer.qkv_proj = load_fused(file,
                                    {prefix + "self_attn.q_proj.weight",
                                     prefix + "self_attn.k_proj.weight",
                                     prefix + "self_attn.v_proj.weight"},
                                    {C::q_dim, C::kv_dim, C::kv_dim}, C::hidden_size);
        if (!layer.qkv_proj) return false;

        layer.o_proj = load_tensor(file, prefix + "self_attn.o_proj.weight",
                                    C::q_dim * C::hidden_size, /*transpose=*/true);
        if (!layer.o_proj) return false;

        layer.gate_up_proj = load_interleaved(file, prefix + "mlp.gate_proj.weight",
                                              prefix + "mlp.up_proj.weight",
                                              C::intermediate_size, C::hidden_size, GATE_UP_GROUP);
        if (!layer.gate_up_proj) return false;

        layer.down_proj = load_tensor(file, prefix + "mlp.down_proj.weight",
                                       C::intermediate_size * C::hidden_size, /*transpose=*/true);
        if (!layer.down_proj) return false;

        layer.attn_norm = load_tensor(file, prefix + "input_layernorm.weight",
                                       C::hidden_size);
        if (!layer.attn_norm) return false;

        layer.ffn_norm = load_tensor(file, prefix + "post_attention_layernorm.weight",
                                      C::hidden_size);
        if (!layer.ffn_norm) return false;

        printf("  Layer %d loaded\n", i);
    }

    printf("  Total GPU memory: %.1f MB\n", gpu_bytes_ / (1024.0 * 1024.0));
    return true;
}

void WeightLoader::free_weights(LlamaWeights& weights) {
    for (half* ptr : allocations_) {
        if (ptr) cudaFree(ptr);
    }
    allocations_.clear();
    gpu_bytes_ = 0;
    std::memset(&weights, 0, sizeof(weights));
}
