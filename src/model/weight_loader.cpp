#include "model/weight_loader.h"
#include "utils/cuda_utils.cuh"

#include <cstdio>
#include <cstring>
#include <filesystem>

namespace fs = std::filesystem;

// ============================================================
//  Load a single tensor: mmap data → GPU memory
//  Supports F16 (direct copy) and F32 (convert to FP16)
// ============================================================

half* WeightLoader::load_tensor(SafetensorsFile& file,
                                 const std::string& name,
                                 size_t expected_elements) {
    const TensorInfo* info = file.tensor(name);
    if (!info) {
        error_ = "Tensor not found: " + name;
        return nullptr;
    }

    // Compute number of elements from shape
    size_t num_elements = 1;
    for (auto d : info->shape) num_elements *= d;

    if (num_elements != expected_elements) {
        error_ = "Shape mismatch for " + name +
                 ": expected " + std::to_string(expected_elements) +
                 " elements, got " + std::to_string(num_elements);
        return nullptr;
    }

    const void* src = file.tensor_data(*info);
    if (!src) {
        error_ = "Failed to get data for: " + name;
        return nullptr;
    }

    size_t gpu_size = num_elements * sizeof(half);

    half* d_ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, gpu_size));
    allocations_.push_back(d_ptr);
    gpu_bytes_ += gpu_size;

    if (info->dtype == "F16") {
        // Direct copy — safetensors F16 is IEEE 754 half, same as CUDA half
        CUDA_CHECK(cudaMemcpy(d_ptr, src, gpu_size, cudaMemcpyHostToDevice));
    } else if (info->dtype == "BF16") {
        // BF16 → FP16 conversion on CPU then copy
        // BF16: 1 sign + 8 exp + 7 mantissa
        // FP16: 1 sign + 5 exp + 10 mantissa
        // Simple approach: BF16 → FP32 → FP16
        const uint16_t* bf16_data = (const uint16_t*)src;
        std::vector<half> host_fp16(num_elements);
        for (size_t i = 0; i < num_elements; i++) {
            // BF16 to FP32: shift left by 16 bits
            uint32_t fp32_bits = (uint32_t)bf16_data[i] << 16;
            float fp32_val;
            std::memcpy(&fp32_val, &fp32_bits, sizeof(float));
            host_fp16[i] = __float2half(fp32_val);
        }
        CUDA_CHECK(cudaMemcpy(d_ptr, host_fp16.data(), gpu_size, cudaMemcpyHostToDevice));
    } else if (info->dtype == "F32") {
        // FP32 → FP16 conversion on CPU then copy
        const float* fp32_data = (const float*)src;
        std::vector<half> host_fp16(num_elements);
        for (size_t i = 0; i < num_elements; i++) {
            host_fp16[i] = __float2half(fp32_data[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_ptr, host_fp16.data(), gpu_size, cudaMemcpyHostToDevice));
    } else {
        error_ = "Unsupported dtype: " + info->dtype + " for " + name;
        return nullptr;
    }

    return d_ptr;
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
    const TensorInfo* lm_head_info = file.tensor("lm_head.weight");
    if (lm_head_info) {
        weights.lm_head = load_tensor(file, "lm_head.weight",
                                       C::vocab_size * C::hidden_size);
        if (!weights.lm_head) return false;
    } else {
        // Tied embeddings: lm_head shares embedding weight
        weights.lm_head = weights.embedding;
        printf("  lm_head tied to embedding\n");
    }

    // Per-layer weights
    for (int i = 0; i < C::num_layers; i++) {
        auto& layer = weights.layers[i];
        std::string prefix = "model.layers." + std::to_string(i) + ".";

        layer.q_proj = load_tensor(file, prefix + "self_attn.q_proj.weight",
                                    C::hidden_size * C::q_dim);
        if (!layer.q_proj) return false;

        layer.k_proj = load_tensor(file, prefix + "self_attn.k_proj.weight",
                                    C::hidden_size * C::kv_dim);
        if (!layer.k_proj) return false;

        layer.v_proj = load_tensor(file, prefix + "self_attn.v_proj.weight",
                                    C::hidden_size * C::kv_dim);
        if (!layer.v_proj) return false;

        layer.o_proj = load_tensor(file, prefix + "self_attn.o_proj.weight",
                                    C::q_dim * C::hidden_size);
        if (!layer.o_proj) return false;

        layer.gate_proj = load_tensor(file, prefix + "mlp.gate_proj.weight",
                                       C::hidden_size * C::intermediate_size);
        if (!layer.gate_proj) return false;

        layer.up_proj = load_tensor(file, prefix + "mlp.up_proj.weight",
                                     C::hidden_size * C::intermediate_size);
        if (!layer.up_proj) return false;

        layer.down_proj = load_tensor(file, prefix + "mlp.down_proj.weight",
                                       C::intermediate_size * C::hidden_size);
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
