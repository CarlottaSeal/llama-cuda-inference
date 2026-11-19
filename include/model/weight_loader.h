#pragma once

#include "model/llama_config.h"
#include "model/safetensors.h"
#include <string>
#include <vector>

// ============================================================
//  Weight Loader
//
//  Loads Llama 3.2 weights from safetensors files into
//  LlamaWeights struct on GPU.
//
//  Llama 3.2 1B uses a single safetensors file:
//    model.safetensors (~2.5 GB)
//
//  Tensor name mapping (HuggingFace format):
//    model.embed_tokens.weight           → embedding
//    model.norm.weight                   → final_norm
//    lm_head.weight                      → lm_head
//    model.layers.{i}.self_attn.q_proj.weight   → layers[i].q_proj
//    model.layers.{i}.self_attn.k_proj.weight   → layers[i].k_proj
//    model.layers.{i}.self_attn.v_proj.weight   → layers[i].v_proj
//    model.layers.{i}.self_attn.o_proj.weight   → layers[i].o_proj
//    model.layers.{i}.mlp.gate_proj.weight      → layers[i].gate_proj
//    model.layers.{i}.mlp.up_proj.weight        → layers[i].up_proj
//    model.layers.{i}.mlp.down_proj.weight      → layers[i].down_proj
//    model.layers.{i}.input_layernorm.weight     → layers[i].attn_norm
//    model.layers.{i}.post_attention_layernorm.weight → layers[i].ffn_norm
// ============================================================

class WeightLoader {
public:
    // Load all weights from safetensors file(s) into GPU memory.
    // Returns false on error (check error()).
    bool load(const std::string& model_dir, LlamaWeights& weights);

    // Free all GPU memory allocated by load()
    void free_weights(LlamaWeights& weights);

    const std::string& error() const { return error_; }

    // Total GPU memory allocated (bytes)
    size_t gpu_bytes() const { return gpu_bytes_; }

private:
    std::string error_;
    size_t gpu_bytes_ = 0;
    std::vector<half*> allocations_;  // track for cleanup

    // Copy a tensor from safetensors to GPU, handling dtype conversion.
    // transpose=true stores a 2-D tensor with its two dims swapped.
    half* load_tensor(SafetensorsFile& file, const std::string& name,
                      size_t expected_elements, bool transpose = false);
};
