// ============================================================
//  CUDA LLM Inference Engine — Llama 3.2 1B
// ============================================================

#include "utils/cuda_utils.cuh"
#include "model/llama_config.h"

#include <cstdio>

static void print_gpu_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %.1f GB, %d SMs)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.multiProcessorCount);
}

static void print_config() {
    using C = LlamaConfig;
    printf("Model: Llama 3.2 1B (L=%d, H=%d, Q=%d, KV=%d, D=%d, FFN=%d, V=%d)\n",
           C::num_layers, C::hidden_size,
           C::num_q_heads, C::num_kv_heads, C::head_dim,
           C::intermediate_size, C::vocab_size);
}

int main() {
    printf("=== CUDA LLM Inference Engine ===\n");
    print_gpu_info();
    print_config();
    return 0;
}
