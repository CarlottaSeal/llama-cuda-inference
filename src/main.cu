// ============================================================
//  CUDA LLM Inference Engine — Llama 3.2 1B
//
//  End-to-end: load model → tokenize → prefill → decode → print
//
//  Usage: ./llama_inference <model_dir> "prompt text" [max_tokens]
//
//  model_dir should contain:
//    - model.safetensors (or model-00001-of-*.safetensors)
//    - tokenizer.json
// ============================================================

#include "utils/cuda_utils.cuh"
#include "model/llama_config.h"
#include "model/inference.h"
#include "model/weight_loader.h"
#include "model/tokenizer.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

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

int main(int argc, char** argv) {
    // ---- Parse arguments ----
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <model_dir> \"prompt\" [max_tokens]\n", argv[0]);
        fprintf(stderr, "  model_dir: directory with model.safetensors and tokenizer.json\n");
        fprintf(stderr, "  prompt: input text\n");
        fprintf(stderr, "  max_tokens: max new tokens to generate (default: 128)\n");
        return 1;
    }

    std::string model_dir = argv[1];
    std::string prompt = argv[2];
    int max_new_tokens = (argc > 3) ? atoi(argv[3]) : 128;

    printf("=== CUDA LLM Inference Engine ===\n");
    print_gpu_info();
    print_config();
    printf("\n");

    // ---- Step 1: Load tokenizer ----
    printf("--- Loading tokenizer ---\n");
    Tokenizer tokenizer;
    std::string tok_path = model_dir + "/tokenizer.json";
    if (!tokenizer.load(tok_path)) {
        fprintf(stderr, "Failed to load tokenizer: %s\n", tokenizer.error().c_str());
        return 1;
    }
    printf("\n");

    // ---- Step 2: Tokenize prompt ----
    printf("--- Tokenizing ---\n");
    std::vector<int> prompt_tokens = tokenizer.encode(prompt, /*add_bos=*/true);
    printf("Prompt: \"%s\"\n", prompt.c_str());
    printf("Tokens: %zu [", prompt_tokens.size());
    for (size_t i = 0; i < prompt_tokens.size() && i < 16; i++) {
        printf("%d", prompt_tokens[i]);
        if (i + 1 < prompt_tokens.size()) printf(", ");
    }
    if (prompt_tokens.size() > 16) printf(", ...");
    printf("]\n\n");

    if (prompt_tokens.empty()) {
        fprintf(stderr, "Empty prompt after tokenization\n");
        return 1;
    }

    int prompt_len = (int)prompt_tokens.size();
    int total_len = prompt_len + max_new_tokens;
    if (total_len > LlamaConfig::max_seq_len) {
        total_len = LlamaConfig::max_seq_len;
        max_new_tokens = total_len - prompt_len;
        printf("Warning: clamped total length to %d (max_seq=%d)\n",
               total_len, LlamaConfig::max_seq_len);
    }

    // ---- Step 3: Load model weights ----
    printf("--- Loading model weights ---\n");
    LlamaWeights weights{};
    WeightLoader loader;

    auto t_load_start = std::chrono::high_resolution_clock::now();
    if (!loader.load(model_dir, weights)) {
        fprintf(stderr, "Failed to load weights: %s\n", loader.error().c_str());
        return 1;
    }
    auto t_load_end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load_end - t_load_start).count();
    printf("Weight loading: %.1f ms\n\n", load_ms);

    // ---- Step 4: Initialize inference engine ----
    printf("--- Initializing inference ---\n");
    LlamaInference engine;
    engine.init(total_len);
    engine.set_use_flash(true);
    engine.set_use_wmma(true);
    printf("\n");

    // ---- Step 5: Prefill ----
    printf("--- Prefill (%d tokens) ---\n", prompt_len);
    engine.set_tokens(prompt_tokens.data(), prompt_len, 0);

    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    engine.forward(weights, prompt_len, /*start_pos=*/0);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    printf("Prefill: %.1f ms (%.0f tok/s)\n\n", prefill_ms, prompt_len / (prefill_ms / 1000.0));

    // ---- Step 6: Autoregressive decode ----
    printf("--- Generating (max %d tokens) ---\n", max_new_tokens);
    printf("Prompt: %s", prompt.c_str());
    fflush(stdout);

    std::vector<int> generated;
    auto t_decode_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < max_new_tokens; step++) {
        // Sample next token from logits
        int next_token = engine.sample_greedy(1);

        // Check for EOS
        if (next_token == tokenizer.eos_id()) {
            printf(" [EOS]");
            break;
        }

        generated.push_back(next_token);

        // Print the decoded token
        std::string token_str = tokenizer.decode_token(next_token);
        printf("%s", token_str.c_str());
        fflush(stdout);

        // Feed token back for next step
        engine.set_tokens(&next_token, 1, prompt_len + step);
        engine.forward(weights, /*seq_len=*/1, /*start_pos=*/prompt_len + step);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto t_decode_end = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(t_decode_end - t_decode_start).count();
    printf("\n\n");

    // ---- Step 7: Stats ----
    int gen_tokens = (int)generated.size();
    printf("=== Performance ===\n");
    printf("Prefill:  %d tokens in %.1f ms = %.1f tok/s\n",
           prompt_len, prefill_ms, prompt_len / (prefill_ms / 1000.0));
    if (gen_tokens > 0) {
        printf("Decode:   %d tokens in %.1f ms = %.1f tok/s\n",
               gen_tokens, decode_ms, gen_tokens / (decode_ms / 1000.0));
        printf("Time to first token: %.1f ms\n", prefill_ms);
    }
    printf("Total:    %.1f ms\n", prefill_ms + decode_ms);

    // ---- Cleanup ----
    engine.cleanup();
    loader.free_weights(weights);

    printf("\nDone.\n");
    return 0;
}
