#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

// ============================================================
//  Kernel unit test framework
//  Each test allocates GPU memory, runs kernel, copies back,
//  and verifies against CPU reference.
// ============================================================

#define TEST_ASSERT(cond, msg) do {                                    \
    if (!(cond)) {                                                     \
        fprintf(stderr, "  FAIL: %s (line %d)\n", msg, __LINE__);     \
        return false;                                                  \
    }                                                                  \
} while (0)

static bool approx_equal(float a, float b, float tol = 1e-2f) {
    return fabsf(a - b) < tol;
}

// Helper: create half array from float array
static std::vector<half> to_half(const std::vector<float>& f) {
    std::vector<half> h(f.size());
    for (size_t i = 0; i < f.size(); i++) {
        h[i] = __float2half(f[i]);
    }
    return h;
}

// Helper: convert half array back to float
static std::vector<float> to_float(const std::vector<half>& h) {
    std::vector<float> f(h.size());
    for (size_t i = 0; i < h.size(); i++) {
        f[i] = __half2float(h[i]);
    }
    return f;
}

// ============================================================
//  M1 Tests: Element-wise kernels
// ============================================================

bool test_residual_add() {
    printf("Testing Residual Add... ");

    const int N = 2048;  // realistic: one token's hidden vector
    std::vector<float> h_a(N), h_b(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i % 100) * 0.01f;
        h_b[i] = (float)(i % 50) * 0.02f - 0.5f;
    }

    auto h_a_h = to_half(h_a);
    auto h_b_h = to_half(h_b);

    half *d_a, *d_b, *d_out;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a_h.data(), N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b_h.data(), N * sizeof(half), cudaMemcpyHostToDevice));

    launch_residual_add(d_out, d_a, d_b, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(half), cudaMemcpyDeviceToHost));

    auto result = to_float(h_out);
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        TEST_ASSERT(approx_equal(result[i], expected, 0.05f), "Residual add mismatch");
    }

    // Edge case: out can alias a (in-place residual)
    launch_residual_add(d_a, d_a, d_b, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_a, N * sizeof(half), cudaMemcpyDeviceToHost));
    result = to_float(h_out);
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        TEST_ASSERT(approx_equal(result[i], expected, 0.05f), "In-place residual mismatch");
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    printf("PASSED\n");
    return true;
}

bool test_silu_mul() {
    printf("Testing SiLU+Mul (SwiGLU)... ");

    const int N = 8192;  // realistic: intermediate_size
    std::vector<float> h_gate(N), h_up(N);
    for (int i = 0; i < N; i++) {
        // Range roughly [-3, 3] to exercise sigmoid around its full range
        h_gate[i] = (float)(i % 600) * 0.01f - 3.0f;
        h_up[i] = (float)(i % 200) * 0.01f + 0.5f;
    }

    auto h_gate_h = to_half(h_gate);
    auto h_up_h = to_half(h_up);

    half *d_gate, *d_up, *d_out;
    CUDA_CHECK(cudaMalloc(&d_gate, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_up, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_gate, h_gate_h.data(), N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_up_h.data(), N * sizeof(half), cudaMemcpyHostToDevice));

    launch_silu_mul(d_out, d_gate, d_up, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_out(N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(half), cudaMemcpyDeviceToHost));

    auto result = to_float(h_out);
    for (int i = 0; i < N; i++) {
        float g = h_gate[i];
        float u = h_up[i];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        float expected = g * sigmoid_g * u;
        // FP16 tolerance is wider for larger values
        float tol = fmaxf(0.05f, fabsf(expected) * 0.02f);
        TEST_ASSERT(approx_equal(result[i], expected, tol), "SiLU+Mul mismatch");
    }

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_out);
    printf("PASSED\n");
    return true;
}

bool test_embedding() {
    printf("Testing Embedding Lookup... ");

    const int VOCAB = 1000;
    const int HIDDEN = 2048;
    const int SEQ = 8;

    // Create a small embedding table with recognizable values
    std::vector<float> h_table(VOCAB * HIDDEN);
    for (int v = 0; v < VOCAB; v++) {
        for (int h = 0; h < HIDDEN; h++) {
            // Each row has a unique pattern based on vocab index
            h_table[v * HIDDEN + h] = (float)v * 0.001f + (float)h * 0.0001f;
        }
    }

    int h_ids[SEQ] = {0, 42, 999, 1, 500, 0, 42, 123};

    auto h_table_h = to_half(h_table);

    half* d_table;
    int* d_ids;
    half* d_out;
    CUDA_CHECK(cudaMalloc(&d_table, VOCAB * HIDDEN * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_ids, SEQ * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, SEQ * HIDDEN * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_table, h_table_h.data(), VOCAB * HIDDEN * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids, SEQ * sizeof(int), cudaMemcpyHostToDevice));

    launch_embedding(d_out, d_table, d_ids, SEQ, HIDDEN);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_out(SEQ * HIDDEN);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, SEQ * HIDDEN * sizeof(half), cudaMemcpyDeviceToHost));

    auto result = to_float(h_out);

    // Verify each token's output matches the correct table row
    for (int s = 0; s < SEQ; s++) {
        int id = h_ids[s];
        for (int h = 0; h < HIDDEN; h += 256) {  // spot check every 256th element
            float expected = h_table[id * HIDDEN + h];
            float got = result[s * HIDDEN + h];
            TEST_ASSERT(approx_equal(got, expected, 0.01f), "Embedding mismatch");
        }
    }

    // Verify that duplicate token ids produce identical rows
    // ids[0] == ids[5] == 0, ids[1] == ids[6] == 42
    for (int h = 0; h < HIDDEN; h += 128) {
        float row0 = result[0 * HIDDEN + h];
        float row5 = result[5 * HIDDEN + h];
        TEST_ASSERT(approx_equal(row0, row5, 1e-6f), "Duplicate id mismatch (id=0)");

        float row1 = result[1 * HIDDEN + h];
        float row6 = result[6 * HIDDEN + h];
        TEST_ASSERT(approx_equal(row1, row6, 1e-6f), "Duplicate id mismatch (id=42)");
    }

    cudaFree(d_table);
    cudaFree(d_ids);
    cudaFree(d_out);
    printf("PASSED\n");
    return true;
}

// ============================================================
//  Test runner
// ============================================================

int main() {
    printf("=== Kernel Unit Tests ===\n\n");

    int passed = 0, failed = 0;

    auto run = [&](bool (*fn)(), const char* name) {
        if (fn()) {
            passed++;
        } else {
            fprintf(stderr, "FAILED: %s\n", name);
            failed++;
        }
    };

    // M1: Element-wise
    run(test_residual_add, "Residual Add");
    run(test_silu_mul,     "SiLU+Mul");
    run(test_embedding,    "Embedding");

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return (failed > 0) ? 1 : 0;
}
