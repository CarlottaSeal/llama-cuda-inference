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
//  M2 Tests: Reduction kernels
// ============================================================

bool test_rmsnorm() {
    printf("Testing RMSNorm... ");

    const int ROWS = 4;    // small batch of tokens
    const int COLS = 2048;  // hidden_size
    const float EPS = 1e-5f;

    // Input: values in a reasonable range for hidden states
    std::vector<float> h_x(ROWS * COLS);
    std::vector<float> h_weight(COLS);
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            // Mix of positive and negative values
            h_x[r * COLS + c] = ((r * 137 + c * 31) % 200 - 100) * 0.01f;
        }
    }
    for (int c = 0; c < COLS; c++) {
        h_weight[c] = 0.5f + ((c % 100) * 0.01f);  // weights around 0.5 - 1.5
    }

    auto h_x_h = to_half(h_x);
    auto h_weight_h = to_half(h_weight);

    half *d_x, *d_weight, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, ROWS * COLS * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weight, COLS * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, ROWS * COLS * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x_h.data(), ROWS * COLS * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight_h.data(), COLS * sizeof(half), cudaMemcpyHostToDevice));

    launch_rmsnorm(d_out, d_x, d_weight, ROWS, COLS, EPS);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_out(ROWS * COLS);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, ROWS * COLS * sizeof(half), cudaMemcpyDeviceToHost));
    auto result = to_float(h_out);

    // CPU reference: RMSNorm
    for (int r = 0; r < ROWS; r++) {
        // Compute RMS
        float ss = 0.0f;
        for (int c = 0; c < COLS; c++) {
            float val = h_x[r * COLS + c];
            ss += val * val;
        }
        float rms_inv = 1.0f / sqrtf(ss / COLS + EPS);

        // Check each element
        for (int c = 0; c < COLS; c += 64) {  // spot check every 64th
            float expected = h_x[r * COLS + c] * rms_inv * h_weight[c];
            float got = result[r * COLS + c];
            float tol = fmaxf(0.05f, fabsf(expected) * 0.05f);
            TEST_ASSERT(approx_equal(got, expected, tol), "RMSNorm mismatch");
        }
    }

    // Verify normalization property: after RMSNorm (ignoring weight),
    // the RMS of the output / weight should be ~1.0
    for (int r = 0; r < ROWS; r++) {
        float ss = 0.0f;
        for (int c = 0; c < COLS; c++) {
            float val = result[r * COLS + c] / h_weight[c];
            ss += val * val;
        }
        float rms = sqrtf(ss / COLS);
        TEST_ASSERT(approx_equal(rms, 1.0f, 0.1f), "RMSNorm output RMS not ~1.0");
    }

    // Test in-place: out == x
    CUDA_CHECK(cudaMemcpy(d_x, h_x_h.data(), ROWS * COLS * sizeof(half), cudaMemcpyHostToDevice));
    launch_rmsnorm(d_x, d_x, d_weight, ROWS, COLS, EPS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_x, ROWS * COLS * sizeof(half), cudaMemcpyDeviceToHost));
    auto result_inplace = to_float(h_out);
    for (int i = 0; i < ROWS * COLS; i += 256) {
        TEST_ASSERT(approx_equal(result[i], result_inplace[i], 1e-4f),
                    "In-place RMSNorm mismatch");
    }

    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_out);
    printf("PASSED\n");
    return true;
}

bool test_softmax() {
    printf("Testing Softmax... ");

    // Test 1: small row (attention-like)
    const int ROWS1 = 2;
    const int COLS1 = 128;

    std::vector<float> h_x1(ROWS1 * COLS1);
    for (int i = 0; i < ROWS1 * COLS1; i++) {
        h_x1[i] = (float)(i % 50) * 0.1f - 2.5f;  // range [-2.5, 2.4]
    }

    float* d_x1;
    CUDA_CHECK(cudaMalloc(&d_x1, ROWS1 * COLS1 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x1, h_x1.data(), ROWS1 * COLS1 * sizeof(float), cudaMemcpyHostToDevice));

    launch_softmax(d_x1, ROWS1, COLS1);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result1(ROWS1 * COLS1);
    CUDA_CHECK(cudaMemcpy(result1.data(), d_x1, ROWS1 * COLS1 * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify: each row sums to 1, all values in [0, 1]
    for (int r = 0; r < ROWS1; r++) {
        float row_sum = 0.0f;
        for (int c = 0; c < COLS1; c++) {
            float val = result1[r * COLS1 + c];
            TEST_ASSERT(val >= 0.0f && val <= 1.0f, "Softmax value out of [0,1]");
            row_sum += val;
        }
        TEST_ASSERT(approx_equal(row_sum, 1.0f, 1e-4f), "Softmax row doesn't sum to 1");
    }

    // Verify against CPU reference
    for (int r = 0; r < ROWS1; r++) {
        float max_val = -FLT_MAX;
        for (int c = 0; c < COLS1; c++) {
            max_val = fmaxf(max_val, h_x1[r * COLS1 + c]);
        }
        float sum = 0.0f;
        for (int c = 0; c < COLS1; c++) {
            sum += expf(h_x1[r * COLS1 + c] - max_val);
        }
        for (int c = 0; c < COLS1; c += 8) {
            float expected = expf(h_x1[r * COLS1 + c] - max_val) / sum;
            TEST_ASSERT(approx_equal(result1[r * COLS1 + c], expected, 1e-5f),
                        "Softmax mismatch vs CPU reference");
        }
    }
    cudaFree(d_x1);

    // Test 2: large row (vocab-size = 128256)
    const int ROWS2 = 1;
    const int COLS2 = 128256;
    printf("\n  Subtesting vocab-size softmax (cols=%d)... ", COLS2);

    std::vector<float> h_x2(COLS2);
    for (int i = 0; i < COLS2; i++) {
        h_x2[i] = (float)(i % 1000) * 0.01f - 5.0f;  // range [-5, 5]
    }

    float* d_x2;
    CUDA_CHECK(cudaMalloc(&d_x2, COLS2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x2, h_x2.data(), COLS2 * sizeof(float), cudaMemcpyHostToDevice));

    launch_softmax(d_x2, ROWS2, COLS2);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> result2(COLS2);
    CUDA_CHECK(cudaMemcpy(result2.data(), d_x2, COLS2 * sizeof(float), cudaMemcpyDeviceToHost));

    // Check sum to 1
    float sum2 = 0.0f;
    for (int i = 0; i < COLS2; i++) {
        TEST_ASSERT(result2[i] >= 0.0f, "Softmax negative value");
        sum2 += result2[i];
    }
    TEST_ASSERT(approx_equal(sum2, 1.0f, 1e-3f),
                "Vocab-size softmax doesn't sum to 1");

    // Check monotonicity: larger input → larger probability
    // Input has a clear pattern, so we can verify relative ordering
    // The max input is at indices where (i % 1000) * 0.01 - 5.0 is max → i % 1000 == 999
    float max_prob = 0.0f;
    int max_idx = -1;
    for (int i = 0; i < COLS2; i++) {
        if (result2[i] > max_prob) {
            max_prob = result2[i];
            max_idx = i;
        }
    }
    TEST_ASSERT(max_idx % 1000 == 999, "Softmax argmax incorrect");

    cudaFree(d_x2);

    printf("PASSED\n");
    return true;
}

// ============================================================
//  M3 Tests: RoPE
// ============================================================

bool test_rope() {
    printf("Testing RoPE... ");

    // Llama 3.2 1B params
    const int SEQ_LEN = 4;
    const int NUM_Q_HEADS = 32;
    const int NUM_KV_HEADS = 8;
    const int HEAD_DIM = 64;
    const float THETA = 500000.0f;
    const int START_POS = 0;

    const int Q_SIZE = SEQ_LEN * NUM_Q_HEADS * HEAD_DIM;
    const int K_SIZE = SEQ_LEN * NUM_KV_HEADS * HEAD_DIM;

    // Fill Q and K with recognizable values
    std::vector<float> h_q(Q_SIZE), h_k(K_SIZE);
    for (int i = 0; i < Q_SIZE; i++) {
        h_q[i] = ((i % 200) - 100) * 0.01f;
    }
    for (int i = 0; i < K_SIZE; i++) {
        h_k[i] = ((i % 150) - 75) * 0.01f;
    }

    // Keep a copy for CPU reference
    std::vector<float> ref_q = h_q;
    std::vector<float> ref_k = h_k;

    auto h_q_h = to_half(h_q);
    auto h_k_h = to_half(h_k);

    half *d_q, *d_k;
    CUDA_CHECK(cudaMalloc(&d_q, Q_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_k, K_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q_h.data(), Q_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k_h.data(), K_SIZE * sizeof(half), cudaMemcpyHostToDevice));

    launch_rope(d_q, d_k, SEQ_LEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, START_POS, THETA);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_q_out(Q_SIZE), h_k_out(K_SIZE);
    CUDA_CHECK(cudaMemcpy(h_q_out.data(), d_q, Q_SIZE * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_k_out.data(), d_k, K_SIZE * sizeof(half), cudaMemcpyDeviceToHost));

    auto result_q = to_float(h_q_out);
    auto result_k = to_float(h_k_out);

    // CPU reference RoPE
    auto cpu_rope = [&](std::vector<float>& x, int seq_len, int num_heads, int start_pos) {
        for (int t = 0; t < seq_len; t++) {
            int pos = start_pos + t;
            for (int h = 0; h < num_heads; h++) {
                for (int p = 0; p < HEAD_DIM / 2; p++) {
                    float freq = 1.0f / powf(THETA, (2.0f * p) / (float)HEAD_DIM);
                    float angle = (float)pos * freq;
                    float cos_a = cosf(angle);
                    float sin_a = sinf(angle);

                    long long off = (long long)t * num_heads * HEAD_DIM
                                  + (long long)h * HEAD_DIM + 2 * p;
                    float x0 = x[off];
                    float x1 = x[off + 1];
                    x[off]     = x0 * cos_a - x1 * sin_a;
                    x[off + 1] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
    };

    cpu_rope(ref_q, SEQ_LEN, NUM_Q_HEADS, START_POS);
    cpu_rope(ref_k, SEQ_LEN, NUM_KV_HEADS, START_POS);

    // Verify Q (spot check every 4th head, every pair)
    for (int t = 0; t < SEQ_LEN; t++) {
        for (int h = 0; h < NUM_Q_HEADS; h += 4) {
            for (int d = 0; d < HEAD_DIM; d++) {
                long long idx = (long long)t * NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM + d;
                float tol = fmaxf(0.05f, fabsf(ref_q[idx]) * 0.05f);
                TEST_ASSERT(approx_equal(result_q[idx], ref_q[idx], tol),
                            "RoPE Q mismatch");
            }
        }
    }

    // Verify K (all heads — only 8)
    for (int t = 0; t < SEQ_LEN; t++) {
        for (int h = 0; h < NUM_KV_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                long long idx = (long long)t * NUM_KV_HEADS * HEAD_DIM + h * HEAD_DIM + d;
                float tol = fmaxf(0.05f, fabsf(ref_k[idx]) * 0.05f);
                TEST_ASSERT(approx_equal(result_k[idx], ref_k[idx], tol),
                            "RoPE K mismatch");
            }
        }
    }

    // Property: RoPE preserves vector magnitude (rotation is unitary)
    for (int t = 0; t < SEQ_LEN; t++) {
        for (int h = 0; h < NUM_Q_HEADS; h += 8) {
            float mag_before = 0.0f, mag_after = 0.0f;
            long long base = (long long)t * NUM_Q_HEADS * HEAD_DIM + h * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; d++) {
                mag_before += h_q[base + d] * h_q[base + d];
                mag_after  += result_q[base + d] * result_q[base + d];
            }
            mag_before = sqrtf(mag_before);
            mag_after = sqrtf(mag_after);
            float tol = fmaxf(0.1f, mag_before * 0.05f);
            TEST_ASSERT(approx_equal(mag_before, mag_after, tol),
                        "RoPE doesn't preserve magnitude");
        }
    }

    cudaFree(d_q);
    cudaFree(d_k);

    // Test 2: with start_pos > 0 (decode phase)
    printf("\n  Subtesting decode phase (start_pos=128)... ");

    const int DECODE_SEQ = 1;  // single token in decode
    const int DECODE_START = 128;
    const int DQ_SIZE = DECODE_SEQ * NUM_Q_HEADS * HEAD_DIM;
    const int DK_SIZE = DECODE_SEQ * NUM_KV_HEADS * HEAD_DIM;

    std::vector<float> h_dq(DQ_SIZE), h_dk(DK_SIZE);
    for (int i = 0; i < DQ_SIZE; i++) h_dq[i] = 0.5f;
    for (int i = 0; i < DK_SIZE; i++) h_dk[i] = 0.3f;

    std::vector<float> ref_dq = h_dq, ref_dk = h_dk;

    auto h_dq_h = to_half(h_dq);
    auto h_dk_h = to_half(h_dk);

    half *d_dq, *d_dk;
    CUDA_CHECK(cudaMalloc(&d_dq, DQ_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dk, DK_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_dq, h_dq_h.data(), DQ_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dk, h_dk_h.data(), DK_SIZE * sizeof(half), cudaMemcpyHostToDevice));

    launch_rope(d_dq, d_dk, DECODE_SEQ, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, DECODE_START, THETA);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_dq_out(DQ_SIZE);
    CUDA_CHECK(cudaMemcpy(h_dq_out.data(), d_dq, DQ_SIZE * sizeof(half), cudaMemcpyDeviceToHost));
    auto result_dq = to_float(h_dq_out);

    // CPU reference with start_pos=128
    cpu_rope(ref_dq, DECODE_SEQ, NUM_Q_HEADS, DECODE_START);

    for (int h = 0; h < NUM_Q_HEADS; h += 8) {
        for (int d = 0; d < HEAD_DIM; d++) {
            long long idx = h * HEAD_DIM + d;
            float tol = fmaxf(0.05f, fabsf(ref_dq[idx]) * 0.05f);
            TEST_ASSERT(approx_equal(result_dq[idx], ref_dq[idx], tol),
                        "RoPE decode-phase mismatch");
        }
    }

    // Verify position 128 gives different result than position 0
    // (at least for the low-frequency pairs)
    {
        // head 0, pair 0 — lowest frequency, still should rotate differently at pos=128
        float val_pos0_d0 = ref_q[0];  // from first test: pos=0, head=0, dim=0
        float val_pos128_d0 = ref_dq[0]; // pos=128, head=0, dim=0
        // They had different inputs so we can't compare directly,
        // but we can verify the decode result isn't identical to the input
        bool changed = false;
        for (int d = 0; d < HEAD_DIM; d++) {
            if (fabsf(result_dq[d] - 0.5f) > 0.01f) {
                changed = true;
                break;
            }
        }
        // At pos=128, the rotation angles are non-trivial, so values should change
        // (input was uniform 0.5, output should differ for most dimensions)
        TEST_ASSERT(changed, "RoPE decode result unchanged from input");
    }

    cudaFree(d_dq);
    cudaFree(d_dk);

    printf("PASSED\n");
    return true;
}

// ============================================================
//  M4 Tests: GEMM
// ============================================================

// CPU reference GEMM for correctness verification
static void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,
                     std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

// Helper: verify GEMM result against CPU reference
static bool verify_gemm(const std::vector<float>& result,
                         const std::vector<float>& ref,
                         int M, int N, const char* name) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float got = result[i * N + j];
            float exp = ref[i * N + j];
            // FP16 accumulation tolerance: relative + absolute
            float tol = fmaxf(0.5f, fabsf(exp) * 0.05f);
            if (!approx_equal(got, exp, tol)) {
                fprintf(stderr, "  %s mismatch at [%d][%d]: got %f, expected %f\n",
                        name, i, j, got, exp);
                return false;
            }
        }
    }
    return true;
}

bool test_gemm_naive() {
    printf("Testing GEMM Naive... ");

    // Small test: 64×128 × 128×96 = 64×96
    const int M = 64, N = 96, K = 128;

    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; i++) h_A[i] = ((i % 200) - 100) * 0.01f;
    for (int i = 0; i < K * N; i++) h_B[i] = ((i % 150) - 75) * 0.01f;

    std::vector<float> ref_C(M * N);
    cpu_gemm(h_A, h_B, ref_C, M, N, K);

    auto h_A_h = to_half(h_A);
    auto h_B_h = to_half(h_B);

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A_h.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_h.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    launch_gemm_naive(d_C, d_A, d_B, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_C(M * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    auto result = to_float(h_C);

    bool ok = verify_gemm(result, ref_C, M, N, "naive");
    TEST_ASSERT(ok, "Naive GEMM failed verification");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    printf("PASSED\n");
    return true;
}

bool test_gemm_tiled() {
    printf("Testing GEMM Tiled... ");

    // Non-tile-aligned to test boundary handling: 65×97 × 97×33
    const int M = 65, N = 33, K = 97;

    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; i++) h_A[i] = ((i % 300) - 150) * 0.005f;
    for (int i = 0; i < K * N; i++) h_B[i] = ((i % 250) - 125) * 0.005f;

    std::vector<float> ref_C(M * N);
    cpu_gemm(h_A, h_B, ref_C, M, N, K);

    auto h_A_h = to_half(h_A);
    auto h_B_h = to_half(h_B);

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A_h.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_h.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    launch_gemm_tiled(d_C, d_A, d_B, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_C(M * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    auto result = to_float(h_C);

    bool ok = verify_gemm(result, ref_C, M, N, "tiled");
    TEST_ASSERT(ok, "Tiled GEMM failed verification");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    printf("PASSED\n");
    return true;
}

bool test_gemm_wmma() {
    printf("Testing GEMM wmma... ");

    // Must be 16-aligned for wmma; use realistic Llama sizes
    // QKV projection: seq_len=16 × hidden=2048 → 16×2048 × 2048×2048
    // Use smaller for quick test: 32×64 × 64×48
    const int M = 32, N = 48, K = 64;

    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; i++) h_A[i] = ((i % 200) - 100) * 0.01f;
    for (int i = 0; i < K * N; i++) h_B[i] = ((i % 150) - 75) * 0.01f;

    std::vector<float> ref_C(M * N);
    cpu_gemm(h_A, h_B, ref_C, M, N, K);

    auto h_A_h = to_half(h_A);
    auto h_B_h = to_half(h_B);

    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A_h.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_h.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    launch_gemm_wmma(d_C, d_A, d_B, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_C(M * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    auto result = to_float(h_C);

    bool ok = verify_gemm(result, ref_C, M, N, "wmma");
    TEST_ASSERT(ok, "wmma GEMM failed verification");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    printf("PASSED\n");
    return true;
}

bool test_gemm_realistic() {
    printf("Testing GEMM realistic sizes (Llama linear projection)... ");

    // Llama 3.2 1B: hidden=2048, intermediate=8192
    // QKV projection: [seq, 2048] × [2048, 2048] (simplified)
    // Use seq=8 to keep test fast
    const int M = 8, N = 2048, K = 2048;

    std::vector<float> h_A(M * K), h_B(K * N);
    for (int i = 0; i < M * K; i++) h_A[i] = ((i * 7 + 3) % 200 - 100) * 0.005f;
    for (int i = 0; i < K * N; i++) h_B[i] = ((i * 11 + 5) % 200 - 100) * 0.005f;

    auto h_A_h = to_half(h_A);
    auto h_B_h = to_half(h_B);

    half *d_A, *d_B, *d_C_naive, *d_C_tiled, *d_C_wmma;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_naive, M * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_tiled, M * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C_wmma, M * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A_h.data(), M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B_h.data(), K * N * sizeof(half), cudaMemcpyHostToDevice));

    launch_gemm_naive(d_C_naive, d_A, d_B, M, N, K);
    launch_gemm_tiled(d_C_tiled, d_A, d_B, M, N, K);
    launch_gemm_wmma(d_C_wmma, d_A, d_B, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_naive(M * N), h_tiled(M * N), h_wmma(M * N);
    CUDA_CHECK(cudaMemcpy(h_naive.data(), d_C_naive, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_tiled.data(), d_C_tiled, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_wmma.data(), d_C_wmma, M * N * sizeof(half), cudaMemcpyDeviceToHost));

    auto r_naive = to_float(h_naive);
    auto r_tiled = to_float(h_tiled);
    auto r_wmma = to_float(h_wmma);

    // Cross-compare: all three should agree within FP16 tolerance
    for (int i = 0; i < M * N; i += 64) {
        float tol = fmaxf(1.0f, fabsf(r_naive[i]) * 0.1f);
        TEST_ASSERT(approx_equal(r_naive[i], r_tiled[i], tol),
                    "naive vs tiled mismatch at realistic size");
        TEST_ASSERT(approx_equal(r_naive[i], r_wmma[i], tol),
                    "naive vs wmma mismatch at realistic size");
    }

    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C_naive); cudaFree(d_C_tiled); cudaFree(d_C_wmma);
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

    // M2: Reduction
    run(test_rmsnorm,  "RMSNorm");
    run(test_softmax,  "Softmax");

    // M3: RoPE
    run(test_rope,     "RoPE");

    // M4: GEMM
    run(test_gemm_naive,     "GEMM Naive");
    run(test_gemm_tiled,     "GEMM Tiled");
    run(test_gemm_wmma,      "GEMM wmma");
    run(test_gemm_realistic, "GEMM Realistic");

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return (failed > 0) ? 1 : 0;
}
