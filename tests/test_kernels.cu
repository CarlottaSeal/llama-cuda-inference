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

                    // HF rotate_half pairing: (p, p + HEAD_DIM/2)
                    long long off = (long long)t * num_heads * HEAD_DIM
                                  + (long long)h * HEAD_DIM + p;
                    float x0 = x[off];
                    float x1 = x[off + HEAD_DIM / 2];
                    x[off]                = x0 * cos_a - x1 * sin_a;
                    x[off + HEAD_DIM / 2] = x0 * sin_a + x1 * cos_a;
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

    // Test 3: Llama 3 frequency scaling (factor 32, low 1, high 4, L 8192)
    printf("\n  Subtesting llama3 rope scaling (start_pos=1000)... ");
    {
        const int S_SEQ = 2, S_START = 1000;
        RopeScaling sc;
        sc.factor = 32.0f;
        sc.low_freq_factor = 1.0f;
        sc.high_freq_factor = 4.0f;
        sc.original_max_pos = 8192;

        const int SQ = S_SEQ * NUM_Q_HEADS * HEAD_DIM;
        const int SK = S_SEQ * NUM_KV_HEADS * HEAD_DIM;
        std::vector<float> h_sq(SQ), h_sk(SK);
        for (int i = 0; i < SQ; i++) h_sq[i] = ((i * 13) % 200 - 100) * 0.01f;
        for (int i = 0; i < SK; i++) h_sk[i] = ((i * 17) % 150 - 75) * 0.01f;

        auto h_sq_h = to_half(h_sq);
        auto h_sk_h = to_half(h_sk);
        half *d_sq, *d_sk;
        CUDA_CHECK(cudaMalloc(&d_sq, SQ * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_sk, SK * sizeof(half)));
        CUDA_CHECK(cudaMemcpy(d_sq, h_sq_h.data(), SQ * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sk, h_sk_h.data(), SK * sizeof(half), cudaMemcpyHostToDevice));

        launch_rope(d_sq, d_sk, S_SEQ, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, S_START, THETA, sc);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> h_sq_out(SQ);
        CUDA_CHECK(cudaMemcpy(h_sq_out.data(), d_sq, SQ * sizeof(half), cudaMemcpyDeviceToHost));
        auto result_sq = to_float(h_sq_out);

        // Reference: HF _compute_llama3_parameters, in double
        const double PI = 3.14159265358979323846;
        std::vector<double> inv_freq(HEAD_DIM / 2);
        int n_unchanged = 0, n_divided = 0, n_smoothed = 0;
        for (int p = 0; p < HEAD_DIM / 2; p++) {
            double f = 1.0 / pow((double)THETA, 2.0 * p / HEAD_DIM);
            double wavelen = 2.0 * PI / f;
            double low_wl = sc.original_max_pos / sc.low_freq_factor;
            double high_wl = sc.original_max_pos / sc.high_freq_factor;
            if (wavelen < high_wl) {
                n_unchanged++;
            } else if (wavelen > low_wl) {
                f /= sc.factor;
                n_divided++;
            } else {
                double s = (sc.original_max_pos / wavelen - sc.low_freq_factor)
                         / (sc.high_freq_factor - sc.low_freq_factor);
                f = (1.0 - s) * f / sc.factor + s * f;
                n_smoothed++;
            }
            inv_freq[p] = f;
        }
        // head_dim=64, theta=5e5 hits all three branches
        TEST_ASSERT(n_unchanged > 0 && n_divided > 0 && n_smoothed > 0,
                    "llama3 scaling test does not cover all frequency bands");

        float max_err = 0.0f;
        for (int t = 0; t < S_SEQ; t++) {
            for (int h = 0; h < NUM_Q_HEADS; h++) {
                long long base = (long long)t * NUM_Q_HEADS * HEAD_DIM + (long long)h * HEAD_DIM;
                for (int p = 0; p < HEAD_DIM / 2; p++) {
                    double a = (S_START + t) * inv_freq[p];
                    double x0 = __half2float(h_sq_h[base + p]);
                    double x1 = __half2float(h_sq_h[base + p + HEAD_DIM / 2]);
                    double e0 = x0 * cos(a) - x1 * sin(a);
                    double e1 = x0 * sin(a) + x1 * cos(a);
                    max_err = fmaxf(max_err, fabsf(result_sq[base + p] - (float)e0));
                    max_err = fmaxf(max_err, fabsf(result_sq[base + p + HEAD_DIM / 2] - (float)e1));
                }
            }
        }
        // FP16 storage of values in [-1.4, 1.4]: half-ulp ~ 5e-4
        TEST_ASSERT(max_err < 2e-3f, "llama3-scaled RoPE mismatch");

        cudaFree(d_sq);
        cudaFree(d_sk);
    }

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

bool test_gemm_wmma_edges() {
    printf("Testing GEMM wmma edge tiles... ");

    // Decode (M=1), q/kv projection widths, and a fully ragged shape.
    // C is allocated at exactly M*N so compute-sanitizer flags any overrun.
    const int shapes[][3] = {
        {1, 2048, 2048},
        {1, 512, 2048},
        {7, 2048, 2048},
        {37, 50, 72},
    };

    for (const auto& s : shapes) {
        int M = s[0], N = s[1], K = s[2];

        std::vector<float> h_A(M * K), h_B(K * N);
        for (int i = 0; i < M * K; i++) h_A[i] = ((i * 7 + 3) % 200 - 100) * 0.005f;
        for (int i = 0; i < K * N; i++) h_B[i] = ((i * 11 + 5) % 200 - 100) * 0.005f;

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

        bool ok = verify_gemm(to_float(h_C), ref_C, M, N, "wmma edge");
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        TEST_ASSERT(ok, "wmma GEMM edge-tile mismatch");
    }

    printf("PASSED\n");
    return true;
}

// ============================================================
//  M5 Tests: Attention
// ============================================================

// CPU reference attention with GQA and causal mask
static void cpu_attention(
    std::vector<float>& O,
    const std::vector<float>& Q,
    const std::vector<float>& K,
    const std::vector<float>& V,
    int seq_len, int kv_len,
    int num_q_heads, int num_kv_heads,
    int head_dim, float scale
) {
    int gqa_ratio = num_q_heads / num_kv_heads;

    for (int q_idx = 0; q_idx < seq_len; q_idx++) {
        int q_abs_pos = kv_len - seq_len + q_idx;
        for (int qh = 0; qh < num_q_heads; qh++) {
            int kv_head = qh / gqa_ratio;

            // Compute scores
            std::vector<float> scores(kv_len);
            float max_s = -FLT_MAX;
            for (int k_idx = 0; k_idx < kv_len; k_idx++) {
                if (k_idx <= q_abs_pos) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        float q_val = Q[(long long)q_idx * num_q_heads * head_dim + qh * head_dim + d];
                        float k_val = K[(long long)k_idx * num_kv_heads * head_dim + kv_head * head_dim + d];
                        dot += q_val * k_val;
                    }
                    scores[k_idx] = dot * scale;
                } else {
                    scores[k_idx] = -FLT_MAX;
                }
                max_s = fmaxf(max_s, scores[k_idx]);
            }

            // Softmax
            float sum_exp = 0.0f;
            for (int k_idx = 0; k_idx < kv_len; k_idx++) {
                scores[k_idx] = expf(scores[k_idx] - max_s);
                sum_exp += scores[k_idx];
            }
            for (int k_idx = 0; k_idx < kv_len; k_idx++) {
                scores[k_idx] /= sum_exp;
            }

            // Weighted sum of V
            for (int d = 0; d < head_dim; d++) {
                float acc = 0.0f;
                for (int k_idx = 0; k_idx < kv_len; k_idx++) {
                    float v_val = V[(long long)k_idx * num_kv_heads * head_dim + kv_head * head_dim + d];
                    acc += scores[k_idx] * v_val;
                }
                O[(long long)q_idx * num_q_heads * head_dim + qh * head_dim + d] = acc;
            }
        }
    }
}

bool test_attention_naive() {
    printf("Testing Naive Attention (GQA)... ");

    // Small test: seq=4, kv=4 (prefill, no cache), 8 Q heads, 2 KV heads, head_dim=64
    const int SEQ = 4, KV_LEN = 4;
    const int NQH = 8, NKVH = 2, HD = 64;
    const float SCALE = 1.0f / sqrtf((float)HD);

    const int Q_SIZE = SEQ * NQH * HD;
    const int KV_SIZE = KV_LEN * NKVH * HD;
    const int O_SIZE = SEQ * NQH * HD;

    std::vector<float> h_Q(Q_SIZE), h_K(KV_SIZE), h_V(KV_SIZE);
    for (int i = 0; i < Q_SIZE; i++) h_Q[i] = ((i % 100) - 50) * 0.01f;
    for (int i = 0; i < KV_SIZE; i++) h_K[i] = ((i % 80) - 40) * 0.01f;
    for (int i = 0; i < KV_SIZE; i++) h_V[i] = ((i % 120) - 60) * 0.01f;

    // CPU reference
    std::vector<float> ref_O(O_SIZE);
    cpu_attention(ref_O, h_Q, h_K, h_V, SEQ, KV_LEN, NQH, NKVH, HD, SCALE);

    auto h_Q_h = to_half(h_Q);
    auto h_K_h = to_half(h_K);
    auto h_V_h = to_half(h_V);

    half *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, Q_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, KV_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, KV_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O, O_SIZE * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q_h.data(), Q_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K_h.data(), KV_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V_h.data(), KV_SIZE * sizeof(half), cudaMemcpyHostToDevice));

    launch_attention_naive(d_O, d_Q, d_K, d_V, SEQ, KV_LEN, NQH, NKVH, HD, SCALE);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_O(O_SIZE);
    CUDA_CHECK(cudaMemcpy(h_O.data(), d_O, O_SIZE * sizeof(half), cudaMemcpyDeviceToHost));
    auto result = to_float(h_O);

    // Verify against CPU reference
    for (int i = 0; i < O_SIZE; i += 8) {
        float tol = fmaxf(0.1f, fabsf(ref_O[i]) * 0.1f);
        TEST_ASSERT(approx_equal(result[i], ref_O[i], tol), "Naive attention mismatch");
    }

    // Verify causal mask: first query should only attend to first KV position
    // (output should be exactly V[0] for the first Q's attention)
    // This is a weak check — the detailed comparison above is the real test

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
    printf("PASSED\n");
    return true;
}

bool test_flash_attention() {
    printf("Testing Flash Attention (GQA)... ");

    // Same params as naive test for cross-comparison
    const int SEQ = 4, KV_LEN = 4;
    const int NQH = 8, NKVH = 2, HD = 64;
    const float SCALE = 1.0f / sqrtf((float)HD);

    const int Q_SIZE = SEQ * NQH * HD;
    const int KV_SIZE = KV_LEN * NKVH * HD;
    const int O_SIZE = SEQ * NQH * HD;

    std::vector<float> h_Q(Q_SIZE), h_K(KV_SIZE), h_V(KV_SIZE);
    for (int i = 0; i < Q_SIZE; i++) h_Q[i] = ((i % 100) - 50) * 0.01f;
    for (int i = 0; i < KV_SIZE; i++) h_K[i] = ((i % 80) - 40) * 0.01f;
    for (int i = 0; i < KV_SIZE; i++) h_V[i] = ((i % 120) - 60) * 0.01f;

    // CPU reference
    std::vector<float> ref_O(O_SIZE);
    cpu_attention(ref_O, h_Q, h_K, h_V, SEQ, KV_LEN, NQH, NKVH, HD, SCALE);

    auto h_Q_h = to_half(h_Q);
    auto h_K_h = to_half(h_K);
    auto h_V_h = to_half(h_V);

    half *d_Q, *d_K, *d_V, *d_O_naive, *d_O_flash;
    CUDA_CHECK(cudaMalloc(&d_Q, Q_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, KV_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, KV_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O_naive, O_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_O_flash, O_SIZE * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q_h.data(), Q_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K_h.data(), KV_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V_h.data(), KV_SIZE * sizeof(half), cudaMemcpyHostToDevice));

    launch_attention_naive(d_O_naive, d_Q, d_K, d_V, SEQ, KV_LEN, NQH, NKVH, HD, SCALE);
    launch_flash_attention_gqa(d_O_flash, d_Q, d_K, d_V, SEQ, KV_LEN, NQH, NKVH, HD, SCALE);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_O_naive(O_SIZE), h_O_flash(O_SIZE);
    CUDA_CHECK(cudaMemcpy(h_O_naive.data(), d_O_naive, O_SIZE * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_O_flash.data(), d_O_flash, O_SIZE * sizeof(half), cudaMemcpyDeviceToHost));

    auto r_naive = to_float(h_O_naive);
    auto r_flash = to_float(h_O_flash);

    // Cross-compare: flash should match naive
    for (int i = 0; i < O_SIZE; i += 4) {
        float tol = fmaxf(0.15f, fabsf(r_naive[i]) * 0.15f);
        TEST_ASSERT(approx_equal(r_naive[i], r_flash[i], tol),
                    "Flash vs naive attention mismatch");
    }

    // Also verify flash vs CPU
    for (int i = 0; i < O_SIZE; i += 8) {
        float tol = fmaxf(0.15f, fabsf(ref_O[i]) * 0.15f);
        TEST_ASSERT(approx_equal(r_flash[i], ref_O[i], tol),
                    "Flash vs CPU attention mismatch");
    }

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O_naive); cudaFree(d_O_flash);

    // Test 2: decode scenario — seq=1, kv_len=64 (simulating cache)
    printf("\n  Subtesting decode scenario (seq=1, kv=64)... ");

    const int DSEQ = 1, DKV = 64;
    const int DQ_SIZE = DSEQ * NQH * HD;
    const int DKV_SIZE = DKV * NKVH * HD;
    const int DO_SIZE = DSEQ * NQH * HD;

    std::vector<float> h_dQ(DQ_SIZE), h_dK(DKV_SIZE), h_dV(DKV_SIZE);
    for (int i = 0; i < DQ_SIZE; i++) h_dQ[i] = ((i * 7) % 100 - 50) * 0.01f;
    for (int i = 0; i < DKV_SIZE; i++) h_dK[i] = ((i * 11) % 80 - 40) * 0.01f;
    for (int i = 0; i < DKV_SIZE; i++) h_dV[i] = ((i * 13) % 120 - 60) * 0.01f;

    std::vector<float> ref_dO(DO_SIZE);
    cpu_attention(ref_dO, h_dQ, h_dK, h_dV, DSEQ, DKV, NQH, NKVH, HD, SCALE);

    auto h_dQ_h = to_half(h_dQ);
    auto h_dK_h = to_half(h_dK);
    auto h_dV_h = to_half(h_dV);

    half *d_dQ, *d_dK, *d_dV, *d_dO;
    CUDA_CHECK(cudaMalloc(&d_dQ, DQ_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dK, DKV_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dV, DKV_SIZE * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_dO, DO_SIZE * sizeof(half)));

    CUDA_CHECK(cudaMemcpy(d_dQ, h_dQ_h.data(), DQ_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dK, h_dK_h.data(), DKV_SIZE * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dV, h_dV_h.data(), DKV_SIZE * sizeof(half), cudaMemcpyHostToDevice));

    launch_flash_attention_gqa(d_dO, d_dQ, d_dK, d_dV, DSEQ, DKV, NQH, NKVH, HD, SCALE);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> h_dO_out(DO_SIZE);
    CUDA_CHECK(cudaMemcpy(h_dO_out.data(), d_dO, DO_SIZE * sizeof(half), cudaMemcpyDeviceToHost));
    auto r_decode = to_float(h_dO_out);

    for (int i = 0; i < DO_SIZE; i += 4) {
        float tol = fmaxf(0.15f, fabsf(ref_dO[i]) * 0.15f);
        TEST_ASSERT(approx_equal(r_decode[i], ref_dO[i], tol),
                    "Flash attention decode mismatch");
    }

    cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV); cudaFree(d_dO);

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
    run(test_gemm_wmma_edges, "GEMM wmma edges");

    // M5: Attention
    run(test_attention_naive, "Attention Naive");
    run(test_flash_attention, "Flash Attention");

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return (failed > 0) ? 1 : 0;
}
