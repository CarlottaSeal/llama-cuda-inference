#pragma once

#include <cuda_fp16.h>

// ============================================================
//  Kernel launch function declarations
//  Added incrementally as each kernel is implemented.
// ============================================================

// --- M1: Element-wise kernels ---
void launch_residual_add(half* out, const half* a, const half* b, int n);
void launch_silu_mul(half* out, const half* gate, const half* up, int n);
void launch_embedding(half* out, const half* table, const int* token_ids, int seq_len, int hidden_size);

// --- M2: Reduction kernels ---
void launch_rmsnorm(half* out, const half* x, const half* weight, int rows, int cols, float eps);
void launch_softmax(float* x, int rows, int cols);
