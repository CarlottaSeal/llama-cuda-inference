#pragma once

#include <cuda_fp16.h>

// ============================================================
//  Kernel launch function declarations
//  Added incrementally as each kernel is implemented.
// ============================================================

// --- M1: Element-wise kernels ---
void launch_residual_add(half* out, const half* a, const half* b, int n);
void launch_silu_mul(half* out, const half* gate, const half* up, int n);
