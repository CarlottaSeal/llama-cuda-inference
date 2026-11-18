#include "kernels/kernels.cuh"
#include "utils/cuda_utils.cuh"

// ============================================================
//  Embedding Lookup
//
//  For each token in the sequence, copy its row from the
//  embedding table into the output buffer.
//
//  out[tok * hidden_size + j] = table[token_ids[tok] * hidden_size + j]
//
//  Grid: one block per token.
//  Threads: stride over hidden_size dimension.
//  Memory-bound (just a gather).
// ============================================================

__global__ void embedding_kernel(
    half* out,
    const half* table,
    const int* token_ids,
    int seq_len,
    int hidden_size
) {
    int tok = blockIdx.x;  // which token in the sequence
    if (tok >= seq_len) return;

    int id = token_ids[tok];
    const half* src = table + (long long)id * hidden_size;
    half* dst = out + (long long)tok * hidden_size;

    // Each thread copies one or more elements
    for (int j = threadIdx.x; j < hidden_size; j += blockDim.x) {
        dst[j] = src[j];
    }
}

void launch_embedding(half* out, const half* table, const int* token_ids,
                       int seq_len, int hidden_size) {
    constexpr int BLOCK = 256;
    embedding_kernel<<<seq_len, BLOCK>>>(out, table, token_ids, seq_len, hidden_size);
}
