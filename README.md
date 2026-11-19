# llama-cuda-inference

A CUDA inference engine for Llama 3.2 1B with **hand-written GPU kernels**. No cuBLAS, no cuDNN, no frameworks — every kernel from scratch.

```
$ ./llama_inference ./models/Llama-3.2-1B "The capital of France is" 32
...
The capital of France is Paris. It is the most populous city in France and the country’s
capital and largest city. Paris is the most populous urban area in the European Union and the
```

**Status:** runs the official `meta-llama/Llama-3.2-1B` checkpoint end to end. Greedy output
matches HF `transformers` token for token on the prompts checked (see [Correctness](#correctness)).
Every kernel is also unit-tested against a CPU reference (14 tests), and the forward pass runs
clean under `compute-sanitizer`. Decode runs at 145–211 tok/s. It started at ~14.6: transposing
weights at load time brought it to ~51, and a dedicated M=1 GEMV did the rest. The large decode
projections now stream weights at 92–95% of DRAM bandwidth; see [Performance](#performance).

## Architecture

```
Token IDs ──► Embedding Lookup ──► hidden[seq, 2048]
                                       │
              ┌────────────────────────┘
              │        ×16 Transformer Layers
              ▼
     ┌─── RMSNorm ───► Q Proj (GEMM) ──► RoPE ──┐
     │                  K Proj (GEMM) ──► RoPE ──►├─► Attention ──► O Proj ──► + residual
     │                  V Proj (GEMM) ──► KV Cache┘
     │
     └─── RMSNorm ───► Gate Proj ──┐
                       Up Proj ───►├─► SiLU×Mul ──► Down Proj ──► + residual
                                   │
              ┌────────────────────┘
              ▼
         Final RMSNorm ──► LM Head (GEMM) ──► Greedy Argmax ──► Token
```

## Kernels

Every kernel is written from first principles. Three optimization levels for GEMM and two for Attention let you trace the path from naive to hardware-optimal.

| Kernel | File | Strategy | Key Technique |
|--------|------|----------|---------------|
| **Embedding** | `embedding.cu` | 1 thread per element | Coalesced lookup from [vocab, hidden] table |
| **Residual Add** | `residual.cu` | 1 thread per element | In-place a += b, vectorizable |
| **SiLU×Mul** | `silu.cu` | 1 thread per element | Fused σ(x)·x·y, FP32 intermediate |
| **RMSNorm** | `rmsnorm.cu` | 1 block per row | Warp shuffle `__shfl_xor_sync` → block reduce via shmem |
| **Softmax** | `softmax.cu` | 1 block per row | 3-pass: max → exp+sum → normalize; numerically stable |
| **RoPE** | `rope.cu` | 1 thread per dim pair | HF rotate_half pairing (i, i+d/2), Llama 3 frequency scaling |
| **GEMM Naive** | `gemm.cu` | 1 thread per output | O(MNK) global loads — the baseline |
| **GEMM Tiled** | `gemm.cu` | 32×32 shmem tiles | 32× fewer global memory loads, collaborative loading |
| **GEMM wmma** | `gemm.cu` | Tensor Core 16×16×16 | `nvcuda::wmma` fragments, 4×4 warp grid, FP16→FP32 accumulate |
| **Decode GEMV** | `gemm.cu` | 64 columns per block, split-K | 128-bit weight loads, block reduce, last-block split-K reduction (one launch, deterministic) |
| **Attention Naive** | `attention.cu` | Materialize S matrix | 3-pass (QK^T → causal mask + softmax → SV), shmem for scores |
| **Flash Attention** | `attention.cu` | Online softmax over KV tiles | KV tiles of 32, running (m, l, O) — O(head_dim) memory per block |

### GEMM Optimization Path

The three GEMM variants demonstrate a clear optimization progression:

**Naive** → Each thread computes one output element. `M×N` threads, each does `K` multiply-adds from global memory.
```
Thread (i,j): C[i,j] = Σ_k A[i,k] * B[k,j]    // K global loads each for A and B
```

**Tiled (Shared Memory)** → 32×32 thread blocks collaboratively load tiles of A and B into shared memory. Each element is loaded once from global memory and reused 32 times.
```
for tile in 0..K/32:
    sA[ty][tx] = A[row, tile*32+tx]     // one global load
    sB[ty][tx] = B[tile*32+ty, col]     // one global load
    __syncthreads()
    for k in 0..31:
        acc += sA[ty][k] * sB[k][tx]    // 32 shmem loads (fast!)
```

**wmma (Tensor Core)** → Hardware 16×16×16 matrix multiply-accumulate via `nvcuda::wmma`. Each warp owns a 16×16 output tile and issues one `mma_sync` per 16-wide K step. A 4×4 warp grid (16 warps = 512 threads per block) covers a 64×64 output tile.
```
wmma::load_matrix_sync(a_frag, A + ...);    // straight from global when the tile is full and aligned
wmma::load_matrix_sync(b_frag, sB + ...);   // otherwise via a zero-padded per-warp shmem tile
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```
Edge tiles (decode has M=1, and N/K need not be multiples of 16) take the staged path and a masked store, so the kernel is safe for any shape.

**GEMV (decode, M=1)** → With one token, every projection is `y[N] = x[K] × W[K,N]`, and each weight element is used exactly once. The job is to stream `W` at DRAM bandwidth, so no Tensor Cores and no reuse. `gemm()` sends every M=1 call (all decode projections, and the LM head in both prefill and decode) here, and everything else to wmma.
```
block = 256 threads, owns 64 columns
  lane → (row group = lane/8, column group = lane%8)
  each thread: uint4 load of W[k, n0 + 8·cg .. +8]   // 8 lanes = one 128 B line, a block covers 32 rows per step
  acc[8] += x[k] · w                                 // FP32
block reduce over the 32 row groups (8 KB shmem)
```
`k_proj`/`v_proj` have N=512, which is only 8 column blocks for 70 SMs. So K is also split across `gridDim.y` until the grid is about 4×SM blocks. Each block writes its 64 partial sums to a static workspace. The last block of a column range, found with an atomic counter, adds the partials in split order and writes FP16, then resets the counter. That keeps it to one launch with a fixed summation order. Two runs give bit-identical output, and the unit test checks that.

### Flash Attention

The Flash Attention kernel avoids materializing the full `[seq, kv_len]` score matrix, using **online softmax** with running statistics:

```
for each KV tile (size 32):
    s[j] = dot(Q, K[j]) * scale           // compute scores for tile
    apply causal mask                       // -inf for masked positions
    m_new = max(m, max(s))                 // running max
    α = exp(m - m_new)                     // rescale factor
    O = O * α + Σ_j exp(s[j] - m_new) * V[j]  // rescale + accumulate
    l = l * α + Σ_j exp(s[j] - m_new)     // running sum
    m = m_new
O = O / l                                  // final normalize
```

Memory usage: O(head_dim) per thread block vs O(kv_len) for naive — critical for long sequences.

One block handles one (query, head) pair, and K/V rows are read from global memory as each tile is processed rather than staged in shared memory, so this is the online-softmax half of FlashAttention without its SRAM tiling.

**GQA (Grouped Query Attention)**: 32 Q heads share 8 KV heads (4:1 ratio). Mapping: `kv_head = q_head / 4`.

## Project Structure

```
llama-cuda-inference/
├── CMakeLists.txt                      # Build config (SM arch configurable)
├── README.md
├── scripts/benchmark.sh                # Prefill / decode throughput sweep
│
├── include/
│   ├── kernels/kernels.cuh             # All kernel launch declarations
│   ├── model/
│   │   ├── llama_config.h              # Hardcoded Llama 3.2 1B config
│   │   ├── inference.h                 # Inference engine (buffers, KV cache)
│   │   ├── safetensors.h               # Safetensors file parser
│   │   ├── weight_loader.h             # HF weight loading + dtype conversion
│   │   └── tokenizer.h                 # BPE tokenizer
│   └── utils/cuda_utils.cuh            # CUDA_CHECK, warp/block reduce, cdiv
│
├── src/
│   ├── kernels/
│   │   ├── embedding.cu                # Token embedding lookup
│   │   ├── residual.cu                 # Residual add (a + b)
│   │   ├── silu.cu                     # SiLU×Mul (fused activation)
│   │   ├── rmsnorm.cu                  # RMSNorm with warp shuffle
│   │   ├── softmax.cu                  # Numerically stable softmax
│   │   ├── rope.cu                     # Rotary Position Embedding
│   │   ├── gemm.cu                     # 3 GEMM variants (naive/tiled/wmma)
│   │   └── attention.cu                # Naive + Flash Attention with GQA
│   ├── model/
│   │   ├── inference.cu                # Forward pass orchestration
│   │   ├── safetensors.cpp             # Safetensors parser (mmap / MapViewOfFile, hand-written JSON)
│   │   ├── weight_loader.cpp           # Weight loading (F16/BF16/F32 → FP16)
│   │   └── tokenizer.cpp              # BPE tokenizer (byte-level, greedy merge)
│   └── main.cu                         # CLI entry point
│
└── tests/
    └── test_kernels.cu                 # Per-kernel unit tests with CPU reference
```

**~4000 lines** of hand-written CUDA/C++. Zero external dependencies beyond CUDA toolkit.

## Build & Run

### Requirements
- CUDA Toolkit 11.0+ and a Tensor Core GPU (sm_70+) for the wmma path
- CMake 3.18+
- A C++17 host compiler (GCC/Clang, or MSVC with Visual Studio 2022)
- ~3 GB VRAM: 2.86 GB of FP16 weights (2.36 GB, plus a 0.5 GB transposed copy of the tied embedding for the LM head), plus KV cache and activations

Developed and tested on Windows (MSVC, CUDA 13.2, RTX 5070 Ti / sm_120). The POSIX `mmap` path is in place but hasn't been built on Linux yet.

### Build
```bash
# CUDA_ARCH defaults to 89; set it to your GPU
# (80=Ampere, 86=GA10x, 89=Ada, 90=Hopper, 120=Blackwell RTX 50)
cmake -B build -DCUDA_ARCH=89
cmake --build build --config Release -j
```
With the Visual Studio generator the binaries land in `build/Release/`.

### Download Model
```bash
# Requires huggingface-cli (pip install huggingface-hub)
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir ./models/Llama-3.2-1B
```
The repo is gated: accept the Llama 3.2 license on its Hugging Face page first. A fine-grained
token also needs "Read contents of gated repos you can access" enabled.

### Run Inference
```bash
./llama_inference ./models/Llama-3.2-1B "The future of AI is" 256
```

### Run Tests
```bash
./test_kernels
```

Tests cover every kernel with CPU reference implementations and cross-variant consistency checks (e.g., flash attention output matches naive attention, all three GEMMs agree at Llama projection sizes, wmma edge-tile shapes like M=1 and 37×50×72).

## Design Decisions

### FP16 Storage, FP32 Accumulation
All weights and activations stored as FP16 (`half`) to minimize memory bandwidth. All dot products and reductions accumulate in FP32 to prevent precision loss, then convert back to FP16 for storage.

### Weight Transpose at Load Time
HuggingFace stores linear layer weights as `[out_features, in_features]`. Our GEMM computes `C[M,N] = A[M,K] × B[K,N]`, so the loader transposes every linear weight once on the host (64×64 blocked) to `[in, out]` before uploading it. The forward pass then calls the GEMM on the weights directly.

The first version transposed each weight into a scratch buffer on every forward: 113 transpose kernels per forward, ~53 ms of a ~70 ms decode step (nsys). Moving that to load time:

- Llama 3.2 1B ties `lm_head` to `embed_tokens`. The embedding lookup needs `[vocab, hidden]` rows and the LM head GEMM needs `[hidden, vocab]`, so the LM head gets its own transposed copy: +0.5 GB of weights, about what the old scratch buffer took.
- Weight loading goes from ~3.1 s to ~7.8 s, because the transpose is single-threaded CPU code. A GPU transpose at upload time, or a `C = A × Bᵀ` GEMM that reads HF's layout directly, would remove that cost. Neither is done yet.

### KV Cache Layout
```
kv_cache_k[layer]: [max_seq_len, num_kv_heads × head_dim]
kv_cache_v[layer]: [max_seq_len, num_kv_heads × head_dim]
```
Separate K/V per layer. During prefill, positions `[0, seq_len)` are filled. During decode, position `[kv_len]` is appended. The attention kernel reads `[0, kv_len)` regardless of mode.

### Hand-Written JSON Parser
Both safetensors headers and tokenizer.json use a minimal hand-written JSON parser. This avoids the nlohmann/json dependency (300K+ lines) for parsing two specific, well-defined JSON structures.

### Safetensors with mmap
Weight files are memory-mapped (`mmap` on POSIX, `MapViewOfFile` on Windows), letting the OS manage page loading. Tensor data is accessed via offset into the mapped region: F16 tensors are copied straight from the mapping to the GPU, while BF16 tensors (what Llama 3.2 ships) are converted to FP16 on the CPU first.

## Profiling Guide

### NSight Systems (Timeline)
```bash
nsys profile --stats=true ./llama_inference ./models/Llama-3.2-1B "Hello" 32
nsys stats report.nsys-rep
```
Look for: kernel launch overhead, memory transfer gaps, kernel concurrency.

### NSight Compute (Kernel Deep Dive)
```bash
# Profile GEMM kernels (113 per forward: wmma in prefill, gemv_kernel in decode; cap the count)
ncu --set full --kernel-name gemm_wmma_kernel --launch-count 8 \
    ./llama_inference ./models/Llama-3.2-1B "Hello" 8
ncu --set full --kernel-name gemv_kernel --launch-skip 113 --launch-count 8 \
    ./llama_inference ./models/Llama-3.2-1B "Hello" 8

# Profile Flash Attention
ncu --kernel-name flash_attention_gqa_kernel --launch-count 8 \
    ./llama_inference ./models/Llama-3.2-1B "Hello" 8
```
Key metrics: achieved occupancy, memory throughput, compute throughput, warp stall reasons.

## Correctness

Checked against HF `transformers` (`generate`, greedy, FP16 and FP32 give the same tokens) on
the official checkpoint:

| Prompt | Generated | Result |
|--------|-----------|--------|
| `The capital of France is` | 32 tokens | prompt ids and all 32 generated tokens identical |
| `In 2024, the GPU had 16,384 cores. def fib(n): return` | 64 tokens | generated text identical; prompt tokenized differently (see Limitations) |

## Performance

RTX 5070 Ti (spec 896 GB/s), CUDA 13.2, Windows, Release build. Prompt `The capital of France is`
(6 tokens), 3 runs each. Each pair of builds was run interleaved in the same session. Builds are
commits `3654f25` (runtime transpose), `cdc00ac` (load-time transpose), and this one (+ GEMV):

| Build | Generated tokens | Prefill (6 tok) | Decode | Weight load |
|-------|------------------|-----------------|--------|-------------|
| runtime transpose | 32  | 80.8–81.8 ms | 14.6–14.8 tok/s | 3.09–3.14 s |
| runtime transpose | 128 | 81.0–82.1 ms | 14.5–14.7 tok/s | 3.11–3.15 s |
| load-time transpose | 32  | 24.0–24.6 ms | 51.9–52.2 tok/s | 7.73–7.89 s |
| load-time transpose | 128 | 24.1–24.4 ms | 51.3–51.8 tok/s | 7.74–7.80 s |
| *second session:* | | | | |
| load-time transpose | 32  | 27.1–27.5 ms | 49.7–50.7 tok/s | |
| load-time transpose | 128 | 27.0–27.9 ms | 50.5–50.7 tok/s | |
| + decode GEMV | 32  | 27.2–27.4 ms | **144.7–181.6 tok/s** | |
| + decode GEMV | 128 | 27.1–27.6 ms | **191.8–211.0 tok/s** | |

Generated text is identical across all three builds on both [Correctness](#correctness) prompts.
Prefill is unchanged by the GEMV because its 6-row projections still go through wmma, and the LM
head is only one of 113 GEMMs. The GEMV build's decode spread is much wider than the others'. At
~5 ms/token, the host side (about 150 launches per token plus the argmax copy back) and GPU clock
ramp-up show up in the wall time. Short runs are hit harder. This has not been broken down further.

GPU kernel time per forward (nsys `cuda_gpu_kern_sum`, total divided by the number of forwards in each capture):

| Kernel | runtime transpose | load-time transpose |
|--------|-------------------|---------------------|
| `transpose_kernel` | ~53 ms | — |
| `gemm_wmma_kernel` (113 launches) | ~17 ms | ~18 ms (**97.9%** of GPU time) |
| everything else | < 0.5 ms | < 0.4 ms |

With load-time transposes, decode took ~19 ms/token. A 1B FP16 model is ~2.5 GB of weights, so a
bandwidth-bound step should take ~3 ms. At M=1 the wmma GEMM took at least ~103 µs per launch,
which for `q_proj` (8 MB) is ~80 GB/s: it pads the single row to a 16-row tile.

With the GEMV, per-shape decode times look like this (nsys, 128-token run, median over all launches of that shape):

| Projection | Weight | Grid (cols × split) | Median | Effective bandwidth |
|------------|--------|---------------------|--------|---------------------|
| q / o | 8.4 MB | 32 × 8 | 11.6 µs | 723 GB/s (81%) |
| k / v | 2.1 MB | 8 × 32 | 4.7 µs | 446 GB/s (50%) |
| gate / up | 33.6 MB | 128 × 3 | 40.8 µs | 822 GB/s (92%) |
| down | 33.6 MB | 32 × 9 | 40.9 µs | 820 GB/s (92%) |
| LM head | 525 MB | 2004 × 1 | 617 µs | 851 GB/s (95%) |

Summed over one forward, that is 3.10 ms for 2.47 GB of weights (797 GB/s, 89%). The means are
higher than the medians (e.g. 47.9 µs for gate/up) because the first steps run before clocks ramp up.
In the GEMV build the GPU kernel mix is GEMV 82.7%, attention 6.6%, the prefill wmma GEMMs 4.3%,
RMSNorm 3.4%, and argmax 1.6% (one ~70 µs launch per token).

What's left: k/v are too small to reach bandwidth on their own. Fusing Q/K/V, and gate/up, into
one GEMV each (items 1 and 2 below) is the direct fix. After that come launch overhead and the
host round trip per token.

To sweep prefill and decode throughput:
```bash
./scripts/benchmark.sh ./models/Llama-3.2-1B [build_dir]
```

## Known Optimization Opportunities

Done:
- **Weight pre-transpose** (see [Weight Transpose at Load Time](#weight-transpose-at-load-time)): 3.5× decode.
- **Decode GEMV** (see [GEMM Optimization Path](#gemm-optimization-path)): another 2.9–4.2× decode. The large projections run at 92–95% of DRAM bandwidth.

Not implemented yet:

1. **Fused QKV projection** — Concatenate W_q, W_k, W_v into one `[hidden, q_dim+2×kv_dim]` matrix, so decode does one 12.6 MB GEMV instead of 8.4 + 2.1 + 2.1 MB. k/v alone reach only ~446 GB/s, and it also removes two launches per layer.

2. **Fused gate/up projection** — Same idea for the MLP: one `[hidden, 2×intermediate]` GEMV, with SiLU×Mul folded into its epilogue.

3. **Launch overhead / host sync in decode** — ~150 launches per token plus a device→host argmax copy. GPU kernel time is ~3.7 ms/token by median (~4.35 ms by mean) against ~4.7 ms of wall time in a 128-token run, so CUDA Graphs (or a persistent kernel) is the next step after fusion.

4. **GEMM (prefill): shared tile reuse + double buffering** — Each warp in the wmma kernel loads its own A and B fragments, so the 16 warps in a block don't share tiles, and loads and MMAs run back to back. Block-level shared-memory tiles with double buffering would cut global traffic and hide load latency.

5. **GEMM: Swizzled shared memory** — Not yet profiled for bank conflicts. If ncu shows them on the staged path, XOR-swizzled layouts (as in CUTLASS) remove them.

6. **Flash Attention v2 optimizations** — The current implementation uses one block per (query, head) and no SRAM tiling. FA2 adds non-square tiles, warp specialization (producer/consumer), and pipelining within the K/V tile loop. It is 6.6% of decode GPU time now.

7. **KV Cache quantization** — Storing the KV cache in INT8/FP8 would halve its memory and allow longer sequences. Requires dequantize-on-the-fly in the attention kernel.

8. **PagedAttention** — The current contiguous KV cache wastes memory on short sequences. Paged allocation (vLLM-style) enables efficient batching of variable-length sequences.

9. **Speculative decoding** — A smaller draft model proposes tokens, and the full model verifies them in parallel. Reported to give 2–3× effective decode speedup.

## Limitations

- Batch size 1, greedy decoding only (`sample_top_p` is a stub that falls back to greedy).
- Model shape is hardcoded in `llama_config.h` for Llama 3.2 1B, with `max_seq_len` = 2048.
- The tokenizer doesn't implement Llama 3's regex pre-tokenization; BPE merges run over the whole byte string, so some inputs tokenize differently from HF. Digits are one case: `2024` becomes `20`+`24` here and `202`+`4` in HF, which splits numbers into groups of at most three digits first.

## License

MIT

## Acknowledgments

- [Meta AI](https://ai.meta.com/) for the Llama model family
- [Tri Dao](https://tridao.me/) for the Flash Attention algorithm
- NVIDIA for CUDA, wmma intrinsics, and NSight profiling tools
