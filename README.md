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
Every kernel is also unit-tested against a CPU reference (17 tests), and the forward pass runs
clean under `compute-sanitizer`. Decode runs at ~273 tok/s at 128 tokens and ~254 tok/s for 200
tokens after an 1802-token prompt. A split-KV decode attention kernel keeps long contexts from slowing it down. It started at ~14.6:
transposing weights at load time brought it to ~51, a dedicated M=1 GEMV to ~200, fusing
Q/K/V into one projection added ~6%, fusing gate/up another ~2%, and replaying each decode step
from a CUDA Graph ~10% (~273 tok/s at 128 tokens). The large decode projections stream weights at 92–95% of
DRAM bandwidth. Prefill attention runs on Tensor Cores (a port of the author's FlashAttention-lite
v5), and with gate/up fused a 2K-token prefill takes ~236 ms, down from ~715 ms; see [Performance](#performance).

## Architecture

```
Token IDs ──► Embedding Lookup ──► hidden[seq, 2048]
                                       │
              ┌────────────────────────┘
              │        ×16 Transformer Layers
              ▼
     ┌─── RMSNorm ───► QKV Proj (one GEMM) ──► split ─┬─ Q ──► RoPE ──┐
     │                                                ├─ K ──► KV Cache ──► RoPE (in place) ──►├─► Attention ──► O Proj ──► + residual
     │                                                └─ V ──► KV Cache ─────────────────────┘
     │
     └─── RMSNorm ───► Gate|Up Proj (one GEMM/GEMV) ──► SiLU×Mul ──► Down Proj ──► + residual
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
| **SiLU×Mul** | `silu.cu` | 1 thread per element | Fused σ(x)·x·y, FP32 intermediate; prefill reads the interleaved gate/up GEMM output (decode folds it into the GEMV) |
| **RMSNorm** | `rmsnorm.cu` | 1 block per row | Warp shuffle `__shfl_xor_sync` → block reduce via shmem |
| **Softmax** | `softmax.cu` | 1 block per row | 3-pass: max → exp+sum → normalize; numerically stable |
| **RoPE** | `rope.cu` | 1 thread per dim pair | HF rotate_half pairing (i, i+d/2), Llama 3 frequency scaling |
| **GEMM Naive** | `gemm.cu` | 1 thread per output | O(MNK) global loads — the baseline |
| **GEMM Tiled** | `gemm.cu` | 32×32 shmem tiles | 32× fewer global memory loads, collaborative loading |
| **GEMM wmma** | `gemm.cu` | Tensor Core 16×16×16 | `nvcuda::wmma` fragments, 4×4 warp grid, FP16→FP32 accumulate |
| **Decode GEMV** | `gemm.cu` | 64 columns per block, split-K | 128-bit weight loads, block reduce, last-block split-K reduction (one launch, deterministic) |
| **Attention Naive** | `attention.cu` | Materialize S matrix | 3-pass (QK^T → causal mask + softmax → SV), shmem for scores |
| **Flash Attention** | `attention.cu` | Online softmax over KV tiles | KV tiles of 32, running (m, l, O) — O(head_dim) memory per block (prefill) |
| **Split-KV decode attention** | `attention.cu` | (KV head, KV chunk) per warp | All 4 GQA Q heads per K/V read, 128-bit K rows, last-block merge of (m, l, O) (decode) |
| **Tensor Core prefill attention** | `attention.cu` | 64 query rows per block, 16 per warp | `mma.sync` m16n8k16, S/P/O in registers, cp.async double-buffered K/V, XOR-swizzled shmem + `ldmatrix`, causal tile skip (prefill) |

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

One block handles one (query, head) pair, and K/V rows are read from global memory as each tile is processed rather than staged in shared memory, so this is the online-softmax half of FlashAttention without its SRAM tiling. It is now only the fallback for shapes the two kernels below don't support (head_dim ≠ 64, or a GQA ratio other than 4 at decode).

### Split-KV Decode Attention

At decode there is one query per head. The kernel above then has only 32 blocks, which walk the whole KV range serially. That cost grows linearly with context: 465 µs per layer at kv_len 2032 ([Context length](#context-length)). `flash_decode_kernel` is flash-decoding adapted to GQA:

```
grid = (kv_head, split), one warp per block; chunk = max(32, kv_len/64 rounded up to 32)
q·scale for the 4 Q heads of this KV head → shmem
for each 32-position tile in this block's chunk:
    lane l: K row (base+l), 8 × uint4 → s[h] = q_h·k for h = 0..3       // each K row read once for 4 heads
    per head: m, l, O rescaled with warp max/sum (online softmax)
    lane l owns dims 2l, 2l+1: for each position j: O_h += p_h[j] · V[j]  // p broadcast by shuffle
store (m, l, O) per Q head; the last block of this KV head (atomic counter)
    merges the splits in order: O = Σ e^(m_s−M) O_s / Σ e^(m_s−M) l_s
```

At kv_len 2048 that is 64 splits × 8 KV heads = 512 single-warp blocks, against 32 before. The merge is in-kernel and in fixed split order, the same scheme as the GEMV's split-K: one launch, bit-identical across runs (unit-tested with kv_len from 1 to 3000).

### Tensor Core Prefill Attention

`flash_prefill_kernel` is the author's FlashAttention-lite v5 (FA-2 structure on raw `mma.sync`), adapted to the inference engine:

```
grid = (ceil(seq_len/64), q_head), 128 threads = 4 warps, each warp owns 16 query rows
Q tile → shmem (cp.async) → registers via ldmatrix, kept for the whole loop
for each 64-position K/V tile, up to the block's last visible position:   // causal tile skip
    prefetch the next K/V tile into the other shmem stage (cp.async)
    S = Q Kᵀ        mma.sync m16n8k16, FP32 accumulators in registers
    mask col > pos or col ≥ kv_len → −inf; online softmax on registers (quad shuffles)
    O = O·α + P V   the S accumulators are repacked to FP16 as the A fragment, V via ldmatrix.trans
O / l → attn_out
```

What changed from FA-lite: a causal mask with per-row positions `start + i` and tile skip (a block stops at the last position its rows can see, so the causal triangle costs about half the full square); GQA (`kv_head = q_head / ratio`, read in place, no K/V duplication); a query offset `start = kv_len − seq_len` so chunked prefill into a non-empty cache works; and strided addressing, since Q is `[seq, q_heads·64]` and the cache is `[pos, kv_heads·64]` rather than contiguous `[B, H, N, D]`. Out-of-range rows load as zeros (cp.async src-size 0) and are not stored. Shared memory is 32 KB (two K/V stages, Q borrowing stage 1 before the loop). The unit test checks seq_len 6, 64, 65, 100 and 513 from an empty cache, plus chunked prefill (130 rows at start_pos 37, 200 at start_pos 300), against the CPU reference (max abs error 3.6e-4).

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
commits `3654f25` (runtime transpose), `cdc00ac` (load-time transpose), `530f370` (+ GEMV), and
this one (+ fused QKV). Absolute numbers drift between sessions (the GEMV build measured 192–211
tok/s in one session and 223–225 in the next), so compare builds only within a session:

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
| *third session:* | | | | |
| + decode GEMV | 32  | 23.5–27.2 ms | 165.0–236.2 tok/s | 7.76–7.85 s |
| + decode GEMV | 128 | 23.7–24.1 ms | 223.1–225.4 tok/s | 7.73–7.79 s |
| + fused QKV | 32  | 19.4–22.4 ms | 200.8–248.0 tok/s | 7.79–7.88 s |
| + fused QKV | 128 | 19.4–19.6 ms | **236.2–237.4 tok/s** | 7.78–7.91 s |

Generated text is identical across all four builds on both [Correctness](#correctness) prompts.
In the third session, the first 32-token run of each build is the slow end of its range
(165.0 and 200.8 tok/s). Runs 2–3 are 234–236 vs 246–248.

Fused QKV saves ~0.26 ms per decode step (4.48 → 4.22 ms at 128 tokens, +5.6%) and ~4 ms of
prefill (3 wmma GEMMs per layer become 1). Its decode GEMV kernel time accounts for only ~0.07 ms
of that (below). The rest presumably comes from dropping 2 GEMM launches and 2 `cudaMemcpy` calls
per layer (245 kernels + 32 copies → 213 kernels per token). That split hasn't been measured separately.
Prefill is unchanged by the GEMV because its 6-row projections still go through wmma, and the LM
head is only one of 113 GEMMs. The GEMV build's decode spread is much wider than the others'. At
~5 ms/token, the host side (245 kernel launches and 32 device-to-device copies per token, plus the argmax copy back) and GPU clock
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
| fused q\|k\|v | 12.6 MB | 48 × 6 | 16.6 µs | 758 GB/s (85%) |
| gate / up | 33.6 MB | 128 × 3 | 40.8 µs | 822 GB/s (92%) |
| down | 33.6 MB | 32 × 9 | 40.9 µs | 820 GB/s (92%) |
| LM head | 525 MB | 2004 × 1 | 617 µs | 851 GB/s (95%) |

The fused row is from the fused-QKV build's capture; the others from the GEMV build's. Fusion
turns 11.6 + 4.7 + 4.7 = 21.0 µs into 16.6 µs per layer.

Summed over one forward (GEMV build), that is 3.10 ms for 2.47 GB of weights (797 GB/s, 89%);
with fused QKV, 3.03 ms (815 GB/s, 91%). The means are
higher than the medians (e.g. 47.9 µs for gate/up) because the first steps run before clocks ramp up.
In the GEMV build the GPU kernel mix is GEMV 82.7%, attention 6.6%, the prefill wmma GEMMs 4.3%,
RMSNorm 3.4%, and argmax 1.6% (one ~70 µs launch per token).

What's left: the weights are now read at ~91% of bandwidth overall, so further kernel-side gains
are small. The larger gap is between GPU kernel time and wall time: launch overhead and the host
round trip per token.

### Fused gate/up projection

`W_gate` and `W_up` are interleaved at load time into one `[2048, 16384]` matrix in groups of 32
columns (`g[0:32] u[0:32] g[32:64] u[32:64] …`), so each 64-column GEMV block holds 32 gate
outputs and their 32 up outputs. Decode runs one GEMV whose epilogue writes `silu(g)·u` directly
(after the split-K merge when K is split), and `silu_mul` and one GEMV launch disappear: 32 fewer
kernels per token (nsys, 128-token run: 229 → 197 kernels per forward on average). Prefill runs one wmma GEMM into a `[seq, 16384]` buffer and an
`silu_mul` variant that reads the interleaved layout. g and u are rounded to FP16 before the
SwiGLU, as in the unfused path. The unit test checks that prefill output matches the unfused
path bit for bit. Weight memory is unchanged (2858 MB). Load time goes from 7.89–7.97 s to
8.26–8.36 s (2 runs each) because of the extra host-side interleave copy.

Kernel medians (nsys, per layer):

| | Before (`a7d9040`) | After |
|---|---|---|
| decode gate + up GEMV | 41.4 + 41.4 µs (grid 128×3 each) | 82.2 µs (grid 256×2) |
| decode `silu_mul` | separate launch | in the GEMV epilogue |
| prefill gate + up wmma, 2002 tokens | 4266 + 4266 µs | **6451 µs** |
| prefill SwiGLU, 2002 tokens | 220 µs | 125 µs |

The decode GEMV takes the same time (it was already at ~92% of bandwidth), so the decode gain is
the two launches. The prefill GEMM gain was not expected and hasn't been analyzed. One N=16384
launch takes 24% less time than two N=8192 launches over the same A.

End to end, interleaved with `a7d9040`, 3 s idle before every run:

| | Before | After |
|---|---|---|
| decode, 32 tokens (5 runs) | 244.5–245.2 tok/s | 244.9–250.6 tok/s |
| decode, 128 tokens (5 runs) | 240.7–243.4 tok/s | **247.0–248.5 tok/s** (+2.4%) |
| prefill 132 tokens (6 runs) | 24.9–25.9 ms | 25.2–26.2 ms |
| prefill 262 tokens (6 runs) | 42.4–43.3 ms | 41.0–41.9 ms |
| prefill 512 tokens (3 runs) | 64.6–66.0 ms | 57.4–57.7 ms |
| prefill 1022 tokens (3 runs) | 150.9–153.1 ms | 137.7–138.7 ms |
| prefill 2002 tokens (3 runs) | 269.2–271.3 ms | **235.6–236.9 ms** (−12.6%) |

At 128 tokens that is 4.13 → 4.03 ms per step, ~6 µs per layer, more than the ~0.6 µs of GEMV
kernel time saved. Prompts of 132 tokens or fewer are unchanged within noise. Generated text is
identical on both [Correctness](#correctness) prompts and the 2002-token prompt. 17/17 unit tests
pass, and `compute-sanitizer` memcheck and synccheck report 0 errors on a 132-token prefill plus 3
decode steps.

### CUDA Graph decode

A decode step is 197 kernels. Launched one by one from the host, with a blocking argmax copy back
before the next step can start, the GPU sat idle for part of every step. The decode step is now
captured once into a CUDA Graph and replayed for every token:

- **Positions on the device.** `d_pos` holds the position of the token the step consumes. The
  embedding, the QKV split (KV cache write row), RoPE and decode attention read it when they run,
  so one captured graph serves every position. Decode attention uses a fixed grid sized for
  `max_seq`; splits past `*d_pos + 1` publish an empty partial (m = −∞, l = 0). With `max_seq` ≤ 2048
  the chunking matches the host-kv_len launch, so the output is bit-identical (unit-tested).
- **Argmax feeds the next step.** The argmax kernel writes the token into `token_ids[pos + 1]` and
  increments `d_pos`, so no host round trip sits between steps.
- **Two-deep host pipeline.** The host enqueues step k, then waits (event + pinned 4-byte copy) for
  step k−1's token, prints it, and checks for EOS. The GPU always has the next step queued. After an
  EOS one extra step may run; its output is discarded.
- The build uses `--default-stream per-thread` so the default stream can be captured
  (`cudaStreamCaptureModeThreadLocal`). Capture takes 5–11 ms at startup. `LLAMA_NO_GRAPH=1`
  falls back to per-kernel launches, for A/B runs and profiling.

nsys, 128-token run, median over 127 steps (`--cuda-graph-trace=node` for the graph run):

| | Eager | Graph |
|---|---|---|
| kernels / step | 197 | 197 |
| kernel time / step | 3410.5 µs | 3410.5 µs |
| step span (argmax to argmax) | 3960.5 µs | **3578.4 µs** |
| idle / step | 550 µs (13.9%) | 168 µs (4.7%) |

End to end, interleaved with `7b464af`, 3 s idle before every run. Of the 5 32-token runs the first
3 had 68–87% GPU utilization from another process before the run and are left out:

| | Before | After |
|---|---|---|
| decode, 32 tokens (runs 4–5) | 246.9–247.9 tok/s | 274.7 tok/s |
| decode, 128 tokens (5 runs) | 248.6–249.4 tok/s | **273.2–274.5 tok/s** (+10%) |
| decode, 200 tokens after a 1802-token prompt (3 runs) | 231.0–232.1 tok/s | **253.4–253.9 tok/s** (+9%) |

Generated text is identical to `7b464af` on both [Correctness](#correctness) prompts and the
2002-token prompt, and identical between graph and `LLAMA_NO_GRAPH=1` in the same build on 200
tokens (6-token prompt) and on the 2002-token prompt, which runs into `max_seq` = 2048. 17/17 unit
tests pass, and `compute-sanitizer` memcheck reports 0 errors.

### Context length

The tables above use a 6-token prompt, so attention barely shows up. With the generic flash
attention kernel it took over at longer contexts, in both phases (nsys). Both phases now have
their own kernel: split-KV for decode, Tensor Core tiles for prefill.

**Decode**: one run from a 6-token prompt to kv_len 2048. Each decode step is matched to its kv_len,
and each row is the median over steps within ±16. "Before" is the fused-QKV build (`b083602`), which
used the generic flash attention kernel for decode too; "after" adds the split-KV decode kernel:

| kv_len | Attention / token (before → after) | GPU time / token (before → after) | Attention share | Attention / layer |
|--------|------------------------------------|-----------------------------------|-----------------|-------------------|
| 64   | 0.28 → 0.12 ms | 3.61 → 3.45 ms  | 7.9% → 3.4%  | 17.8 → 7.3 µs |
| 128  | 0.50 → 0.12 ms | 3.84 → 3.44 ms  | 13.1% → 3.5% | 31.4 → 7.5 µs |
| 256  | 0.97 → 0.14 ms | 4.29 → 3.47 ms  | 22.6% → 4.1% | 60.8 → 8.9 µs |
| 512  | 1.88 → 0.18 ms | 5.21 → 3.51 ms  | 36.1% → 5.2% | 118 → 11.4 µs |
| 1024 | 3.82 → 0.25 ms | 7.27 → 3.60 ms  | 52.5% → 7.0% | 239 → 15.8 µs |
| 1536 | 5.50 → 0.33 ms | 8.82 → 3.90 ms  | 62.3% → 8.4% | 344 → 20.4 µs |
| 2032 | 7.44 → 0.39 ms | 10.98 → 3.73 ms | 67.8% → 10.5% | 465 → 24.5 µs |

The old kernel launched one block per (query, head): 32 blocks of 64 threads, each walking the
whole KV range serially, and each KV head read by 4 separate Q-head blocks. At kv_len 2032 a
layer's K and V are 4.2 MB, read in 465 µs: ~9 GB/s, latency-bound. The split-KV kernel
([design](#split-kv-decode-attention)) reads the same 4.2 MB in 24.5 µs, ~171 GB/s. That is still
well under DRAM bandwidth, but attention is now ~10% of a step even at 2K context. The ~7 µs floor
at short contexts is one launch plus the in-kernel merge.

End to end (3 runs each, interleaved with `b083602`, both builds checked idle before and after every run):

| Generated tokens | Before | After |
|------------------|--------|-------|
| 32   | 130.0–181.8 tok/s | 132.3–196.8 tok/s |
| 128  | 191.0–211.1 tok/s | 197.6–209.1 tok/s |
| 2042 | 124.7–125.1 tok/s | **224.1–226.9 tok/s** (1.80×) |

At 32 and 128 tokens the difference (~0.16 ms/step) is inside run-to-run noise. Generated text is
identical between the builds, including the 2042-token run (7,876 characters).

**Prefill**: prompts of repeated text, generating 1 token. Wall time is from 3 plain runs; the
kernel split is from one nsys run of the prefill forward. Measured on the split-KV build (`eb370e7`),
which still used the generic kernel for prefill:

| Prompt tokens | Prefill wall | Attention | wmma GEMM | GEMV (LM head) | Attention share |
|---------------|--------------|-----------|-----------|----------------|-----------------|
| 131  | 28.1–29.2 ms   | 2.2 ms   | 23.4 ms  | 0.6 ms | 8.4%  |
| 511  | 114.4–115.6 ms | 31.0 ms  | 79.2 ms  | 0.6 ms | 27.7% |
| 1021 | 272.7–278.8 ms | 119.0 ms | 147.3 ms | 0.6 ms | 44.2% |
| 2001 | 717.4–723.2 ms | 454.0 ms | 265.0 ms | 0.6 ms | 62.7% |

Attention grows quadratically: at 2001 tokens it is ~0.26 TFLOP of causal attention in 454 ms,
~0.6 TFLOPS. The wmma GEMMs do ~3.9 TFLOP in 265 ms, ~15 TFLOPS. Both are far from the GPU's
Tensor Core rate. For comparison, the author's FlashAttention-lite v5 (mma.sync, cp.async, swizzled
shmem) runs a non-causal N=2048, D=64, 16-head forward in 0.273 ms, ~63 TFLOPS on this GPU.

With the [Tensor Core prefill kernel](#tensor-core-prefill-attention) (3 runs each, interleaved with
`eb370e7`, a 3 s idle gap before every run, GPU at ≤ 8% utilization before each). Prompt
`"The quick brown fox jumps over the lazy dog. "` repeated 13/51/102/200 times:

| Prompt tokens | Prefill wall (before → after) |
|---------------|-------------------------------|
| 132  | 27.8–98.1 → 25.4–26.4 ms |
| 512  | 94.4–97.0 → 65.3–67.4 ms |
| 1022 | 276.5–286.4 → 154.1–162.6 ms |
| 2002 | 712.4–719.4 → **272.4–281.7 ms** (2.6×) |

At 132 tokens the base build's first two runs were 98.1 and 85.0 ms and its third 27.8 ms, so that
row is mostly noise. Kernel time at 2002 tokens (nsys, one prefill forward):

| Kernel | Before | After |
|--------|--------|-------|
| attention (16 launches) | 448.4 ms (62.6%) | **3.94 ms** (1.4%), median 231 µs/layer |
| `gemm_wmma_kernel` (80 launches) | 259.3 ms | 259.6 ms (95.4%) |

Causal attention at 2002 tokens is ~0.26 TFLOP, so the new kernel runs at ~67 TFLOPS, the same
range as FA-lite's own non-causal benchmark. Prefill is now almost entirely the wmma GEMM (item 2 under [Known Optimization Opportunities](#known-optimization-opportunities)).

Generated text is identical to `eb370e7` on both [Correctness](#correctness) prompts and on the
2002-token prompt (64 generated tokens). Decode doesn't touch the new kernel (seq_len 1 goes to split-KV).
At 128 tokens, 3 interleaved runs gave 197.2–210.9 tok/s before and 203.0–213.0 after. At 32 tokens
a first set of 3 runs gave 170.0–233.2 before and 136.6–162.2 after. A repeat with 5 runs gave
143.2–241.9 before and 227.8–245.9 after, so 32-token decode here is noise of about ±50 tok/s. `compute-sanitizer`
memcheck on a 132-token prefill (ragged last tile) plus 2 decode steps reports 0 errors.

To sweep prefill and decode throughput:
```bash
./scripts/benchmark.sh ./models/Llama-3.2-1B [build_dir]
```

## Known Optimization Opportunities

Done:
- **Weight pre-transpose** (see [Weight Transpose at Load Time](#weight-transpose-at-load-time)): 3.5× decode.
- **Decode GEMV** (see [GEMM Optimization Path](#gemm-optimization-path)): another 2.9–4.2× decode. The large projections run at 92–95% of DRAM bandwidth.
- **Fused QKV projection**: W_q | W_k | W_v concatenated at load time into one `[hidden, 3072]` matrix. A split kernel writes Q to its buffer and K/V straight into the KV cache, which replaces the two cache copies. +5.6% decode, −4 ms prefill.
- **Split-KV decode attention** (see [Split-KV Decode Attention](#split-kv-decode-attention)): decode attention at kv_len 2032 goes from 465 to 24.5 µs per layer, and 2K-context decode from ~125 to ~225 tok/s.
- **Tensor Core prefill attention** (see [Tensor Core Prefill Attention](#tensor-core-prefill-attention)): FlashAttention-lite v5 ported with causal tile skip, GQA, query offset and strided KV. Prefill attention at 2002 tokens goes from 448 to 3.9 ms, and the 2K-token prefill from ~715 to ~275 ms.
- **Fused gate/up projection** (see [Fused gate/up projection](#fused-gategup-projection)): W_gate and W_up interleaved in 32-column groups; decode runs one GEMV with SwiGLU in its epilogue, prefill one wmma GEMM. +2.4% decode at 128 tokens, −12.6% prefill at 2002 tokens.
- **CUDA Graph decode** (see [CUDA Graph decode](#cuda-graph-decode)): one captured step replayed per token, positions read on the device, argmax writes the next input token, host one step behind. Idle per step 550 → 168 µs; +10% decode at 128 tokens.

Not implemented yet, in the order the measurements suggest:

1. **Decode kernel time** — with graphs, a step is 95% kernel time (3.41 ms). Argmax is one ~70 µs single-block launch per token; a multi-block argmax would cut most of that. The remaining ~168 µs idle per step comes from gaps between the 197 graph nodes; fewer, larger kernels (e.g. fusing RMSNorm and residual adds into neighbouring kernels) would shrink both.

2. **GEMM (prefill): shared tile reuse + double buffering** — ~95% of a 2K-token prefill's GPU time. With gate/up fused, the same work in one launch twice as wide took 24% less time, which isn't explained yet; ncu on this kernel is the first step. Each warp in the wmma kernel loads its own A and B fragments, so the 16 warps in a block don't share tiles, and loads and MMAs run back to back. Block-level shared-memory tiles with double buffering would cut global traffic and hide load latency.

3. **GEMM: Swizzled shared memory** — Not yet profiled for bank conflicts. If ncu shows them on the staged path, XOR-swizzled layouts (as in CUTLASS, and as the prefill attention kernel already uses) remove them.

4. **KV Cache quantization** — Storing the KV cache in INT8/FP8 would halve its memory and allow longer sequences. Requires dequantize-on-the-fly in the attention kernel.

5. **PagedAttention** — The current contiguous KV cache wastes memory on short sequences. Paged allocation (vLLM-style) enables efficient batching of variable-length sequences.

6. **Speculative decoding** — A smaller draft model proposes tokens, and the full model verifies them in parallel. Reported to give 2–3× effective decode speedup.

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
