# llama-cuda-inference

A CUDA inference engine for Llama 3.2 1B with **hand-written GPU kernels**. No cuBLAS, no cuDNN, no frameworks — every kernel from scratch.

```
$ ./llama_inference ./models/Llama-3.2-1B "The meaning of life is" 64
```

**Status:** every kernel is unit-tested against a CPU reference (13 tests), and the full
forward pass (prefill + KV-cache decode) runs clean under `compute-sanitizer` with random
weights. It has not yet been run on the real checkpoint or compared against the HF
reference, so there is no generated text or end-to-end performance data here yet.

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
- ~3 GB VRAM: 2.5 GB of FP16 weights, a 0.5 GB transpose scratch buffer (sized for the LM head), plus KV cache and activations

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

### Weight Transpose at Runtime
HuggingFace stores linear layer weights as `[out_features, in_features]`. Our GEMM computes `C[M,N] = A[M,K] × B[K,N]`, so each projection first transposes its weight into a static scratch buffer, then runs the GEMM. That keeps one GEMM interface for everything, but it is not cheap: 7 transposes per layer plus the LM head is 113 transpose kernels per forward, and the 128256×2048 LM head alone is ~0.5 GB read and ~0.5 GB written. For decode, which should only need to read each weight once, this likely dominates the step time. Pre-transposing once at load time (or a `C = A × Bᵀ` GEMM) is the first optimization on the list below.

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
# Profile GEMM kernel (each forward launches it ~113 times; cap the count)
ncu --set full --kernel-name gemm_wmma_kernel --launch-count 8 \
    ./llama_inference ./models/Llama-3.2-1B "Hello" 8

# Profile Flash Attention
ncu --kernel-name flash_attention_gqa_kernel --launch-count 8 \
    ./llama_inference ./models/Llama-3.2-1B "Hello" 8
```
Key metrics: achieved occupancy, memory throughput, compute throughput, warp stall reasons.

### Performance

Not measured yet. To sweep prefill and decode throughput:
```bash
./scripts/benchmark.sh ./models/Llama-3.2-1B [build_dir]
```

## Known Optimization Opportunities

Not implemented yet:

1. **Weight pre-transpose** — Weights are re-transposed on every forward pass (113 transpose kernels, ~1 GB of extra traffic from the LM head alone). Transposing once at load time removes all of it.

2. **GEMM: Shared tile reuse + double buffering** — Each warp in the wmma kernel loads its own A and B fragments, so the 16 warps in a block don't share tiles, and loads and MMAs run back to back. Block-level shared-memory tiles with double buffering would cut global traffic and hide load latency.

3. **GEMM: Swizzled shared memory** — Not yet profiled for bank conflicts; if ncu shows them on the staged path, XOR-swizzled layouts (as in CUTLASS) remove them.

4. **Fused QKV projection** — Three separate GEMMs for Q, K, V can be fused into one `[hidden, q_dim+2×kv_dim]` GEMM, reducing kernel launch overhead and improving memory coalescing.

5. **Flash Attention v2 optimizations** — Current implementation uses a simple tiling scheme. FA2 adds: non-square tiles, warp specialization (producer/consumer), and pipelining within the K/V tile loop.

6. **KV Cache quantization** — Storing KV cache in INT8/FP8 would halve cache memory, enabling longer sequences. Requires dequantize-on-the-fly in the attention kernel.

7. **Persistent kernels for decode** — During decode (seq=1), kernel launch overhead dominates. A persistent kernel that stays resident and processes all layers would eliminate launch overhead.

8. **PagedAttention** — Current contiguous KV cache wastes memory on short sequences. Paged allocation (vLLM-style) enables efficient batching of variable-length sequences.

9. **Speculative decoding** — Use a smaller draft model to propose tokens, verify in parallel with the full model. Can achieve 2-3× effective decode speedup.

## Limitations

- Batch size 1, greedy decoding only (`sample_top_p` is a stub that falls back to greedy).
- Model shape is hardcoded in `llama_config.h` for Llama 3.2 1B, with `max_seq_len` = 2048.
- The tokenizer doesn't implement Llama 3's regex pre-tokenization; BPE merges run over the whole byte string, so some inputs can tokenize differently from HF.

## License

MIT

## Acknowledgments

- [Meta AI](https://ai.meta.com/) for the Llama model family
- [Tri Dao](https://tridao.me/) for the Flash Attention algorithm
- NVIDIA for CUDA, wmma intrinsics, and NSight profiling tools
