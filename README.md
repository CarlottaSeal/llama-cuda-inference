# llama-cuda-inference

A CUDA inference engine for Llama 3.2 1B with **hand-written GPU kernels**. No cuBLAS, no cuDNN, no frameworks — every kernel from scratch.

```
$ ./llama_inference ./models/Llama-3.2-1B "The meaning of life is"
GPU: NVIDIA GeForce RTX 4090 (SM 8.9, 24.0 GB, 128 SMs)
Model: Llama 3.2 1B (L=16, H=2048, Q=32, KV=8, D=64, FFN=8192, V=128256)
--- Prefill (12 tokens) ---
Prefill: 45.2 ms (265 tok/s)
--- Generating ---
The meaning of life is a question that has been asked by philosophers,
theologians, and everyday people for centuries...
```

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
| **RoPE** | `rope.cu` | 1 thread per dim pair | freq = 1/θ^(2i/d), rotation matrix per (pos, pair) |
| **GEMM Naive** | `gemm.cu` | 1 thread per output | O(MNK) global loads, ~2% peak — the baseline |
| **GEMM Tiled** | `gemm.cu` | 32×32 shmem tiles | 32× fewer global memory loads, collaborative loading |
| **GEMM wmma** | `gemm.cu` | Tensor Core 16×16×16 | `nvcuda::wmma` fragments, 4×4 warp grid, FP16→FP32 accumulate |
| **Attention Naive** | `attention.cu` | Materialize S matrix | 3-pass (QK^T → causal mask + softmax → SV), shmem for scores |
| **Flash Attention** | `attention.cu` | Online softmax, tiled | KV tiles of 32, running (m, l, O) — O(head_dim) memory per block |

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

**wmma (Tensor Core)** → Hardware 16×16×16 matrix multiply-accumulate via `nvcuda::wmma`. Each warp computes a 16×16 output tile in a single instruction. 4×4 warp grid (16 warps = 512 threads per block) computes a 64×64 output tile.
```
wmma::load_matrix_sync(a_frag, sA + ...);   // load 16×16 from shmem
wmma::load_matrix_sync(b_frag, sB + ...);   // load 16×16 from shmem
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // hardware MMA!
```

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

**GQA (Grouped Query Attention)**: 32 Q heads share 8 KV heads (4:1 ratio). Mapping: `kv_head = q_head / 4`.

## Project Structure

```
llama-cuda-inference/
├── CMakeLists.txt                      # Build config (SM arch configurable)
├── README.md
├── KNOWLEDGE.md                        # Deep technical notes (1050 lines)
├── DEVPLAN.md                          # Incremental development plan
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
│   │   ├── safetensors.cpp             # Safetensors parser (mmap, hand-written JSON)
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
- CUDA Toolkit 11.0+ (for wmma support)
- CMake 3.18+
- GCC/Clang with C++17
- ~2.5 GB VRAM (1.2 GB weights + KV cache + activations)

### Build
```bash
mkdir build && cd build

# Auto-detect GPU architecture
cmake ..

# Or specify explicitly (80=Ampere, 86=GA10x, 89=Ada, 90=Blackwell)
cmake .. -DCUDA_ARCH=89

make -j$(nproc)
```

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

Tests cover every kernel with CPU reference implementations and cross-variant consistency checks (e.g., flash attention output matches naive attention).

## Design Decisions

### FP16 Storage, FP32 Accumulation
All weights and activations stored as FP16 (`half`) to minimize memory bandwidth. All dot products and reductions accumulate in FP32 to prevent precision loss, then convert back to FP16 for storage.

### Weight Transpose at Runtime
HuggingFace stores linear layer weights as `[out_features, in_features]`. Our GEMM computes `C[M,N] = A[M,K] × B[K,N]`, so we transpose weights on-the-fly with a dedicated kernel. A static buffer avoids per-call allocation. This trades a small amount of compute for keeping the GEMM interface clean and reusable.

### KV Cache Layout
```
kv_cache_k[layer]: [max_seq_len, num_kv_heads × head_dim]
kv_cache_v[layer]: [max_seq_len, num_kv_heads × head_dim]
```
Separate K/V per layer. During prefill, positions `[0, seq_len)` are filled. During decode, position `[kv_len]` is appended. The attention kernel reads `[0, kv_len)` regardless of mode.

### Hand-Written JSON Parser
Both safetensors headers and tokenizer.json use a minimal hand-written JSON parser. This avoids the nlohmann/json dependency (300K+ lines) for parsing two specific, well-defined JSON structures.

### Safetensors with mmap
Weight files are memory-mapped (`mmap`), letting the OS manage page loading. Tensor data is accessed via offset into the mapped region — zero-copy for FP16 weights.

## Profiling Guide

### NSight Systems (Timeline)
```bash
nsys profile --stats=true ./llama_inference ./models/Llama-3.2-1B "Hello" 32
nsys stats report.nsys-rep
```
Look for: kernel launch overhead, memory transfer gaps, kernel concurrency.

### NSight Compute (Kernel Deep Dive)
```bash
# Profile GEMM kernel
ncu --target-processes all --set full \
    --kernel-name "gemm_wmma_kernel" \
    ./llama_inference ./models/Llama-3.2-1B "Hello" 8

# Profile Flash Attention
ncu --kernel-name "flash_attention_gqa_kernel" \
    ./llama_inference ./models/Llama-3.2-1B "Hello" 8
```
Key metrics: achieved occupancy, memory throughput, compute throughput, warp stall reasons.

### Expected Performance (RTX 4090)

| Metric | Expected | Notes |
|--------|----------|-------|
| Prefill throughput | 200-400 tok/s | Depends on prompt length |
| Decode latency | 20-50 ms/tok | Single token, memory-bound |
| GEMM efficiency (wmma) | 30-50% peak TFLOPS | Room for swizzle, double buffering |
| Flash Attention | ~2× faster than naive | For kv_len > 128 |

Measure yours:
```bash
# Benchmark different prompt lengths
for len in 16 64 256; do
    echo "=== Prompt length ~$len ==="
    ./llama_inference ./models/Llama-3.2-1B "$(python3 -c "print('hello ' * $len)")" 64
done
```

## Known Optimization Opportunities

These are intentionally left as future work — each is a well-understood optimization I can discuss in depth:

1. **GEMM: Register tiling + double buffering** — Current wmma kernel loads tiles synchronously. Overlapping global memory loads with computation via double-buffered shared memory would hide latency. Expected ~1.5-2× speedup.

2. **GEMM: Swizzled shared memory** — Current layout causes bank conflicts on shmem loads. Swizzling (XOR-based remapping) eliminates conflicts. See CUTLASS `SmemSwizzle`.

3. **Weight pre-transpose** — Currently transposing weights every forward pass. Pre-transposing at load time eliminates ~30 transpose kernels per forward call.

4. **Fused QKV projection** — Three separate GEMMs for Q, K, V can be fused into one `[hidden, q_dim+2×kv_dim]` GEMM, reducing kernel launch overhead and improving memory coalescing.

5. **Flash Attention v2 optimizations** — Current implementation uses a simple tiling scheme. FA2 adds: non-square tiles, warp specialization (producer/consumer), and pipelining within the K/V tile loop.

6. **KV Cache quantization** — Storing KV cache in INT8/FP8 would halve cache memory, enabling longer sequences. Requires dequantize-on-the-fly in the attention kernel.

7. **Persistent kernels for decode** — During decode (seq=1), kernel launch overhead dominates. A persistent kernel that stays resident and processes all layers would eliminate launch overhead.

8. **PagedAttention** — Current contiguous KV cache wastes memory on short sequences. Paged allocation (vLLM-style) enables efficient batching of variable-length sequences.

9. **Speculative decoding** — Use a smaller draft model to propose tokens, verify in parallel with the full model. Can achieve 2-3× effective decode speedup.

## License

MIT

## Acknowledgments

- [Meta AI](https://ai.meta.com/) for the Llama model family
- [Tri Dao](https://tridao.me/) for the Flash Attention algorithm
- NVIDIA for CUDA, wmma intrinsics, and NSight profiling tools
