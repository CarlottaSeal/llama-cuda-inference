#!/bin/bash
# ============================================================
#  Benchmark script for llama-cuda-inference
#
#  Usage: ./scripts/benchmark.sh <model_dir> [build_dir]
#
#  Runs inference with varying prompt lengths and generation
#  counts to measure prefill throughput and decode latency.
# ============================================================

set -e

MODEL_DIR="${1:?Usage: $0 <model_dir> [build_dir]}"
BUILD_DIR="${2:-build}"
BIN="$BUILD_DIR/llama_inference"

if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Build first: cd $BUILD_DIR && cmake .. && make -j"
    exit 1
fi

echo "============================================"
echo "  llama-cuda-inference Benchmark"
echo "============================================"
echo "Model: $MODEL_DIR"
echo "Binary: $BIN"
echo ""

# Warmup
echo "--- Warmup ---"
$BIN "$MODEL_DIR" "Hello" 4 2>/dev/null || true
echo ""

# Prefill benchmark: vary prompt length
echo "============================================"
echo "  Prefill Throughput (tokens/s)"
echo "============================================"
for len in 8 32 64 128 256 512; do
    # Generate a prompt of approximately $len tokens
    PROMPT=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * ($len // 10 + 1))" 2>/dev/null || echo "Hello world. This is a test prompt for benchmarking purposes.")
    echo "--- Prompt ~${len} tokens, generate 1 ---"
    $BIN "$MODEL_DIR" "$PROMPT" 1 2>&1 | grep -E "(Prefill|Tokens)"
    echo ""
done

# Decode benchmark: fixed prompt, vary generation length
echo "============================================"
echo "  Decode Throughput (tokens/s)"
echo "============================================"
for gen in 16 32 64 128 256; do
    echo "--- Prompt 16 tokens, generate $gen ---"
    $BIN "$MODEL_DIR" "The meaning of life is a question" $gen 2>&1 | grep -E "(Prefill|Decode|Total)"
    echo ""
done

echo "============================================"
echo "  Done"
echo "============================================"
