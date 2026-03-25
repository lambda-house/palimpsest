#!/bin/bash
set -euo pipefail

# Convert merged model to GGUF Q5_K_M for CPU inference
# Run after training completes

MODEL_DIR="${1:-./output/instructions/merged_16bit}"
OUTPUT="${2:-./pelevin-qwen32b.gguf}"
QUANT="${3:-q5_k_m}"

echo "=== Converting ${MODEL_DIR} to GGUF (${QUANT}) ==="

# Install llama.cpp converter if needed
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
    pip install gguf
fi

# Convert to f16 GGUF first
echo "Converting to f16 GGUF..."
python3 llama.cpp/convert_hf_to_gguf.py "${MODEL_DIR}" \
    --outfile model-f16.gguf \
    --outtype f16

# Quantize
echo "Quantizing to ${QUANT}..."
# Build llama-quantize if not present
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    cd llama.cpp
    cmake -B build
    cmake --build build --target llama-quantize -j$(nproc)
    cd ..
fi

llama.cpp/build/bin/llama-quantize model-f16.gguf "${OUTPUT}" "${QUANT}"

# Cleanup intermediate
rm -f model-f16.gguf

echo ""
ls -lh "${OUTPUT}"
echo "=== Done: ${OUTPUT} ==="
