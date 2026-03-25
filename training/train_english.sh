#!/bin/bash
set -euo pipefail

# Train English LoRA adapters on Mistral Nemo abliterated
# Run AFTER Russian training completes

source /tmp/venv/bin/activate
cd /tmp/pelevin

EN_MODEL="natong19/Mistral-Nemo-Instruct-2407-abliterated"
COMMON_ARGS="--model $EN_MODEL --mode instructions --batch-size 4 --grad-accum 2 --lr 1e-4 --lora-r 32 --lora-alpha 16 --epochs 3 --max-seq-length 2048"

# === Convert English base model to GGUF ===
EN_BASE_PATH=$(find /tmp/hf_cache/hub -name 'config.json' -path '*abliterated*' -exec dirname {} \; | head -1)
if [ -z "$EN_BASE_PATH" ]; then
  echo "English base model not cached yet — will be downloaded during first training run"
fi

# === Train each English author ===
train_author() {
  local name=$1
  local data_dir=$2

  echo ""
  echo "=== Training $name ==="
  python3 train.py $COMMON_ARGS --data-dir "$data_dir" --output-dir "/tmp/pelevin/output-$name"
  ls -lh "/tmp/pelevin/output-$name/instructions/final/adapter_model.safetensors"

  # Find base path (may have been downloaded during training)
  local base_path=$(find /tmp/hf_cache/hub -name 'config.json' -path '*abliterated*' -exec dirname {} \; | head -1)

  echo "=== Converting $name adapter to LoRA GGUF ==="
  python3 /tmp/llama.cpp/convert_lora_to_gguf.py \
    --base "$base_path" \
    --outfile "/tmp/pelevin/$name.lora.gguf" \
    "/tmp/pelevin/output-$name/instructions/final/"
  ls -lh "/tmp/pelevin/$name.lora.gguf"
}

train_author lovecraft_en /tmp/pelevin/data-lovecraft_en
train_author doyle_en /tmp/pelevin/data-doyle_en
train_author poe_en /tmp/pelevin/data-poe_en
train_author wilde_en /tmp/pelevin/data-wilde_en
train_author london_en /tmp/pelevin/data-london_en

# === Convert English base model to GGUF ===
EN_BASE_PATH=$(find /tmp/hf_cache/hub -name 'config.json' -path '*abliterated*' -exec dirname {} \; | head -1)
echo ""
echo "=== Converting English base model to GGUF ==="
python3 /tmp/llama.cpp/convert_hf_to_gguf.py "$EN_BASE_PATH" \
  --outfile /tmp/pelevin/nemo-abliterated-f16.gguf --outtype f16
/tmp/llama.cpp/build/bin/llama-quantize \
  /tmp/pelevin/nemo-abliterated-f16.gguf /tmp/pelevin/nemo-abliterated-base-q6k.gguf q6_k
rm -f /tmp/pelevin/nemo-abliterated-f16.gguf
ls -lh /tmp/pelevin/nemo-abliterated-base-q6k.gguf

echo ""
echo "=== ALL ENGLISH DONE ==="
echo "Files to download:"
ls -lh /tmp/pelevin/nemo-abliterated-base-q6k.gguf
ls -lh /tmp/pelevin/*_en.lora.gguf
echo ""
echo "Adapters (safetensors):"
ls -lh /tmp/pelevin/output-*_en/instructions/final/adapter_model.safetensors
