# Palimpsest

**Literary style mixing** -- blend multiple authors' writing styles in real-time using LoRA adapters.

A single base model serves as the "pianist." LoRA adapters are the "sheet music" -- one per author, ~0.4 GB each. At inference time, adapters are mixed at user-specified ratios: 60% Poe + 40% Lovecraft produces Gothic cosmic horror. The mix changes on every request without reloading anything.

## Authors

| Author | Style | Corpus |
|--------|-------|--------|
| **Lovecraft** | Cosmic horror, archaic dread, unknowable entities | 262K words |
| **Conan Doyle** | Deductive logic, Victorian adventure, precise observation | 733K words |
| **Poe** | Gothic horror, unreliable narrator, psychological tension | 515K words |
| **Wilde** | Devastating wit, social satire, ornate prose | 308K words |
| **Jack London** | Raw naturalism, survival, frontier stoicism | 747K words |

All English training corpora are from Project Gutenberg (public domain). Base model: [Mistral Nemo abliterated](https://huggingface.co/natong19/Mistral-Nemo-Instruct-2407-abliterated) (12B, no content refusals).

Pre-trained adapters and the base model GGUF are available on HuggingFace: [lambdahouse/palimpsest-lora-adapters](https://huggingface.co/lambdahouse/palimpsest-lora-adapters)

## How it works

```
Browser (mixer sliders + chat)
  |  WebSocket
  v
Node.js app (queue, sessions, prompt enhancement)
  |  HTTP + SSE (OpenAI-compatible)
  v
llama-server (llama.cpp)
  Base: Mistral Nemo abliterated Q6_K (9.4 GB)
  + lovecraft_en.lora.gguf   (0.4 GB)
  + doyle_en.lora.gguf       (0.4 GB)
  + poe_en.lora.gguf         (0.4 GB)
  + wilde_en.lora.gguf       (0.4 GB)
  + london_en.lora.gguf      (0.4 GB)
  Total: ~12 GB RAM
```

Per-request, the client sends adapter weights:
```json
{"lora": [{"id": 0, "scale": 0.6}, {"id": 2, "scale": 0.4}]}
```

llama.cpp adds the weighted LoRA corrections at inference time -- no model reload, no merge, instant style change.

## Quick start

```bash
# 1. Install llama.cpp
brew install llama.cpp  # macOS
# or build from source: https://github.com/ggml-org/llama.cpp

# 2. Download models from HuggingFace
# https://huggingface.co/lambdahouse/palimpsest-lora-adapters
# Place in models/ directory (~12 GB total)

# 3. Start llama-server
llama-server \
  -m models/nemo-abliterated-base-q6k.gguf \
  --lora models/lovecraft_en.lora.gguf \
  --lora models/doyle_en.lora.gguf \
  --lora models/poe_en.lora.gguf \
  --lora models/wilde_en.lora.gguf \
  --lora models/london_en.lora.gguf \
  --lora-init-without-apply \
  --port 8081 --ctx-size 8192

# 4. Start the app
npm install
LLAMA_URL_EN=http://localhost:8081 \
ADAPTERS_EN="lovecraft_en:Lovecraft,doyle_en:Conan Doyle,poe_en:Poe,wilde_en:Wilde,london_en:London" \
node server.js

# 5. Open http://localhost:3000
```

### Optional: prompt enhancement

If you set `ANTHROPIC_API_KEY`, user prompts are enriched by Claude before being sent to llama -- the small model produces dramatically better output with well-crafted prompts. This is optional; without it, prompts are passed through as-is.

## Example outputs

**Poe 60% + Lovecraft 40%** -- *a man descends into the cellar of his ancestral home*

> The stone steps of the cellar descended steeply, and at its bottom there yawned only blackness. Then, turning my attention to the cellar walls, I noticed for the first time a low door of heavy planks which lay hid behind a great heap of casks. The sight of the crudely fashioned planks and rusty bolt filled me with a strange excitement which was not all of terror; for I seemed to sense in it something of that subtle, hidden quality which haunts the oldest parts of our most familiar abodes.

**Wilde 50% + Conan Doyle 50%** -- *a dinner party revelation delivered with devastating wit*

> "Come, come, my dear young lady," said Lord Henry, buttoning his evening gloves. "We all do our little mischief here and there." ... "I am not so sure of that. True, we have only just been introduced; but I _have_ had the pleasure of listening to you for over a quarter of an hour, without your knowing me. That generally gives us a certain right to be impertinent."

**Jack London 70% + Poe 30%** -- *trappers in the Yukon sensing something wrong*

> "There ain't nothin' to do but go on," said Bill. "It's sure death for you to go on," Bill urged. "I'll go alone, an' I'll see that your share of the skins comes to you right enough." Again the other man indicated his dissent by a negative movement of the head and a grunt.

## Training your own adapter

```bash
# 1. Scrape author texts from Project Gutenberg
python pipeline/scrape_gutenberg.py --author "Edgar Allan Poe" --output-dir corpus/clean/poe_en

# 2. Generate instruction pairs (needs OPENROUTER_API_KEY)
python pipeline/format_instructions.py --input-dir corpus/clean/poe_en --output-dir data/poe_en/instructions --author poe_en

# 3. Validate and split
python pipeline/validate.py --instructions-dir data/poe_en/instructions
python pipeline/split.py --instructions-dir data/poe_en/instructions --output-dir data/poe_en/final

# 4. Train on GPU (~2-4h on L40S, ~$2-5)
python training/train.py \
  --model natong19/Mistral-Nemo-Instruct-2407-abliterated \
  --data-dir data/poe_en/final \
  --output-dir output/poe_en \
  --mode instructions --batch-size 4 --grad-accum 2 \
  --lr 1e-4 --lora-r 32 --lora-alpha 16 --epochs 3

# 5. Convert to LoRA GGUF for llama.cpp
python llama.cpp/convert_lora_to_gguf.py \
  --base /path/to/nemo-abliterated \
  --outfile models/poe_en.lora.gguf \
  output/poe_en/instructions/final/

# 6. Add --lora flag to llama-server and entry to ADAPTERS_EN
```

## Cocktail recipes

| Recipe | Mix | Vibe |
|--------|-----|------|
| **The Depths Below** | Poe 60% + Lovecraft 40% | Gothic descent into cosmic unknown |
| **The Wit and the Deduction** | Wilde 50% + Doyle 50% | Sharp social observation meets logical precision |
| **The Last Trail** | London 70% + Poe 30% | Frontier survival with creeping psychological dread |
| **The Yellow Fog** | Doyle 40% + Lovecraft 30% + Wilde 30% | Victorian mystery in a city hiding eldritch secrets |

## Project structure

```
palimpsest/
  server.js, index.html, sources.html   -- web application
  src/                                   -- server modules (llama client, prompt enhancement, queue)
  pipeline/                              -- corpus scraping and processing
  training/                              -- GPU training scripts
  dpo/                                   -- DPO evaluation tools
  helm/                                  -- Kubernetes deployment
  models/                               -- GGUF files (gitignored, ~12 GB)
  corpus/                               -- author texts (gitignored)
  data/                                  -- training JSONL (gitignored)
```

## Hardware requirements

| Setup | RAM | Speed |
|-------|-----|-------|
| MacBook (Apple Silicon 16GB+) | ~12 GB | ~20 tok/sec |
| CPU server (Hetzner EX-series) | ~12 GB | ~8-12 tok/sec |
| GPU server (RTX 3090+) | 12 GB VRAM | ~30+ tok/sec |

Five adapters loaded simultaneously add ~2 GB to the base model's 9.4 GB.

## License

MIT. Training corpora are sourced from Project Gutenberg (public domain).
