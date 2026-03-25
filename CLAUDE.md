# Palimpsest

Literary style mixing -- blend multiple authors' writing styles in real-time using LoRA adapters on llama.cpp.

## Architecture

```
Browser (mixer sliders + chat)
  <-> WebSocket
Node.js app (queue, sessions, prompt enhancement)
  <-> HTTP/SSE (OpenAI-compatible)
llama-server (Mistral Nemo abliterated + 5 EN adapters)
```

## Authors (English, public domain)

Lovecraft, Conan Doyle, Poe, Wilde, Jack London

Base model: Mistral Nemo abliterated (12B, no content refusals)

## Running locally

```bash
# Start llama-server
llama-server -m models/nemo-abliterated-base-q6k.gguf \
  --lora models/lovecraft_en.lora.gguf --lora models/doyle_en.lora.gguf \
  --lora models/poe_en.lora.gguf --lora models/wilde_en.lora.gguf \
  --lora models/london_en.lora.gguf \
  --lora-init-without-apply --port 8081 --ctx-size 8192

# Start app
npm install
LLAMA_URL_EN=http://localhost:8081 \
ADAPTERS_EN="lovecraft_en:Lovecraft,doyle_en:Conan Doyle,poe_en:Poe,wilde_en:Wilde,london_en:London" \
node server.js
```

## Project structure

```
palimpsest/
  server.js, index.html, sources.html  -- web app
  src/
    llama.js        -- llama.cpp client with LoRA routing
    enhance.js      -- Claude prompt enhancement (optional)
    summarize.js    -- Claude conversation summarization
    ws-handler.js   -- WebSocket handler
    queue.js        -- request queue
    users.js        -- session store
  pipeline/         -- corpus scraping and processing
  training/         -- GPU training scripts
  dpo/              -- DPO evaluation tools
  helm/palimpsest/  -- Kubernetes deployment
```

## Environment variables

- `LLAMA_URL_EN` -- English llama-server URL
- `ADAPTERS_EN` -- English adapters: `id:Label,id:Label,...`
- `ANTHROPIC_API_KEY` -- for prompt enhancement (optional)
- `ENHANCE_MODEL` -- Claude model for enhancement (default: claude-sonnet-4-20250514)

## Training a new author

1. Scrape from Project Gutenberg: `pipeline/scrape_gutenberg.py`
2. Generate instruction pairs: `pipeline/format_instructions.py` (needs OPENROUTER_API_KEY)
3. Train on GPU: `training/train.py` (~$2-5 per author on L40S)
4. Convert adapter to LoRA GGUF
5. Add to llama-server `--lora` flags + ADAPTERS_EN env
