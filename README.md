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

**Poe 60% + Lovecraft 40%** -- *a scholar descends into the cellar of his ancestral home and discovers a passage into darkness*

> I have said that I found the passage clear of obstructions. This was not entirely true, for as I advanced I encountered several low-swinging beams, which I was compelled to feel my way around. About me the darkness thickened; then, after a long descent, I began to notice a queer change in the atmosphere -- a stifling, sickish quality which my brain told me could not be altogether due to the natural mustiness of an uninhabited place.
>
> Suddenly my foot slipped on something wet, and I started back despite the fact that I could see nothing. A cold shiver ran through me; and then, as I recovered my composure, I realized that what my feet had slipped on must have been some mere trick of the mold and fungus. Yet a moment later I felt less sure; for my hand fell upon what seemed to be a great slab of stone, wet and clammy and giving off an evil odor.
>
> Scarcely had I gone a few paces when a foul gust from out the darkness seemed to strike me in the face like the puff from some opening grave. And then I knew, even before I heard the sound, that behind me had come a sudden rush of air through some door or passage which I must have left open. In another second I was enveloped in total darkness -- a gloom so utter that it seemed to stifle my very breath.

**Wilde 50% + Conan Doyle 50%** -- *Lord Henry observes a fellow dinner guest and delivers a devastating character study*

> "I adore her. I love people who are very, very stupid. There is a certain charm in every one of them. They never try to be more than what they are. Their ignorance is so considerable that I really respect it. And they always like you. They are such excellent listeners."
>
> "Give me one example," she cried.
>
> "Very well!" said Lord Henry, and, looking across the room, met Mrs. Cheveley's eye. "Do you see that lady in the corner? She has never read a book in her life. No literature whatever has ever stirred those uninteresting volumes of which Hayman is so unjustly proud. And a few weeks ago one of the most celebrated novelists of our day had to present her with an autographed copy of his latest work."
>
> "She was very flattered, I suppose?"
>
> "Flattered! She was deeply moved. Poor dear Lady Agnes! You know, Duchess, she is sweetly, wonderfully silly."

**Jack London 70% + Poe 30%** -- *two trappers in the Yukon discuss the wolves that have been following them -- wolves that don't behave like any natural wolf*

> "They ain't no common timber wolves," Joe said after a while. "I never seen their like before."
>
> "They travel too much in packs," Bill said. "I never seen so many together before."
>
> "And did you notice how they moved?" Joe asked.
>
> Bill nodded. "Like a lot of dogs or cattle, all alike at the same time. They're not natural, that's what."
>
> "I've got an idea," Bill said suddenly. "They ain't no common wolves and they don't act like any wolves I ever saw. But if we was to take a look at them, we'd find they've got the faces of wolves --"
>
> "I know," Joe interrupted.
>
> "And their feet would be like the feet of wolves?" Bill went on.
>
> "But if it was all wolves except that, what do you think?"
>
> "I don't know. I ain't never heard of anything like it before."

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
