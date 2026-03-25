#!/usr/bin/env python3
"""Step 2: Generate multiple completions per prompt from the fine-tuned model.

Requires the model to be loaded in Ollama. Generates N completions per prompt
at varying temperatures. Output: completions.jsonl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

SYSTEM_PROMPT = (
    "Ты — писатель художественной прозы. Пиши в стиле, характерном для "
    "русской постмодернистской прозы: философские отступления, буддийские "
    "метафоры, сатира на современность, многослойная ирония, игра с "
    "поп-культурными отсылками."
)

TEMPERATURES = [0.5, 0.7, 0.9, 1.1, 1.3]


def generate_ollama(model: str, prompt: str, temperature: float,
                    num_predict: int = 1024) -> str | None:
    """Generate a single completion via Ollama CLI."""
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "num_predict": num_predict,
        },
    })

    try:
        result = subprocess.run(
            ["curl", "-s", "http://localhost:11434/api/generate",
             "-d", payload],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        return data.get("response", "").strip()
    except Exception as e:
        print(f"WARNING: Generation failed: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate completions for DPO')
    parser.add_argument('--prompts', default='./dpo/prompts.jsonl')
    parser.add_argument('--output', default='./dpo/completions.jsonl')
    parser.add_argument('--model', default='pelevin',
                        help='Ollama model name')
    parser.add_argument('--num-completions', type=int, default=5,
                        help='Completions per prompt')
    parser.add_argument('--max-tokens', type=int, default=1024)
    args = parser.parse_args()

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"ERROR: {prompts_path} not found", file=sys.stderr)
        sys.exit(1)

    prompts = [json.loads(l) for l in prompts_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(prompts)} prompts")
    print(f"Generating {args.num_completions} completions each = {len(prompts) * args.num_completions} total")

    temperatures = TEMPERATURES[:args.num_completions]
    # Pad with extra values if more completions requested
    while len(temperatures) < args.num_completions:
        temperatures.append(0.8 + len(temperatures) * 0.1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(output_path, 'w', encoding='utf-8') as f:
        for prompt_data in tqdm(prompts, desc="Generating"):
            prompt_text = prompt_data["prompt"]
            completions = []

            for i, temp in enumerate(temperatures):
                text = generate_ollama(args.model, prompt_text, temp, args.max_tokens)
                if text:
                    completions.append({
                        "text": text,
                        "temperature": temp,
                        "index": i,
                    })

            result = {
                "prompt": prompt_text,
                "type": prompt_data.get("type", ""),
                "constraint": prompt_data.get("constraint", ""),
                "completions": completions,
            }
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()

    total_completions = sum(len(r["completions"]) for r in results)
    print(f"\nGenerated {total_completions} completions for {len(results)} prompts → {args.output}")


if __name__ == '__main__':
    main()
