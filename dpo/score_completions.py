#!/usr/bin/env python3
"""Step 3: Auto-score completions and create preference pairs.

Uses OpenRouter API to score each completion on prompt adherence and style
fidelity. Creates chosen/rejected pairs. Output: preferences.jsonl
"""

import asyncio
import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

SCORING_SYSTEM = """Ты оцениваешь качество прозаического текста по двум критериям.

Для каждого текста выдай два балла от 1 до 10:

1. **Следование заданию** (adherence): насколько текст соответствует заданию? Раскрыта ли тема? Если задание просит про музыку — текст про музыку? Если просит диалог — есть диалог?
2. **Стилистическое качество** (style): насколько текст литературно интересен? Есть ли ирония, философская глубина, неожиданные метафоры, игра слов?

Отвечай ТОЛЬКО в формате JSON:
{"adherence": N, "style": N}

Без пояснений."""


async def score_completion(prompt: str, text: str, client, semaphore, model: str) -> dict | None:
    """Score a single completion via API."""
    async with semaphore:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                user_msg = f"ЗАДАНИЕ: {prompt}\n\nТЕКСТ:\n{text[:3000]}"
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    max_tokens=50,
                    messages=[
                        {"role": "system", "content": SCORING_SYSTEM},
                        {"role": "user", "content": user_msg},
                    ],
                )
                content = response.choices[0].message.content.strip()
                # Parse JSON, handle markdown code blocks
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                return json.loads(content)
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"WARNING: Scoring failed: {e}", file=sys.stderr)
                    return None


async def score_all(completions_data: list, model: str, max_concurrent: int = 20):
    """Score all completions and create preference pairs."""
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    # Flatten all completions for scoring
    tasks = []
    task_index = []  # (prompt_idx, completion_idx)

    for pi, item in enumerate(completions_data):
        for ci, comp in enumerate(item["completions"]):
            tasks.append(score_completion(item["prompt"], comp["text"], client, semaphore, model))
            task_index.append((pi, ci))

    print(f"Scoring {len(tasks)} completions...")
    results = []
    batch_size = max_concurrent * 2
    for i in tqdm(range(0, len(tasks), batch_size), desc="Scoring"):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)

    # Attach scores back to completions
    for (pi, ci), score in zip(task_index, results):
        if score:
            completions_data[pi]["completions"][ci]["adherence"] = score.get("adherence", 5)
            completions_data[pi]["completions"][ci]["style"] = score.get("style", 5)
            completions_data[pi]["completions"][ci]["total"] = (
                score.get("adherence", 5) + score.get("style", 5)
            )
        else:
            completions_data[pi]["completions"][ci]["adherence"] = 5
            completions_data[pi]["completions"][ci]["style"] = 5
            completions_data[pi]["completions"][ci]["total"] = 10

    return completions_data


def create_preference_pairs(scored_data: list, min_gap: int = 3) -> list:
    """Create chosen/rejected pairs from scored completions.

    min_gap: minimum total score difference between chosen and rejected.
    """
    pairs = []

    for item in scored_data:
        comps = item["completions"]
        if len(comps) < 2:
            continue

        # Sort by total score
        sorted_comps = sorted(comps, key=lambda c: c.get("total", 10), reverse=True)
        best = sorted_comps[0]
        worst = sorted_comps[-1]

        gap = best.get("total", 10) - worst.get("total", 10)
        if gap < min_gap:
            continue  # Skip if no clear contrast

        pairs.append({
            "prompt": item["prompt"],
            "chosen": best["text"],
            "rejected": worst["text"],
            "chosen_score": {"adherence": best["adherence"], "style": best["style"]},
            "rejected_score": {"adherence": worst["adherence"], "style": worst["style"]},
            "score_gap": gap,
            "needs_review": True,
        })

    return pairs


def main():
    parser = argparse.ArgumentParser(description='Score completions and create preference pairs')
    parser.add_argument('--completions', default='./dpo/completions.jsonl')
    parser.add_argument('--output', default='./dpo/preferences.jsonl')
    parser.add_argument('--scored-output', default='./dpo/scored_completions.jsonl',
                        help='Save scored completions for inspection')
    parser.add_argument('--model', default='google/gemini-2.0-flash-001')
    parser.add_argument('--min-gap', type=int, default=3,
                        help='Minimum score gap for a valid preference pair')
    parser.add_argument('--max-concurrent', type=int, default=20)
    args = parser.parse_args()

    completions_path = Path(args.completions)
    if not completions_path.exists():
        print(f"ERROR: {completions_path} not found", file=sys.stderr)
        sys.exit(1)

    data = [json.loads(l) for l in completions_path.read_text().splitlines() if l.strip()]
    total_completions = sum(len(d["completions"]) for d in data)
    print(f"Loaded {len(data)} prompts with {total_completions} total completions")

    # Score
    scored = asyncio.run(score_all(data, args.model, args.max_concurrent))

    # Save scored completions
    scored_path = Path(args.scored_output)
    scored_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scored_path, 'w', encoding='utf-8') as f:
        for item in scored:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Scored completions → {scored_path}")

    # Create pairs
    pairs = create_preference_pairs(scored, args.min_gap)

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\nPreference pairs: {len(pairs)} (from {len(data)} prompts, min_gap={args.min_gap})")
    if pairs:
        avg_gap = sum(p["score_gap"] for p in pairs) / len(pairs)
        print(f"Average score gap: {avg_gap:.1f}")

    # Summary
    print(f"\nNext steps:")
    print(f"  1. Review pairs:  python dpo/review_ui.py")
    print(f"  2. Train DPO:     python dpo/train_dpo.py")


if __name__ == '__main__':
    main()
