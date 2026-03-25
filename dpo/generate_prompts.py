#!/usr/bin/env python3
"""Step 1: Generate diverse prompts for DPO preference collection.

Uses OpenRouter API to generate prompts covering various types, topics,
and constraints. Output: prompts.jsonl
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

PROMPT_TYPES = [
    "эссе",
    "сцена с диалогом",
    "описание места",
    "внутренний монолог",
    "философское размышление",
    "сатирическая зарисовка",
    "сцена действия",
    "притча или аллегория",
    "письмо или дневниковая запись",
    "пересказ мифа на современный лад",
]

TOPICS = [
    "современная музыка", "социальные сети", "искусственный интеллект",
    "московское метро", "буддизм и медитация", "реклама и маркетинг",
    "компьютерные игры", "русская деревня", "криптовалюта",
    "современное искусство", "советская ностальгия", "мода и гламур",
    "политика и власть", "смерть и бессмертие", "любовь и одиночество",
    "телевидение", "эмиграция", "война", "детство", "наркотики и сознание",
]

CONSTRAINTS = [
    "с иронией и многослойным подтекстом",
    "с буддийскими метафорами",
    "с отсылками к поп-культуре",
    "в мрачном тоне",
    "с юмором и абсурдом",
    "с философскими отступлениями",
    "с неожиданным финалом",
    "от первого лица",
    "с точки зрения неодушевленного предмета",
    "как разговор двух интеллектуалов",
]

SYSTEM = (
    "Ты генерируешь разнообразные творческие задания для русскоязычного писателя-постмодерниста. "
    "Задания должны быть конкретными (с указанием темы, места, ситуации), но не слишком длинными "
    "(1-3 предложения). Каждое задание должно быть уникальным и отличаться от остальных. "
    "Не упоминай конкретных авторов. Отвечай только заданием, без пояснений и нумерации."
)


def generate_batch_prompt(prompt_type: str, topics: list[str], constraint: str) -> str:
    topic_list = ", ".join(topics)
    return (
        f"Сгенерируй 5 разных творческих заданий. Тип: {prompt_type}. "
        f"Темы для вдохновения (не обязательно использовать все): {topic_list}. "
        f"Стилистическое ограничение: {constraint}. "
        f"Каждое задание на отдельной строке."
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate diverse prompts for DPO')
    parser.add_argument('--output', default='./dpo/prompts.jsonl')
    parser.add_argument('--model', default='google/gemini-2.0-flash-001')
    parser.add_argument('--num-batches', type=int, default=20,
                        help='Number of API calls, each produces ~5 prompts')
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    import random
    random.seed(42)

    all_prompts = []
    seen = set()

    for i in range(args.num_batches):
        ptype = PROMPT_TYPES[i % len(PROMPT_TYPES)]
        constraint = CONSTRAINTS[i % len(CONSTRAINTS)]
        topics = random.sample(TOPICS, 4)

        user_msg = generate_batch_prompt(ptype, topics, constraint)

        try:
            response = client.chat.completions.create(
                model=args.model,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
            )
            text = response.choices[0].message.content.strip()
            lines = [l.strip().lstrip("0123456789.-) ") for l in text.split("\n") if l.strip()]

            for line in lines:
                if len(line) > 20 and line not in seen:
                    seen.add(line)
                    all_prompts.append({
                        "prompt": line,
                        "type": ptype,
                        "constraint": constraint,
                    })

            print(f"Batch {i+1}/{args.num_batches}: {len(lines)} prompts (total: {len(all_prompts)})")
        except Exception as e:
            print(f"WARNING: Batch {i+1} failed: {e}", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for p in all_prompts:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\nGenerated {len(all_prompts)} unique prompts → {args.output}")


if __name__ == '__main__':
    main()
