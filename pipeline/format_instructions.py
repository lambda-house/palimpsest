#!/usr/bin/env python3
"""Step 4: Training Data Formatter — Instruction Format.

Splits text into passages, generates synthetic instruction prompts
via OpenRouter API (or Anthropic API) or heuristics.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

DEFAULT_SYSTEM_PROMPT = "Ты — писатель художественной прозы."

AUTHOR_SYSTEM_PROMPTS = {
    "pelevin": (
        "Ты — писатель художественной прозы. Пиши в стиле, характерном для "
        "русской постмодернистской прозы: философские отступления, буддийские "
        "метафоры, сатира на современность, многослойная ирония, игра с "
        "поп-культурными отсылками."
    ),
    "lovecraft": (
        "Ты — писатель художественной прозы. Пиши в стиле космического ужаса: "
        "нарастающая атмосфера страха перед непознаваемым, архаичная лексика, "
        "древние культы, безумие от столкновения с запредельным, детальные "
        "описания гнетущих пейзажей и зловещей архитектуры."
    ),
    "marquez": (
        "Ты — писатель художественной прозы. Пиши в стиле магического реализма: "
        "чудесное переплетается с обыденным, длинные плавные предложения, "
        "циклическое время, семейные саги, латиноамериканский колорит, "
        "одиночество и любовь как центральные темы."
    ),
    "sorokin": (
        "Ты — писатель художественной прозы. Пиши в стиле русского "
        "концептуализма: деконструкция языка и литературных форм, шоковые "
        "переходы от стилизации к абсурду, телесность, провокация, "
        "гротескная гиперболизация советского и постсоветского быта."
    ),
    "bulgakov": (
        "Ты — писатель художественной прозы. Пиши в стиле сатирической "
        "фантастики: дьявольщина в московском быту, лирический гротеск, "
        "точные бытовые детали на фоне сверхъестественного, острая сатира "
        "на бюрократию и мещанство, тёплый юмор."
    ),
    "strugatsky": (
        "Ты — писатель художественной прозы. Пиши в стиле философской "
        "фантастики: контакт с неизведанным как зеркало человечности, "
        "этические дилеммы прогресса, живые диалоги учёных и практиков, "
        "советский быт на фоне космических масштабов."
    ),
    "dovlatov": (
        "Ты — писатель художественной прозы. Пиши в стиле минимализма: "
        "короткие точные предложения, сухой юмор, автобиографичность, "
        "абсурд повседневности, ни слова лишнего, трагикомизм "
        "советской и эмигрантской жизни."
    ),
    # English authors
    "lovecraft_en": (
        "You are a fiction writer. Write in the style of cosmic horror: "
        "mounting atmosphere of dread before the unknowable, archaic and "
        "elaborate vocabulary, ancient cults and forbidden knowledge, "
        "madness from confronting the beyond, oppressive landscapes "
        "and sinister non-Euclidean architecture."
    ),
    "doyle_en": (
        "You are a fiction writer. Write in the style of Victorian detective fiction: "
        "precise deductive reasoning, sharp observation of detail, "
        "dry British wit, atmospheric London settings, "
        "a narrator of limited perspective marveling at a brilliant companion, "
        "methodical unraveling of mystery through logic and evidence."
    ),
    "poe_en": (
        "You are a fiction writer. Write in the style of Gothic horror: "
        "psychological torment, unreliable narrators consumed by obsession, "
        "poetic and rhythmic prose, mounting dread and inevitable doom, "
        "dark romanticism, the macabre rendered with terrible beauty."
    ),
    "wilde_en": (
        "You are a fiction writer. Write in the style of aesthetic wit: "
        "devastating epigrammatic dialogue, paradox and inversion, "
        "social satire delivered with elegance, beauty as philosophy, "
        "the mask of frivolity concealing sharp moral observation."
    ),
    "london_en": (
        "You are a fiction writer. Write in the style of naturalist adventure: "
        "raw visceral prose, man against indifferent nature, "
        "survival and primal instinct, vivid physical description, "
        "the struggle between civilization and the wild, "
        "unflinching portrayal of violence and endurance."
    ),
}

INSTRUCTION_SYSTEM = (
    "Ты помощник, который создаёт краткие творческие задания для писателя. "
    "На основе данного отрывка прозы сформулируй краткое задание (1-2 предложения) "
    "на русском языке, которое могло бы привести к написанию этого отрывка. "
    "Не упоминай автора. Отвечай только заданием, без пояснений."
)


def split_into_passages(text: str, min_words: int = 300, max_words: int = 800) -> list[str]:
    """Split text into passages of 1-5 paragraphs, targeting word count range."""
    # Remove chapter markers
    text = text.replace('---CHAPTER---', '\n\n')
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    passages = []
    current_parts = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        current_parts.append(para)
        current_words += para_words

        if current_words >= min_words:
            if current_words >= max_words or len(current_parts) >= 5:
                passages.append('\n\n'.join(current_parts))
                current_parts = []
                current_words = 0

    # Handle remaining
    if current_parts:
        if passages and current_words < min_words // 2:
            # Append to last passage if too short
            passages[-1] += '\n\n' + '\n\n'.join(current_parts)
        else:
            passages.append('\n\n'.join(current_parts))

    return passages


def generate_instruction_heuristic(passage: str) -> str:
    """Generate a synthetic instruction from passage content using heuristics."""
    has_dialogue = '«' in passage and '»' in passage
    dialogue_count = passage.count('«')

    # Detect philosophical/descriptive content
    philo_markers = ['смысл', 'сознани', 'реальност', 'истин', 'бытие', 'пустот',
                     'медитац', 'ум', 'дух', 'свобод', 'иллюзи', 'просветлен']
    has_philosophy = any(m in passage.lower() for m in philo_markers)

    # Detect action
    action_markers = ['побежал', 'выстрелил', 'ударил', 'бросил', 'закричал',
                      'схватил', 'упал', 'взорвал', 'вскочил', 'ринулся']
    has_action = any(m in passage.lower() for m in action_markers)

    if has_dialogue and dialogue_count >= 3:
        # Extract a hint about the dialogue topic
        first_quote = re.search(r'«([^»]{10,60})', passage)
        hint = first_quote.group(1) if first_quote else "различные темы"
        return f"Напиши сцену с диалогом, в которой персонажи обсуждают {hint}."
    elif has_philosophy:
        return "Напиши философское размышление о природе реальности и сознания в ироничном стиле."
    elif has_action:
        return "Напиши динамичную сцену с неожиданным поворотом событий."
    else:
        # Generic
        first_sentence = re.split(r'[.!?…]', passage)[0].strip()
        if len(first_sentence) > 80:
            first_sentence = first_sentence[:80]
        return f"Напиши прозаический отрывок, начинающийся с темы: «{first_sentence}»."


async def generate_instruction_openrouter(passage: str, client, semaphore, model: str) -> str:
    """Generate instruction via OpenRouter API with rate limiting."""
    async with semaphore:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model=model,
                    max_tokens=200,
                    messages=[
                        {"role": "system", "content": INSTRUCTION_SYSTEM},
                        {"role": "user", "content": passage[:3000]},
                    ],
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"WARNING: API error (attempt {attempt + 1}): {e}, "
                          f"retrying in {wait}s", file=sys.stderr)
                    await asyncio.sleep(wait)
                else:
                    print(f"WARNING: API failed after {max_retries} attempts: {e}",
                          file=sys.stderr)
                    return generate_instruction_heuristic(passage)


async def process_with_api_ordered(passages: list[str], max_concurrent: int = 10,
                                   model: str = "google/gemini-2.0-flash-001") -> list[str]:
    """Process all passages through OpenRouter API, preserving order."""
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    semaphore = asyncio.Semaphore(max_concurrent)

    print(f"Using model: {model}")
    print(f"Concurrency: {max_concurrent}")

    tasks = [generate_instruction_openrouter(p, client, semaphore, model) for p in passages]
    results = []
    batch_size = max_concurrent * 2
    for i in tqdm(range(0, len(tasks), batch_size), desc="API batches"):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    return results


def main():
    parser = argparse.ArgumentParser(description='Format instruction training data')
    parser.add_argument('--input-dir', default='./corpus/clean',
                        help='Directory with clean .txt files')
    parser.add_argument('--output-dir', default='./training_data/instructions',
                        help='Output directory for JSONL')
    parser.add_argument('--skip-api', action='store_true',
                        help='Use heuristic instructions instead of API calls')
    parser.add_argument('--model', default='google/gemini-2.0-flash-001',
                        help='OpenRouter model for instruction generation')
    parser.add_argument('--min-words', type=int, default=300,
                        help='Minimum passage word count')
    parser.add_argument('--max-words', type=int, default=800,
                        help='Maximum passage word count')
    parser.add_argument('--max-concurrent', type=int, default=10,
                        help='Max concurrent API requests')
    parser.add_argument('--author', default=None,
                        help='Author key for system prompt (pelevin, lovecraft, marquez, etc.)')
    parser.add_argument('--system-prompt', default=None,
                        help='Custom system prompt (overrides --author)')
    args = parser.parse_args()

    # Determine system prompt
    if args.system_prompt:
        SYSTEM_PROMPT = args.system_prompt
    elif args.author and args.author in AUTHOR_SYSTEM_PROMPTS:
        SYSTEM_PROMPT = AUTHOR_SYSTEM_PROMPTS[args.author]
    elif args.author:
        print(f"WARNING: Unknown author '{args.author}', using default", file=sys.stderr)
        SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
    else:
        # Try to guess from input dir name
        dir_name = Path(args.input_dir).name
        SYSTEM_PROMPT = AUTHOR_SYSTEM_PROMPTS.get(dir_name, DEFAULT_SYSTEM_PROMPT)
        if dir_name in AUTHOR_SYSTEM_PROMPTS:
            print(f"Auto-detected author: {dir_name}")

    print(f"System prompt: {SYSTEM_PROMPT[:80]}...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    txt_files = sorted(input_dir.glob('*.txt'))
    if not txt_files:
        print(f"ERROR: No .txt files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all passages
    all_passages = []
    for fp in tqdm(txt_files, desc="Splitting passages"):
        text = fp.read_text(encoding='utf-8')
        passages = split_into_passages(text, args.min_words, args.max_words)
        all_passages.extend(passages)

    print(f"Total passages: {len(all_passages)}")

    # Generate instructions
    if args.skip_api:
        print("Using heuristic instruction generation (--skip-api)")
        instructions = [generate_instruction_heuristic(p) for p in
                        tqdm(all_passages, desc="Generating instructions")]
    else:
        print("Using OpenRouter API for instruction generation")
        instructions = asyncio.run(
            process_with_api_ordered(all_passages, args.max_concurrent, args.model)
        )

    # Write JSONL
    jsonl_path = output_dir / 'instructions.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for instruction, passage in zip(instructions, all_passages):
            record = {
                'messages': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': instruction},
                    {'role': 'assistant', 'content': passage},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Stats
    instruction_lengths = [len(inst.split()) for inst in instructions]
    passage_lengths = [len(p.split()) for p in all_passages]

    stats = {
        'total_pairs': len(all_passages),
        'avg_passage_length': round(sum(passage_lengths) / len(passage_lengths), 1) if passage_lengths else 0,
        'avg_instruction_length': round(sum(instruction_lengths) / len(instruction_lengths), 1) if instruction_lengths else 0,
        'min_passage_length': min(passage_lengths) if passage_lengths else 0,
        'max_passage_length': max(passage_lengths) if passage_lengths else 0,
        'instruction_length_distribution': {
            f"{(l // 5) * 5}-{(l // 5) * 5 + 4}": 0
            for l in range(0, max(instruction_lengths, default=0) + 5, 5)
        },
    }
    # Fill histogram
    for l in instruction_lengths:
        bucket = f"{(l // 5) * 5}-{(l // 5) * 5 + 4}"
        stats['instruction_length_distribution'][bucket] = \
            stats['instruction_length_distribution'].get(bucket, 0) + 1
    # Remove empty buckets
    stats['instruction_length_distribution'] = {
        k: v for k, v in stats['instruction_length_distribution'].items() if v > 0
    }

    stats_path = output_dir / 'instruction_stats.json'
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    print(f"\nTotal instruction pairs: {stats['total_pairs']}")
    print(f"Avg passage length: {stats['avg_passage_length']} words")
    print(f"Avg instruction length: {stats['avg_instruction_length']} words")


if __name__ == '__main__':
    main()
