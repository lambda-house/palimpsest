#!/usr/bin/env python3
"""Step 2: Corpus Statistics & Quality Check.

Computes word counts, sentence lengths, lexical diversity,
punctuation density, dialogue ratio, etc.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# Russian stopwords (common function words)
RUSSIAN_STOPWORDS = set("""
и в не на я что он с а как это по но к из то за его ещё она
был бы мы ну вы ты уже от так все же всё мне ему было когда
вот только при до где после нет ли если об ей ко мной них ней
ему нас про нам нем без нее ими вам чем она они каждый между
тем кто тут ведь сам там еще свой себя свою этот вас его чего
или нибудь ни какой другой весь здесь может надо два мой раз
будто чтобы другие лишь под через ото три ней знать уж никто
опять наш больше однако именно должен перед более тебя хотя
был были была было быть есть чтоб даже один этой ему одна
""".split())


def sentence_split(text: str) -> list[str]:
    """Split text into sentences. Uses regex, falls back gracefully."""
    try:
        from razdel import sentenize
        return [s.text for s in sentenize(text)]
    except ImportError:
        pass
    # Regex fallback
    sents = re.split(r'(?<=[.!?…])\s+', text)
    return [s for s in sents if s.strip()]


def compute_lexical_diversity(words: list[str], window: int = 1000) -> float:
    """Type-token ratio averaged over windows."""
    if len(words) < window:
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    ratios = []
    for i in range(0, len(words) - window + 1, window):
        w = words[i:i + window]
        ratios.append(len(set(w)) / len(w))
    return sum(ratios) / len(ratios) if ratios else 0.0


def compute_punctuation_density(text: str, word_count: int) -> dict:
    """Frequency of specific punctuation per 1000 words."""
    if word_count == 0:
        return {}
    factor = 1000.0 / word_count
    return {
        '«»': text.count('«') * factor,
        '—': text.count('—') * factor,
        '…': text.count('…') * factor,
        ';': text.count(';') * factor,
        ':': text.count(':') * factor,
        '()': text.count('(') * factor,
    }


def compute_dialogue_ratio(text: str) -> float:
    """Percentage of text inside «» quotes."""
    in_dialogue = 0
    total = len(text)
    if total == 0:
        return 0.0
    inside = False
    for ch in text:
        if ch == '«':
            inside = True
        elif ch == '»':
            inside = False
        elif inside:
            in_dialogue += 1
    return in_dialogue / total * 100


def analyze_file(filepath: Path) -> dict:
    """Compute statistics for a single file."""
    text = filepath.read_text(encoding='utf-8')
    words_raw = text.split()
    words_lower = [w.lower().strip('.,!?;:—«»""…()[]') for w in words_raw]
    words_lower = [w for w in words_lower if w]

    word_count = len(words_raw)
    char_count = len(text)
    unique_words = len(set(words_lower))

    sentences = sentence_split(text)
    avg_sentence_length = (
        sum(len(s.split()) for s in sentences) / len(sentences)
        if sentences else 0.0
    )

    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    para_sentence_counts = [len(sentence_split(p)) for p in paragraphs]
    avg_para_length = (
        sum(para_sentence_counts) / len(para_sentence_counts)
        if para_sentence_counts else 0.0
    )

    lex_diversity = compute_lexical_diversity(words_lower)
    punct_density = compute_punctuation_density(text, word_count)
    dialogue_ratio = compute_dialogue_ratio(text)

    # Top 50 words excluding stopwords
    content_words = [w for w in words_lower if w not in RUSSIAN_STOPWORDS and len(w) > 1]
    top_words = Counter(content_words).most_common(50)

    # Flags
    flags = []
    if word_count < 5000:
        flags.append('suspiciously_short')
    if '\x00' in text or '\ufffd' in text or '????' in text:
        flags.append('encoding_issues')
    if dialogue_ratio > 80:
        flags.append('high_dialogue_ratio')

    return {
        'filename': filepath.name,
        'word_count': word_count,
        'char_count': char_count,
        'unique_words': unique_words,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'avg_paragraph_length_sentences': round(avg_para_length, 1),
        'lexical_diversity': round(lex_diversity, 4),
        'punctuation_density': {k: round(v, 2) for k, v in punct_density.items()},
        'dialogue_ratio': round(dialogue_ratio, 2),
        'top_50_words': top_words,
        'flags': flags,
    }


def main():
    parser = argparse.ArgumentParser(description='Compute corpus statistics')
    parser.add_argument('--input-dir', default='./corpus/clean',
                        help='Directory with clean .txt files')
    parser.add_argument('--stats-json', default='./corpus_stats.json',
                        help='Path for JSON stats output')
    parser.add_argument('--report', default='./corpus_report.txt',
                        help='Path for human-readable report')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    txt_files = sorted(input_dir.glob('*.txt'))
    if not txt_files:
        print(f"ERROR: No .txt files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Analyzing {len(txt_files)} files...")

    all_stats = []
    flagged = []

    for fp in tqdm(txt_files, desc="Analyzing"):
        stats = analyze_file(fp)
        all_stats.append(stats)
        if stats['flags']:
            flagged.append(stats)

    # Aggregate stats
    total_words = sum(s['word_count'] for s in all_stats)
    total_chars = sum(s['char_count'] for s in all_stats)
    avg_lex = sum(s['lexical_diversity'] for s in all_stats) / len(all_stats)
    avg_dialogue = sum(s['dialogue_ratio'] for s in all_stats) / len(all_stats)

    aggregate = {
        'total_files': len(all_stats),
        'total_words': total_words,
        'total_chars': total_chars,
        'avg_lexical_diversity': round(avg_lex, 4),
        'avg_dialogue_ratio': round(avg_dialogue, 2),
        'estimated_tokens': int(total_words * 2.0),
    }

    output = {
        'aggregate': aggregate,
        'per_file': all_stats,
        'flagged': [s['filename'] for s in flagged],
    }

    # Write JSON
    Path(args.stats_json).write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8'
    )

    # Write human-readable report
    lines = []
    lines.append("=" * 60)
    lines.append("PELEVIN CORPUS REPORT")
    lines.append("=" * 60)
    lines.append(f"\nTotal files: {aggregate['total_files']}")
    lines.append(f"Total words: {aggregate['total_words']:,}")
    lines.append(f"Total characters: {aggregate['total_chars']:,}")
    lines.append(f"Estimated tokens: {aggregate['estimated_tokens']:,}")
    lines.append(f"Avg lexical diversity: {aggregate['avg_lexical_diversity']:.4f}")
    lines.append(f"Avg dialogue ratio: {aggregate['avg_dialogue_ratio']:.1f}%")

    lines.append(f"\n{'─' * 60}")
    lines.append("PER-FILE STATISTICS")
    lines.append(f"{'─' * 60}")

    for s in sorted(all_stats, key=lambda x: x['word_count'], reverse=True):
        lines.append(f"\n  {s['filename']}")
        lines.append(f"    Words: {s['word_count']:,}  |  Chars: {s['char_count']:,}")
        lines.append(f"    Avg sentence: {s['avg_sentence_length']} words  |  Avg paragraph: {s['avg_paragraph_length_sentences']} sentences")
        lines.append(f"    Lexical diversity: {s['lexical_diversity']:.4f}  |  Dialogue: {s['dialogue_ratio']:.1f}%")
        if s['flags']:
            lines.append(f"    ⚠ FLAGS: {', '.join(s['flags'])}")

    if flagged:
        lines.append(f"\n{'─' * 60}")
        lines.append("FLAGGED FILES")
        lines.append(f"{'─' * 60}")
        for s in flagged:
            lines.append(f"  {s['filename']}: {', '.join(s['flags'])}")

    report_text = '\n'.join(lines) + '\n'
    Path(args.report).write_text(report_text, encoding='utf-8')

    print(f"\nTotal words: {total_words:,}")
    print(f"Estimated tokens: {aggregate['estimated_tokens']:,}")
    print(f"Flagged files: {len(flagged)}")
    print(f"Stats written to {args.stats_json}")
    print(f"Report written to {args.report}")


if __name__ == '__main__':
    main()
