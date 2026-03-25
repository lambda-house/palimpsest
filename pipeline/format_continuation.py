#!/usr/bin/env python3
"""Step 3: Training Data Formatter — Continuation Format.

Chunks text into token-sized sequences with overlap,
splitting at paragraph boundaries.
"""

import argparse
import json
import sys
from pathlib import Path

import tiktoken
from tqdm import tqdm


def load_tokenizer(tokenizer_path: str | None = None):
    """Load tokenizer. Uses tiktoken cl100k_base by default."""
    if tokenizer_path:
        try:
            from sentencepiece import SentencePieceProcessor
            sp = SentencePieceProcessor(model_file=tokenizer_path)
            return sp.encode, sp.decode
        except ImportError:
            print("WARNING: sentencepiece not installed, falling back to tiktoken",
                  file=sys.stderr)
    enc = tiktoken.get_encoding('cl100k_base')
    return enc.encode, enc.decode


def chunk_text(text: str, encode, chunk_size: int, overlap: int) -> list[str]:
    """Split text into token-sized chunks at paragraph boundaries."""
    # Split on chapter markers first
    chapters = text.split('---CHAPTER---')
    chunks = []

    for chapter in chapters:
        chapter = chapter.strip()
        if not chapter:
            continue
        paragraphs = chapter.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # Build chunks from paragraphs
        current_parts = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(encode(para))

            # If single paragraph exceeds chunk_size, split it by sentences
            if para_tokens > chunk_size:
                if current_parts:
                    chunks.append('\n\n'.join(current_parts))
                    current_parts = []
                    current_tokens = 0
                # Split long paragraph at sentence boundaries
                sentences = _split_sentences(para)
                sent_parts = []
                sent_tokens = 0
                for sent in sentences:
                    st = len(encode(sent))
                    if sent_tokens + st > chunk_size and sent_parts:
                        chunks.append(' '.join(sent_parts))
                        # Overlap: keep last few sentences
                        overlap_parts = []
                        overlap_tokens = 0
                        for s in reversed(sent_parts):
                            t = len(encode(s))
                            if overlap_tokens + t > overlap:
                                break
                            overlap_parts.insert(0, s)
                            overlap_tokens += t
                        sent_parts = overlap_parts
                        sent_tokens = overlap_tokens
                    sent_parts.append(sent)
                    sent_tokens += st
                if sent_parts:
                    chunks.append(' '.join(sent_parts))
                continue

            if current_tokens + para_tokens > chunk_size and current_parts:
                chunks.append('\n\n'.join(current_parts))

                # Overlap: keep trailing paragraphs
                overlap_parts = []
                overlap_tokens = 0
                for p in reversed(current_parts):
                    pt = len(encode(p))
                    if overlap_tokens + pt > overlap:
                        break
                    overlap_parts.insert(0, p)
                    overlap_tokens += pt
                current_parts = overlap_parts
                current_tokens = overlap_tokens

            current_parts.append(para)
            current_tokens += para_tokens

        if current_parts:
            chunks.append('\n\n'.join(current_parts))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter for fallback."""
    import re
    sents = re.split(r'(?<=[.!?…])\s+', text)
    return [s for s in sents if s.strip()]


def main():
    parser = argparse.ArgumentParser(description='Format continuation training data')
    parser.add_argument('--input-dir', default='./corpus/clean',
                        help='Directory with clean .txt files')
    parser.add_argument('--output-dir', default='./training_data/continuation',
                        help='Output directory for JSONL')
    parser.add_argument('--chunk-size', type=int, default=2048,
                        help='Target chunk size in tokens')
    parser.add_argument('--overlap', type=int, default=256,
                        help='Overlap between chunks in tokens')
    parser.add_argument('--tokenizer-path', default=None,
                        help='Path to sentencepiece model (optional)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encode, _ = load_tokenizer(args.tokenizer_path)

    input_dir = Path(args.input_dir)
    txt_files = sorted(input_dir.glob('*.txt'))
    if not txt_files:
        print(f"ERROR: No .txt files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    all_chunks = []
    chunk_lengths = []

    for fp in tqdm(txt_files, desc="Chunking"):
        text = fp.read_text(encoding='utf-8')
        chunks = chunk_text(text, encode, args.chunk_size, args.overlap)
        for chunk in chunks:
            # Remove chapter markers from final output
            chunk = chunk.replace('---CHAPTER---', '').strip()
            if not chunk:
                continue
            token_len = len(encode(chunk))
            all_chunks.append(chunk)
            chunk_lengths.append(token_len)

    # Write JSONL
    jsonl_path = output_dir / 'continuation.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

    # Compute stats
    if chunk_lengths:
        avg_len = sum(chunk_lengths) / len(chunk_lengths)
        min_len = min(chunk_lengths)
        max_len = max(chunk_lengths)
    else:
        avg_len = min_len = max_len = 0

    # Histogram buckets of 256 tokens
    histogram = {}
    for length in chunk_lengths:
        bucket = (length // 256) * 256
        key = f"{bucket}-{bucket + 255}"
        histogram[key] = histogram.get(key, 0) + 1

    stats = {
        'total_chunks': len(all_chunks),
        'avg_chunk_length': round(avg_len, 1),
        'min_chunk_length': min_len,
        'max_chunk_length': max_len,
        'total_tokens': sum(chunk_lengths),
        'histogram': dict(sorted(histogram.items())),
    }

    stats_path = output_dir / 'continuation_stats.json'
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    print(f"\nTotal chunks: {stats['total_chunks']}")
    print(f"Avg chunk length: {stats['avg_chunk_length']} tokens")
    print(f"Min/Max: {stats['min_chunk_length']}/{stats['max_chunk_length']} tokens")
    print(f"Total tokens: {stats['total_tokens']:,}")


if __name__ == '__main__':
    main()
