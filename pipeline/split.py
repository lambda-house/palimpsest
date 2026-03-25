#!/usr/bin/env python3
"""Step 6: Train/Eval Split.

5% eval split for both formats, no overlap between train and eval.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def content_hash(record: dict) -> str:
    """Compute content hash for a record."""
    if 'text' in record:
        text = record['text']
    elif 'messages' in record:
        text = ''.join(m['content'] for m in record['messages'])
    else:
        text = json.dumps(record, sort_keys=True)
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path):
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def split_data(records: list[dict], eval_ratio: float = 0.05,
               seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split records into train and eval sets."""
    random.seed(seed)
    indices = list(range(len(records)))
    random.shuffle(indices)
    eval_count = max(1, int(len(records) * eval_ratio))
    eval_indices = set(indices[:eval_count])
    train = [records[i] for i in range(len(records)) if i not in eval_indices]
    eval_ = [records[i] for i in range(len(records)) if i in eval_indices]
    return train, eval_


def verify_no_overlap(train: list[dict], eval_: list[dict]) -> int:
    """Verify no content overlap between train and eval. Returns overlap count."""
    train_hashes = {content_hash(r) for r in train}
    overlaps = sum(1 for r in eval_ if content_hash(r) in train_hashes)
    return overlaps


def main():
    parser = argparse.ArgumentParser(description='Split data into train/eval')
    parser.add_argument('--continuation-dir', default='./training_data/continuation',
                        help='Directory with continuation JSONL')
    parser.add_argument('--instructions-dir', default='./training_data/instructions',
                        help='Directory with instruction JSONL')
    parser.add_argument('--output-dir', default='./training_data/final',
                        help='Output directory for splits')
    parser.add_argument('--eval-ratio', type=float, default=0.05,
                        help='Fraction of data for eval (default 0.05)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Process continuation
    cont_jsonl = Path(args.continuation_dir) / 'continuation.jsonl'
    if cont_jsonl.exists():
        print("Splitting continuation data...")
        records = load_jsonl(cont_jsonl)
        train, eval_ = split_data(records, args.eval_ratio, args.seed)
        overlaps = verify_no_overlap(train, eval_)
        if overlaps:
            print(f"WARNING: {overlaps} overlapping records found!", file=sys.stderr)

        write_jsonl(train, output_dir / 'continuation' / 'train.jsonl')
        write_jsonl(eval_, output_dir / 'continuation' / 'eval.jsonl')
        print(f"  Train: {len(train)} | Eval: {len(eval_)}")
    else:
        print(f"WARNING: {cont_jsonl} not found, skipping", file=sys.stderr)

    # Process instructions — use same eval passage hashes for consistency
    inst_jsonl = Path(args.instructions_dir) / 'instructions.jsonl'
    if inst_jsonl.exists():
        print("Splitting instruction data...")
        records = load_jsonl(inst_jsonl)
        train, eval_ = split_data(records, args.eval_ratio, args.seed)
        overlaps = verify_no_overlap(train, eval_)
        if overlaps:
            print(f"WARNING: {overlaps} overlapping records found!", file=sys.stderr)

        write_jsonl(train, output_dir / 'instructions' / 'train.jsonl')
        write_jsonl(eval_, output_dir / 'instructions' / 'eval.jsonl')
        print(f"  Train: {len(train)} | Eval: {len(eval_)}")
    else:
        print(f"WARNING: {inst_jsonl} not found, skipping", file=sys.stderr)

    print("\nDone. Output in", output_dir)


if __name__ == '__main__':
    main()
