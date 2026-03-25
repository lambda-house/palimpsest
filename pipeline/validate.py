#!/usr/bin/env python3
"""Step 5: Deduplication & Validation.

MinHash dedup, JSONL validation, encoding corruption check.
"""

import argparse
import json
import sys
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


def compute_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Compute MinHash for a text using word 3-grams."""
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(len(words) - 2):
        gram = ' '.join(words[i:i + 3])
        m.update(gram.encode('utf-8'))
    return m


def extract_text_from_record(record: dict) -> str:
    """Extract the main text from a JSONL record (either format)."""
    if 'text' in record:
        return record['text']
    if 'messages' in record:
        for msg in record['messages']:
            if msg['role'] == 'assistant':
                return msg['content']
    return ''


def check_encoding(text: str) -> list[str]:
    """Check for encoding corruption."""
    issues = []
    if '\x00' in text:
        issues.append('null_bytes')
    if '\ufffd' in text:
        issues.append('replacement_characters')
    if '????' in text:
        issues.append('question_mark_sequences')
    return issues


def validate_and_dedup(jsonl_path: Path, threshold: float = 0.85) -> tuple[list[dict], dict]:
    """Validate and deduplicate a JSONL file.

    Returns (clean_records, report_dict).
    """
    records = []
    parse_errors = 0
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append((line_num, record))
            except json.JSONDecodeError as e:
                print(f"WARNING: JSON parse error at {jsonl_path.name}:{line_num}: {e}",
                      file=sys.stderr)
                parse_errors += 1

    # Validation
    valid_records = []
    empty_fields = 0
    too_short = 0
    too_long = 0
    encoding_issues = 0

    for line_num, record in records:
        text = extract_text_from_record(record)

        if not text.strip():
            empty_fields += 1
            continue

        word_count = len(text.split())
        if word_count < 50:
            too_short += 1
            continue
        if word_count > 3000:
            too_long += 1
            continue

        issues = check_encoding(text)
        if issues:
            encoding_issues += 1
            print(f"WARNING: Encoding issues in {jsonl_path.name}:{line_num}: "
                  f"{', '.join(issues)}", file=sys.stderr)
            continue

        valid_records.append(record)

    # Deduplication using MinHash LSH
    print(f"  Deduplicating {len(valid_records)} records...")
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = []

    for i, record in enumerate(tqdm(valid_records, desc="  Computing hashes", leave=False)):
        text = extract_text_from_record(record)
        mh = compute_minhash(text)
        minhashes.append(mh)
        try:
            lsh.insert(str(i), mh)
        except ValueError:
            pass  # Duplicate key, will be caught in next pass

    # Find duplicates
    seen = set()
    duplicates = set()
    for i, mh in enumerate(minhashes):
        if i in duplicates:
            continue
        results = lsh.query(mh)
        candidates = [int(r) for r in results if int(r) != i and int(r) not in duplicates]
        for j in candidates:
            # Keep the longer one
            text_i = extract_text_from_record(valid_records[i])
            text_j = extract_text_from_record(valid_records[j])
            if len(text_j) > len(text_i):
                duplicates.add(i)
                break
            else:
                duplicates.add(j)

    clean_records = [r for i, r in enumerate(valid_records) if i not in duplicates]

    report = {
        'input_file': jsonl_path.name,
        'total_input': len(records),
        'parse_errors': parse_errors,
        'empty_fields': empty_fields,
        'too_short': too_short,
        'too_long': too_long,
        'encoding_issues': encoding_issues,
        'duplicates_removed': len(duplicates),
        'final_count': len(clean_records),
    }

    return clean_records, report


def main():
    parser = argparse.ArgumentParser(description='Validate and deduplicate JSONL files')
    parser.add_argument('--continuation-dir', default='./training_data/continuation',
                        help='Directory with continuation JSONL')
    parser.add_argument('--instructions-dir', default='./training_data/instructions',
                        help='Directory with instruction JSONL')
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='Jaccard similarity threshold for dedup')
    parser.add_argument('--report', default='./validation_report.txt',
                        help='Path for validation report')
    args = parser.parse_args()

    reports = []

    # Process continuation data
    cont_jsonl = Path(args.continuation_dir) / 'continuation.jsonl'
    if cont_jsonl.exists():
        print(f"Validating {cont_jsonl}...")
        clean, report = validate_and_dedup(cont_jsonl, args.threshold)
        reports.append(report)

        out_path = cont_jsonl
        with open(out_path, 'w', encoding='utf-8') as f:
            for record in clean:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  {report['total_input']} → {report['final_count']} records")
    else:
        print(f"WARNING: {cont_jsonl} not found, skipping", file=sys.stderr)

    # Process instruction data
    inst_jsonl = Path(args.instructions_dir) / 'instructions.jsonl'
    if inst_jsonl.exists():
        print(f"Validating {inst_jsonl}...")
        clean, report = validate_and_dedup(inst_jsonl, args.threshold)
        reports.append(report)

        out_path = inst_jsonl
        with open(out_path, 'w', encoding='utf-8') as f:
            for record in clean:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"  {report['total_input']} → {report['final_count']} records")
    else:
        print(f"WARNING: {inst_jsonl} not found, skipping", file=sys.stderr)

    # Write report
    lines = []
    lines.append("=" * 60)
    lines.append("VALIDATION REPORT")
    lines.append("=" * 60)

    for r in reports:
        lines.append(f"\nFile: {r['input_file']}")
        lines.append(f"  Total input records: {r['total_input']}")
        lines.append(f"  Parse errors: {r['parse_errors']}")
        lines.append(f"  Empty fields removed: {r['empty_fields']}")
        lines.append(f"  Too short (<50 words): {r['too_short']}")
        lines.append(f"  Too long (>3000 words): {r['too_long']}")
        lines.append(f"  Encoding issues: {r['encoding_issues']}")
        lines.append(f"  Duplicates removed: {r['duplicates_removed']}")
        lines.append(f"  Final count: {r['final_count']}")

    total_removed = sum(
        r['parse_errors'] + r['empty_fields'] + r['too_short'] +
        r['too_long'] + r['encoding_issues'] + r['duplicates_removed']
        for r in reports
    )
    total_final = sum(r['final_count'] for r in reports)
    lines.append(f"\n{'─' * 60}")
    lines.append(f"Total removed: {total_removed}")
    lines.append(f"Total final records: {total_final}")

    report_text = '\n'.join(lines) + '\n'
    Path(args.report).write_text(report_text, encoding='utf-8')
    print(f"\nReport written to {args.report}")


if __name__ == '__main__':
    main()
