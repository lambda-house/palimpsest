#!/usr/bin/env python3
"""Orchestrator: runs steps 1-6 in sequence.

Usage:
    python run_all.py                    # full pipeline, skip API
    python run_all.py --with-api         # include Anthropic API calls
    python run_all.py --fb2-dir ./my_books --chunk-size 4096
"""

import argparse
import subprocess
import sys
import time


def run_step(name: str, cmd: list[str]) -> bool:
    """Run a pipeline step, returning True on success."""
    print(f"\n{'=' * 60}")
    print(f"STEP: {name}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}\n")

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nFAILED: {name} (exit code {result.returncode})", file=sys.stderr)
        return False

    print(f"\n✓ {name} completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description='Run the full Pelevin corpus pipeline')
    parser.add_argument('--fb2-dir', default='./corpus/fb2',
                        help='Directory containing FB2 files')
    parser.add_argument('--chunk-size', type=int, default=2048,
                        help='Chunk size in tokens for continuation format')
    parser.add_argument('--overlap', type=int, default=256,
                        help='Overlap in tokens between chunks')
    parser.add_argument('--with-api', action='store_true',
                        help='Enable Anthropic API calls for instruction generation')
    parser.add_argument('--eval-ratio', type=float, default=0.05,
                        help='Eval split ratio')
    args = parser.parse_args()

    python = sys.executable
    total_start = time.time()

    steps = [
        ("1. Extract text from FB2", [
            python, 'extract.py',
            '--fb2-dir', args.fb2_dir,
        ]),
        ("2. Analyze corpus", [
            python, 'analyze.py',
        ]),
        ("3. Format continuation data", [
            python, 'format_continuation.py',
            '--chunk-size', str(args.chunk_size),
            '--overlap', str(args.overlap),
        ]),
        ("4. Format instruction data", [
            python, 'format_instructions.py',
        ] + ([] if args.with_api else ['--skip-api'])),
        ("5. Validate & deduplicate", [
            python, 'validate.py',
        ]),
        ("6. Train/eval split", [
            python, 'split.py',
            '--eval-ratio', str(args.eval_ratio),
        ]),
    ]

    for name, cmd in steps:
        if not run_step(name, cmd):
            print(f"\nPipeline halted at: {name}", file=sys.stderr)
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {total_elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"\nOutput files:")
    print(f"  Continuation: ./training_data/final/continuation/{{train,eval}}.jsonl")
    print(f"  Instructions: ./training_data/final/instructions/{{train,eval}}.jsonl")
    print(f"  Reports: ./corpus_report.txt, ./validation_report.txt")


if __name__ == '__main__':
    main()
