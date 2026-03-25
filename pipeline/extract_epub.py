#!/usr/bin/env python3
"""Extract clean text from EPUB files."""

import re
import sys
from pathlib import Path

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


def extract_epub(epub_path: str, output_path: str):
    book = epub.read_epub(epub_path, options={"ignore_ncx": True})
    title = book.get_metadata('DC', 'title')
    title_str = title[0][0] if title else Path(epub_path).stem

    texts = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), 'html.parser')

        # Remove scripts, styles, notes
        for tag in soup.find_all(['script', 'style', 'aside']):
            tag.decompose()

        text = soup.get_text(separator='\n')
        # Clean up
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        if text.strip() and len(text.strip().split()) > 20:
            texts.append(text.strip())

    full_text = '\n\n'.join(texts)

    # Remove publisher boilerplate at end
    for marker in ['© ', 'ISBN', 'Издательство', 'Все права']:
        idx = full_text.rfind(marker)
        if idx > len(full_text) * 0.95:
            full_text = full_text[:idx].rstrip()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    word_count = len(full_text.split())
    print(f"  {Path(output_path).name}: {word_count:,} words")
    return word_count


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epub-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    epub_dir = Path(args.epub_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for ep in sorted(epub_dir.glob('*.epub')):
        # Clean filename for output
        name = re.sub(r'Гарсиа Маркес-\d{4}-', '', ep.stem)
        name = re.sub(r'\s*\([^)]*\)\s*', '', name)
        name = re.sub(r'\.-новый перевод$', '', name)
        name = re.sub(r'\.$', '', name).strip()
        out = output_dir / f"{name}.txt"
        try:
            wc = extract_epub(str(ep), str(out))
            total += wc
        except Exception as e:
            print(f"  ERROR {ep.name}: {e}", file=sys.stderr)

    print(f"\nTotal: {total:,} words")
