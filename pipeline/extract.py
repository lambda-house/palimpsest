#!/usr/bin/env python3
"""Step 1: FB2 Parser & Text Extractor.

Parses FB2 files, extracts body text preserving formatting,
outputs clean .txt files and manifest.json.
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tqdm import tqdm

FB2_NS = "{http://www.gribuser.ru/xml/fictionbook/2.0}"


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.strip('. ')
    return name[:200] if name else "untitled"


def extract_text_recursive(element, ns: str) -> str:
    """Recursively extract text from an FB2 element, preserving structure."""
    tag = element.tag.replace(ns, '')
    parts = []

    if tag == 'title':
        lines = []
        for p in element.findall(f'{ns}p'):
            t = get_all_text(p, ns).strip()
            if t:
                lines.append(t)
        if lines:
            parts.append('\n'.join(lines))
            parts.append('')  # blank line after title
        return '\n'.join(parts)

    if tag == 'epigraph':
        lines = []
        for child in element:
            ctag = child.tag.replace(ns, '')
            if ctag == 'p':
                t = get_all_text(child, ns).strip()
                if t:
                    lines.append(t)
            elif ctag == 'text-author':
                t = get_all_text(child, ns).strip()
                if t:
                    lines.append(f"— {t}")
            elif ctag == 'poem':
                lines.append(extract_poem(child, ns))
        if lines:
            parts.append('\n'.join(lines))
            parts.append('')
        return '\n'.join(parts)

    if tag == 'poem':
        return extract_poem(element, ns) + '\n'

    if tag == 'p':
        t = get_all_text(element, ns).strip()
        if t:
            return t + '\n\n'
        return ''

    if tag == 'empty-line':
        return '\n'

    if tag == 'subtitle':
        t = get_all_text(element, ns).strip()
        if t:
            return f"\n{t}\n\n"
        return ''

    if tag == 'section':
        section_parts = []
        for child in element:
            section_parts.append(extract_text_recursive(child, ns))
        return ''.join(section_parts)

    if tag == 'cite':
        lines = []
        for child in element:
            ctag = child.tag.replace(ns, '')
            if ctag == 'p':
                t = get_all_text(child, ns).strip()
                if t:
                    lines.append(t)
            elif ctag == 'text-author':
                t = get_all_text(child, ns).strip()
                if t:
                    lines.append(f"— {t}")
            elif ctag == 'poem':
                lines.append(extract_poem(child, ns))
        if lines:
            return '\n'.join(lines) + '\n\n'
        return ''

    if tag == 'table':
        return ''

    if tag == 'image':
        return ''

    # Fallback: recurse into children
    for child in element:
        parts.append(extract_text_recursive(child, ns))
    return ''.join(parts)


def extract_poem(element, ns: str) -> str:
    """Extract poem text preserving verse formatting."""
    lines = []
    for child in element:
        ctag = child.tag.replace(ns, '')
        if ctag == 'title':
            for p in child.findall(f'{ns}p'):
                t = get_all_text(p, ns).strip()
                if t:
                    lines.append(t)
        elif ctag == 'stanza':
            for v in child.findall(f'{ns}v'):
                t = get_all_text(v, ns).strip()
                lines.append(t)
            lines.append('')  # blank line between stanzas
        elif ctag == 'text-author':
            t = get_all_text(child, ns).strip()
            if t:
                lines.append(f"— {t}")
    return '\n'.join(lines)


def get_all_text(element, ns: str) -> str:
    """Get all text from an element, stripping inline tags but keeping their text."""
    parts = []
    if element.text:
        parts.append(element.text)
    for child in element:
        ctag = child.tag.replace(ns, '')
        if ctag in ('emphasis', 'strong', 'strikethrough', 'sub', 'sup', 'code'):
            if child.text:
                parts.append(child.text)
        elif ctag == 'a':
            # Skip footnote links but keep inline text if any
            pass
        else:
            parts.append(get_all_text(child, ns))
        if child.tail:
            parts.append(child.tail)
    return ''.join(parts)


def detect_namespace(root) -> str:
    """Detect the FB2 namespace from the root element."""
    if '}' in root.tag:
        return root.tag.split('}')[0] + '}'
    return ''


def get_metadata(root, ns: str) -> dict:
    """Extract metadata from FB2 description."""
    meta = {'title': 'untitled', 'author': 'unknown', 'year': ''}
    ti = root.find(f'{ns}description/{ns}title-info')
    if ti is None:
        return meta

    book_title = ti.find(f'{ns}book-title')
    if book_title is not None and book_title.text:
        meta['title'] = book_title.text.strip()

    author = ti.find(f'{ns}author')
    if author is not None:
        fn = author.find(f'{ns}first-name')
        ln = author.find(f'{ns}last-name')
        parts = []
        if fn is not None and fn.text:
            parts.append(fn.text.strip())
        if ln is not None and ln.text:
            parts.append(ln.text.strip())
        if parts:
            meta['author'] = ' '.join(parts)

    date = ti.find(f'{ns}date')
    if date is not None and date.text:
        meta['year'] = date.text.strip()

    return meta


def extract_fb2(filepath: Path) -> tuple[dict, str]:
    """Extract text and metadata from a single FB2 file.

    Returns (metadata_dict, extracted_text).
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    ns = detect_namespace(root)

    meta = get_metadata(root, ns)

    # Try to extract year from filename if not in metadata
    if not meta['year']:
        m = re.match(r'(\d{4})\s', filepath.stem)
        if m:
            meta['year'] = m.group(1)

    # Find main body (skip notes body)
    text_parts = []
    bodies = root.findall(f'{ns}body')
    for body in bodies:
        # Skip notes body
        if body.get('name') == 'notes':
            continue

        # Process top-level sections
        first_section = True
        for child in body:
            ctag = child.tag.replace(ns, '')
            if ctag == 'section':
                if not first_section:
                    text_parts.append('\n---CHAPTER---\n\n')
                first_section = False
                text_parts.append(extract_text_recursive(child, ns))
            elif ctag in ('title', 'epigraph'):
                text_parts.append(extract_text_recursive(child, ns))

    text = ''.join(text_parts)

    # Clean up excessive whitespace but preserve intentional formatting
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = text.strip()

    return meta, text


def main():
    parser = argparse.ArgumentParser(description='Extract text from FB2 files')
    parser.add_argument('--fb2-dir', default='./corpus/fb2',
                        help='Directory containing FB2 files (searched recursively)')
    parser.add_argument('--output-dir', default='./corpus/clean',
                        help='Output directory for clean text files')
    parser.add_argument('--manifest', default='./manifest.json',
                        help='Path for manifest JSON output')
    args = parser.parse_args()

    fb2_dir = Path(args.fb2_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fb2_files = sorted(fb2_dir.rglob('*.fb2'))
    if not fb2_files:
        print(f"ERROR: No .fb2 files found in {fb2_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(fb2_files)} FB2 files")

    manifest = []
    errors = []

    for fp in tqdm(fb2_files, desc="Extracting"):
        try:
            meta, text = extract_fb2(fp)
        except Exception as e:
            msg = f"WARNING: Failed to parse {fp.name}: {e}"
            print(msg, file=sys.stderr)
            errors.append(msg)
            continue

        if not text.strip():
            msg = f"WARNING: Empty text extracted from {fp.name}"
            print(msg, file=sys.stderr)
            errors.append(msg)
            continue

        # Build output filename
        out_name = sanitize_filename(meta['title']) + '.txt'
        out_path = output_dir / out_name

        # Handle duplicate filenames
        if out_path.exists():
            base = sanitize_filename(meta['title'])
            i = 2
            while (output_dir / f"{base}_{i}.txt").exists():
                i += 1
            out_name = f"{base}_{i}.txt"
            out_path = output_dir / out_name

        out_path.write_text(text, encoding='utf-8')

        word_count = len(text.split())
        char_count = len(text)
        token_estimate = int(word_count * 2.0)

        manifest.append({
            'filename': out_name,
            'source_file': str(fp),
            'title': meta['title'],
            'author': meta['author'],
            'year': meta['year'],
            'word_count': word_count,
            'char_count': char_count,
            'token_count_estimate': token_estimate,
        })

    # Write manifest
    manifest_path = Path(args.manifest)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    total_words = sum(m['word_count'] for m in manifest)
    total_tokens = sum(m['token_count_estimate'] for m in manifest)
    print(f"\nExtracted {len(manifest)} files")
    print(f"Total words: {total_words:,}")
    print(f"Estimated tokens: {total_tokens:,}")
    if errors:
        print(f"Errors: {len(errors)}")


if __name__ == '__main__':
    main()
