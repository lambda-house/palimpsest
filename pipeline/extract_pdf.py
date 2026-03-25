#!/usr/bin/env python3
"""Extract clean text from Pelevin PDF files for the training corpus."""

import sys
import re
import fitz  # PyMuPDF


def extract_pdf(pdf_path: str, output_path: str, skip_pages: int = 0,
                running_headers: list[str] | None = None,
                skip_until: str | None = None):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Processing: {pdf_path} ({total_pages} pages, skipping first {skip_pages})")

    lines = []
    for page_num in range(skip_pages, total_pages):
        page = doc[page_num]
        text = page.get_text("text")

        # Remove page headers like "В. О. Пелевин. «Путешествие в Элевсин»"
        text = re.sub(r'^В\.\s*О\.\s*Пелевин[^\n]*\n', '', text)
        # Remove running headers (book title appearing on pages)
        if running_headers:
            for header in running_headers:
                text = re.sub(r'^\s*' + re.escape(header) + r'\s*$', '', text, flags=re.MULTILINE)
        # Remove standalone page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        # Remove trailing whitespace per line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        if text.strip():
            lines.append(text.strip())

    full_text = '\n\n'.join(lines)

    # Skip front matter (copyright, disclaimers) up to a marker
    if skip_until:
        idx = full_text.find(skip_until)
        if idx >= 0:
            full_text = full_text[idx:]

    # Clean up common PDF artifacts
    # Fix hyphenation at line breaks (Russian word broken across lines)
    full_text = re.sub(r'(\w)-\n(\w)', r'\1\2', full_text)
    # Normalize multiple blank lines
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)

    # Remove publisher boilerplate at the end
    # Find the earliest boilerplate marker in the last 5% of text
    cutoff = int(len(full_text) * 0.95)
    earliest_marker = len(full_text)
    for marker in ['© Пелевин', '© Эксмо', '© В.О. Пелевин', '© В. О. Пелевин',
                    '© Оформление', 'ISBN ', 'УДК ', 'ББК ',
                    'Обособленное подразделение', 'Интернет-магазин',
                    'Розничная продажа', 'www.eksmo.ru', 'www.chitai-gorod',
                    'Все права защищены. Книга или любая ее часть',
                    'Литературно-художественное издание',
                    'Страна происхождения',
                    '\nОглавление\n', '\nСодержание\n']:
        idx = full_text.find(marker, cutoff)
        if idx >= 0 and idx < earliest_marker:
            earliest_marker = idx
    if earliest_marker < len(full_text):
        full_text = full_text[:earliest_marker].rstrip()

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    word_count = len(full_text.split())
    print(f"  Output: {output_path}")
    print(f"  Words: {word_count:,}")
    print(f"  Characters: {len(full_text):,}")


if __name__ == '__main__':
    # Путешествие в Элевсин - skip front matter, start at the epigraph
    extract_pdf(
        '/Users/timur/Downloads/Pelevin_V._Edinstvennyiyi._Puteshestvie_V_Yelevsin.a4.pdf',
        '/Users/timur/work/pelevin/corpus/clean/Путешествие в Элевсин.txt',
        skip_pages=5,
        skip_until='Mirabile futurum',
    )

    # Круть - skip front matter, start at the epigraph; remove running headers
    extract_pdf(
        '/Users/timur/Downloads/Pelevin_V._Vselennayatran4._Krut.a4.pdf',
        '/Users/timur/work/pelevin/corpus/clean/Круть.txt',
        skip_pages=5,
        running_headers=['КРУТЬ'],
        skip_until='I got two strong arms',
    )
