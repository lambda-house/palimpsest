#!/usr/bin/env python3
"""Scrape English author texts from Project Gutenberg."""

import re
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request

# Gutenberg book IDs for curated works
AUTHORS = {
    "lovecraft_en": {
        "label": "H.P. Lovecraft",
        "books": {
            # Lovecraft's works are mostly NOT on Gutenberg (entered public domain recently)
            # Use hplovecraft.com or Wikisource instead
        },
        "alt_url": "https://www.hplovecraft.com/writings/texts/",
    },
    "doyle_en": {
        "label": "Arthur Conan Doyle",
        "books": {
            "A Study in Scarlet": 244,
            "The Sign of the Four": 2097,
            "Adventures of Sherlock Holmes": 1661,
            "Memoirs of Sherlock Holmes": 834,
            "The Hound of the Baskervilles": 2852,
            "The Return of Sherlock Holmes": 108,
            "The Valley of Fear": 3289,
            "His Last Bow": 2350,
            "The Case-Book of Sherlock Holmes": 69700,
            "The Lost World": 139,
        },
    },
    "poe_en": {
        "label": "Edgar Allan Poe",
        "books": {
            "Works of Edgar Allan Poe Vol 1": 2147,
            "Works of Edgar Allan Poe Vol 2": 2148,
            "Works of Edgar Allan Poe Vol 3": 2149,
            "Works of Edgar Allan Poe Vol 4": 2150,
            "Works of Edgar Allan Poe Vol 5": 2151,
            "The Narrative of Arthur Gordon Pym": 51060,
        },
    },
    "wilde_en": {
        "label": "Oscar Wilde",
        "books": {
            "The Picture of Dorian Gray": 174,
            "The Importance of Being Earnest": 844,
            "An Ideal Husband": 885,
            "Lady Windermere's Fan": 790,
            "A Woman of No Importance": 854,
            "Salome": 42704,
            "The Happy Prince and Other Tales": 902,
            "Lord Arthur Savile's Crime": 773,
            "De Profundis": 921,
            "The Soul of Man Under Socialism": 1017,
            "A House of Pomegranates": 873,
        },
    },
    "london_en": {
        "label": "Jack London",
        "books": {
            "The Call of the Wild": 215,
            "White Fang": 910,
            "The Sea-Wolf": 1074,
            "Martin Eden": 1056,
            "The Iron Heel": 1164,
            "Burning Daylight": 4689,
            "The Star Rover": 1083,
            "South Sea Tales": 2429,
            "John Barleycorn": 318,
            "The People of the Abyss": 1688,
        },
    },
}


def fetch_gutenberg(book_id):
    """Fetch plain text from Gutenberg mirrors."""
    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
    ]
    for url in urls:
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=30) as resp:
                raw = resp.read()
                # Try UTF-8 first, fall back to latin-1
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError:
                    return raw.decode("latin-1")
        except Exception:
            continue
    return None


def strip_gutenberg_header_footer(text):
    """Remove Project Gutenberg header and footer boilerplate."""
    # Find start of actual text
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "***START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    for marker in start_markers:
        idx = text.find(marker)
        if idx >= 0:
            # Skip to next line after marker
            nl = text.find("\n", idx)
            if nl >= 0:
                text = text[nl + 1:]
            break

    # Find end
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "***END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    for marker in end_markers:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
            break

    # Clean up
    text = re.sub(r"\r\n", "\n", text)
    # Unwrap hard-wrapped paragraphs (lines ending without period followed by lowercase)
    text = re.sub(r"(?<!\n)\n(?!\n)(?=[a-z])", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scrape_author(author_key, output_dir):
    author = AUTHORS[author_key]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {author['label']} ===")

    if not author.get("books"):
        print(f"  No Gutenberg IDs — needs manual download from {author.get('alt_url', 'other source')}")
        return 0

    total = 0
    for title, book_id in author["books"].items():
        text = fetch_gutenberg(book_id)
        if not text:
            print(f"  FAIL: {title} (id={book_id})")
            continue

        clean = strip_gutenberg_header_footer(text)
        wc = len(clean.split())

        if wc < 1000:
            print(f"  SKIP: {title} ({wc} words — too short)")
            continue

        fname = re.sub(r"[^\w\s-]", "", title).strip() + ".txt"
        (out / fname).write_text(clean, encoding="utf-8")
        total += wc
        print(f"  {title}: {wc:,} words")
        time.sleep(1)  # polite

    print(f"Total: {total:,} words")
    return total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--author", choices=list(AUTHORS.keys()) + ["all"], default="all")
    parser.add_argument("--output-base", default="./corpus/clean")
    args = parser.parse_args()

    to_scrape = AUTHORS if args.author == "all" else {args.author: AUTHORS[args.author]}
    grand_total = 0

    for key in to_scrape:
        total = scrape_author(key, f"{args.output_base}/{key}")
        grand_total += total

    print(f"\n=== GRAND TOTAL: {grand_total:,} words ===")
