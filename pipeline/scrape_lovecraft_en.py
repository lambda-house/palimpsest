#!/usr/bin/env python3
"""Scrape Lovecraft's works from hplovecraft.com"""

import re
import time
from pathlib import Path
from urllib.request import urlopen, Request
from html.parser import HTMLParser


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'nav', 'header', 'footer'):
            self.skip = True
        if tag in ('p', 'br', 'div', 'h1', 'h2', 'h3'):
            self.text.append('\n')

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'nav', 'header', 'footer'):
            self.skip = False
        if tag == 'p':
            self.text.append('\n')

    def handle_data(self, data):
        if not self.skip:
            self.text.append(data)

    def get_text(self):
        return ''.join(self.text)


def fetch(url):
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode('utf-8', errors='replace')


# Major Lovecraft works with their URLs on hplovecraft.com
WORKS = {
    "At the Mountains of Madness": "fiction/mountain",
    "The Call of Cthulhu": "fiction/cc",
    "The Shadow over Innsmouth": "fiction/innsmouth",
    "The Dunwich Horror": "fiction/dunwich",
    "The Colour Out of Space": "fiction/colour",
    "The Whisperer in Darkness": "fiction/whisperer",
    "The Case of Charles Dexter Ward": "fiction/cdw",
    "The Dream-Quest of Unknown Kadath": "fiction/dq",
    "The Shadow Out of Time": "fiction/sot",
    "The Thing on the Doorstep": "fiction/doorstep",
    "The Haunter of the Dark": "fiction/haunt",
    "The Rats in the Walls": "fiction/rats",
    "Pickman's Model": "fiction/pickman",
    "The Music of Erich Zann": "fiction/zann",
    "The Outsider": "fiction/outsider",
    "Dagon": "fiction/dagon",
    "Herbert West-Reanimator": "fiction/herbert",
    "The Festival": "fiction/festival",
    "The Lurking Fear": "fiction/lurking",
    "The Horror at Red Hook": "fiction/redhook",
    "Cool Air": "fiction/cool",
    "He": "fiction/he",
    "In the Vault": "fiction/vault",
    "The Terrible Old Man": "fiction/oldman",
    "The Cats of Ulthar": "fiction/cats",
    "The Statement of Randolph Carter": "fiction/rc",
    "From Beyond": "fiction/beyond",
    "The Temple": "fiction/temple",
    "The Tomb": "fiction/tomb",
    "The Silver Key": "fiction/silver",
    "Through the Gates of the Silver Key": "fiction/gates",
    "The Dreams in the Witch House": "fiction/witch",
    "The Shunned House": "fiction/shunned",
}

BASE_URL = "https://www.hplovecraft.com/writings/texts/"


def main():
    out = Path("./corpus/clean/lovecraft_en")
    out.mkdir(parents=True, exist_ok=True)

    total = 0
    for title, path in WORKS.items():
        try:
            html = fetch(f"{BASE_URL}{path}")
            parser = TextExtractor()
            parser.feed(html)
            text = parser.get_text()

            # Clean up
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = text.strip()

            wc = len(text.split())
            if wc < 500:
                print(f"  SKIP: {title} ({wc} words)")
                continue

            fname = re.sub(r"[^\w\s-]", "", title).strip() + ".txt"
            (out / fname).write_text(text, encoding='utf-8')
            total += wc
            print(f"  {title}: {wc:,} words")
            time.sleep(1)
        except Exception as e:
            print(f"  FAIL: {title}: {e}")

    print(f"\nTotal: {total:,} words")


if __name__ == '__main__':
    main()
