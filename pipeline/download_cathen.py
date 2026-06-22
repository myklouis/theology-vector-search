"""
download_cathen.py
Download the Catholic Encyclopedia (1907-1913, public domain) from newadvent.org
and save each article as a plain-text .txt file.

Usage:
    python pipeline/download_cathen.py --out-dir data/cathen_pages

The URL structure on newadvent.org maps directly to the output filenames:
    https://www.newadvent.org/cathen/01001a.htm  →  cathen_01001a.htm.txt

Fetches ~3,700 articles. Be polite — requests are throttled to 1/second by default.
"""

import argparse
import re
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser
from pathlib import Path

INDEX_BASE = "https://www.newadvent.org/cathen/"
ARTICLE_BASE = "https://www.newadvent.org/cathen/"
THROTTLE_SEC = 1.0
USER_AGENT = "theology-vector-search/1.0 (public domain corpus builder; github.com)"


class ArticleExtractor(HTMLParser):
    """Extract article body text from Catholic Encyclopedia HTML."""

    def __init__(self):
        super().__init__()
        self.in_body = False
        self.depth = 0
        self.chunks: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        cls = attrs_d.get("class", "")
        if tag == "div" and "articlebody" in cls:
            self.in_body = True
            self.depth = 1
        elif self.in_body and tag == "div":
            self.depth += 1
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "header", "footer"):
            self._skip = False
        if self.in_body and tag == "div":
            self.depth -= 1
            if self.depth <= 0:
                self.in_body = False

    def handle_data(self, data):
        if self.in_body and not self._skip:
            self.chunks.append(data)

    def text(self) -> str:
        return " ".join(self.chunks)


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def _strip_boilerplate(text: str) -> str:
    text = re.sub(r"Please help support the mission of New Advent.*?only \$\d+\.\d+\.\.\.", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _article_ids_from_index(letter_html: str) -> list[str]:
    """Parse article IDs like '01001a' from an index page."""
    return re.findall(r'href="(\d{5}[a-z]?)\.htm"', letter_html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Catholic Encyclopedia from newadvent.org")
    parser.add_argument("--out-dir", default="data/cathen_pages", help="Output directory for .txt files")
    parser.add_argument("--throttle", type=float, default=THROTTLE_SEC, help="Seconds between requests (default 1.0)")
    parser.add_argument("--resume", action="store_true", help="Skip files that already exist")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Fetching article index from newadvent.org/cathen/...")
    article_ids: list[str] = []
    for letter in "abcdefghijklmnopqrstuvwxyz":
        try:
            html = _fetch(f"{INDEX_BASE}{letter}.htm")
            ids = _article_ids_from_index(html)
            article_ids.extend(ids)
            print(f"  [{letter.upper()}] {len(ids)} articles")
            time.sleep(args.throttle)
        except Exception as e:
            print(f"  [{letter.upper()}] index fetch failed: {e}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_ids = [x for x in article_ids if not (x in seen or seen.add(x))]
    print(f"\nTotal articles found: {len(unique_ids)}")

    downloaded = skipped = errors = 0
    for i, article_id in enumerate(unique_ids):
        filename = f"cathen_{article_id}.htm.txt"
        dest = out / filename
        if args.resume and dest.exists():
            skipped += 1
            continue

        url = f"{ARTICLE_BASE}{article_id}.htm"
        try:
            html = _fetch(url)

            parser_obj = ArticleExtractor()
            parser_obj.feed(html)
            text = parser_obj.text()

            if not text.strip():
                title_match = re.search(r"<title>(.*?)</title>", html, re.I)
                title = title_match.group(1) if title_match else article_id
                body_match = re.search(r"<body[^>]*>(.*?)</body>", html, re.I | re.S)
                if body_match:
                    raw = re.sub(r"<[^>]+>", " ", body_match.group(1))
                    text = re.sub(r"\s+", " ", raw).strip()
                else:
                    text = title

            text = _strip_boilerplate(text)
            dest.write_text(text, encoding="utf-8")
            downloaded += 1

            if (i + 1) % 100 == 0 or (i + 1) == len(unique_ids):
                print(f"  [{i+1}/{len(unique_ids)}] {filename}")
        except Exception as e:
            print(f"  ERROR {article_id}: {e}")
            errors += 1

        time.sleep(args.throttle)

    print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}, Errors: {errors}")
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()
