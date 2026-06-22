"""
download_gutenberg.py
Download theology and religion books from Project Gutenberg using the
Gutendex API (gutendex.com — free, unofficial, no key required).

Usage:
    # Download all English books tagged with theology/religion subjects:
    python pipeline/download_gutenberg.py --out-dir data/gutenberg_pages

    # Limit to first N books (useful for a quick test):
    python pipeline/download_gutenberg.py --out-dir data/gutenberg_pages --limit 500

    # Download specific book IDs (one per line in a text file):
    python pipeline/download_gutenberg.py --out-dir data/gutenberg_pages --ids config/gutenberg_ids.txt

    # Add extra subjects beyond the defaults:
    python pipeline/download_gutenberg.py --out-dir data/gutenberg_pages --subjects "mysticism" "eschatology"

Full theology corpus (77k+ books) takes several hours and ~25 GB of disk.
Use --limit 1000 for a representative sample.
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

GUTENDEX = "https://gutendex.com/books/"
THROTTLE_SEC = 1.0
USER_AGENT = "theology-vector-search/1.0 (public domain corpus builder; github.com)"

DEFAULT_SUBJECTS = [
    "religion",
    "theology",
    "bible",
    "church history",
    "christian life",
    "philosophy",
    "ethics",
    "mysticism",
    "religious literature",
    "devotional literature",
    "sermons",
    "prayer",
]


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def _best_text_url(formats: dict) -> str | None:
    preferred = [
        "text/plain; charset=utf-8",
        "text/plain; charset=us-ascii",
        "text/plain",
    ]
    for fmt in preferred:
        if fmt in formats:
            return formats[fmt]
    for key, url in formats.items():
        if "text/plain" in key:
            return url
    return None


def _safe_filename(book_id: int, title: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*]', "", title)
    safe = re.sub(r"\s+", "_", safe.strip())[:80]
    return f"{book_id}_{safe}.txt"


def fetch_books_by_subject(subjects: list[str], limit: int | None) -> list[dict]:
    """Return book metadata from gutendex for given subjects, in English."""
    seen_ids: set[int] = set()
    books: list[dict] = []

    for subject in subjects:
        params = urllib.parse.urlencode({"topic": subject, "languages": "en"})
        url = f"{GUTENDEX}?{params}"
        page = 1
        while url:
            try:
                data = _get(url)
            except Exception as e:
                print(f"  WARNING: failed fetching page {page} for '{subject}': {e}")
                break

            for book in data.get("results", []):
                bid = book["id"]
                if bid not in seen_ids:
                    seen_ids.add(bid)
                    books.append(book)
                    if limit and len(books) >= limit:
                        return books

            url = data.get("next")
            page += 1
            time.sleep(THROTTLE_SEC)

        print(f"  Subject '{subject}': {sum(1 for b in books if subject in str(b.get('subjects', '')).lower()):,} total so far")

    return books


def fetch_books_by_ids(ids_path: Path) -> list[dict]:
    """Return book metadata for specific Gutenberg IDs."""
    book_ids = [int(line.strip()) for line in ids_path.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
    print(f"Fetching metadata for {len(book_ids)} specific book IDs...")
    books = []
    for i, bid in enumerate(book_ids):
        try:
            data = _get(f"{GUTENDEX}{bid}/")
            books.append(data)
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(book_ids)}] metadata loaded")
        except Exception as e:
            print(f"  WARNING: book {bid} metadata failed: {e}")
        time.sleep(THROTTLE_SEC * 0.5)
    return books


def download_book(book: dict, out: Path, resume: bool) -> bool:
    bid = book["id"]
    title = book.get("title", f"book_{bid}")
    filename = _safe_filename(bid, title)
    dest = out / filename

    if resume and dest.exists() and dest.stat().st_size > 100:
        return False  # skipped

    formats = book.get("formats", {})
    url = _best_text_url(formats)
    if not url:
        return False  # no plain text available

    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8", errors="replace")

        # Strip Gutenberg boilerplate header/footer
        start = re.search(r"\*{3} ?START OF (THIS|THE) PROJECT GUTENBERG", text, re.I)
        end = re.search(r"\*{3} ?END OF (THIS|THE) PROJECT GUTENBERG", text, re.I)
        if start:
            text = text[start.end():]
        if end:
            text = text[:end.start()]

        dest.write_text(f"{title}\n\n{text.strip()}", encoding="utf-8")
        return True
    except Exception as e:
        print(f"  ERROR downloading {bid} '{title}': {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download theology books from Project Gutenberg")
    parser.add_argument("--out-dir", default="data/gutenberg_pages", help="Output directory for .txt files")
    parser.add_argument("--subjects", nargs="*", default=None, help="Subject topics to search (default: theology/religion set)")
    parser.add_argument("--ids", help="Path to a file of specific Gutenberg book IDs (one per line)")
    parser.add_argument("--limit", type=int, default=None, help="Max books to download (useful for testing)")
    parser.add_argument("--throttle", type=float, default=THROTTLE_SEC, help="Seconds between requests (default 1.0)")
    parser.add_argument("--resume", action="store_true", help="Skip files that already exist")
    args = parser.parse_args()

    global THROTTLE_SEC
    THROTTLE_SEC = args.throttle

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.ids:
        books = fetch_books_by_ids(Path(args.ids))
    else:
        subjects = args.subjects if args.subjects else DEFAULT_SUBJECTS
        print(f"Searching Gutenberg for subjects: {subjects}")
        books = fetch_books_by_subject(subjects, args.limit)

    print(f"\nFound {len(books)} books. Downloading text files → {out}")

    downloaded = skipped = no_text = 0
    for i, book in enumerate(books):
        result = download_book(book, out, args.resume)
        if result is False:
            formats = book.get("formats", {})
            if _best_text_url(formats):
                skipped += 1
            else:
                no_text += 1
        else:
            downloaded += 1

        if (i + 1) % 100 == 0 or (i + 1) == len(books):
            print(f"  [{i+1}/{len(books)}] downloaded={downloaded} skipped={skipped} no_text={no_text}")

        time.sleep(THROTTLE_SEC)

    print(f"\nDone. Downloaded: {downloaded}, Skipped (resume): {skipped}, No plain text: {no_text}")
    print(f"Output: {out.resolve()}")
    print(f"\nNext step: run build_index.py --source-dirs {out}")


if __name__ == "__main__":
    main()
