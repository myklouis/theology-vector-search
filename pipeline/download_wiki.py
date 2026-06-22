"""
download_wiki.py
Download curated theology Wikipedia articles as plain text using the Wikipedia API.

Usage:
    python pipeline/download_wiki.py --out-dir data/wiki_pages
    python pipeline/download_wiki.py --out-dir data/wiki_pages --titles config/wiki_titles.txt

Fetches article plaintext via the Wikipedia extracts API. No scraping — this
uses the official MediaWiki API, which is stable and does not require an API key.
"""

import argparse
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
THROTTLE_SEC = 0.5
USER_AGENT = "theology-vector-search/1.0 (public domain corpus builder; github.com)"
DEFAULT_TITLES = Path(__file__).parent.parent / "config" / "wiki_titles.txt"


def load_titles(path: Path) -> list[str]:
    titles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            titles.append(line)
    return titles


def fetch_extract(title: str) -> tuple[str, str]:
    """Return (resolved_title, plaintext) for a Wikipedia article title."""
    params = urllib.parse.urlencode({
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "explaintext": "1",
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
    })
    req = urllib.request.Request(
        f"{API}?{params}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())

    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return title, ""
    page = pages[0]
    if page.get("missing"):
        return title, ""
    resolved = page.get("title", title)
    text = page.get("extract", "")
    return resolved, text


def safe_filename(title: str) -> str:
    """Convert article title to a safe filename."""
    safe = title.replace("/", "-").replace("\\", "-").replace(":", " -")
    for ch in '<>"|?*':
        safe = safe.replace(ch, "")
    return safe[:200] + ".txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wikipedia theology articles")
    parser.add_argument("--out-dir", default="data/wiki_pages", help="Output directory")
    parser.add_argument(
        "--titles",
        default=str(DEFAULT_TITLES),
        help="File with one article title per line (default: config/wiki_titles.txt)",
    )
    parser.add_argument("--throttle", type=float, default=THROTTLE_SEC, help="Seconds between requests")
    parser.add_argument("--resume", action="store_true", help="Skip files that already exist")
    args = parser.parse_args()

    titles_path = Path(args.titles)
    if not titles_path.exists():
        print(f"Titles file not found: {titles_path}")
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    titles = load_titles(titles_path)
    print(f"Fetching {len(titles)} Wikipedia articles → {out}")

    downloaded = skipped = errors = missing = 0
    for i, title in enumerate(titles):
        filename = safe_filename(title)
        dest = out / filename
        if args.resume and dest.exists():
            skipped += 1
            continue
        try:
            resolved, text = fetch_extract(title)
            if not text.strip():
                print(f"  MISSING: {title!r}")
                missing += 1
            else:
                header = f"{resolved}\n\n" if resolved != title else ""
                dest.write_text(header + text, encoding="utf-8")
                downloaded += 1
                if (i + 1) % 50 == 0 or (i + 1) == len(titles):
                    print(f"  [{i+1}/{len(titles)}] {resolved}")
        except Exception as e:
            print(f"  ERROR {title!r}: {e}")
            errors += 1
        time.sleep(args.throttle)

    print(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}, Missing: {missing}, Errors: {errors}")
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()
