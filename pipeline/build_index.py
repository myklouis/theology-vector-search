"""
build_index.py
Chunk text sources, embed via nomic-embed-text (Ollama), and upsert into Qdrant.

Usage:
    python pipeline/build_index.py \\
        --source-dirs data/cathen_pages data/wiki_pages \\
        --out-dir out \\
        --ollama-url http://localhost:11434 \\
        --qdrant-host localhost --qdrant-port 6333

Resumes safely on crash: checkpoint per source saves vectors as numpy so
re-runs skip already-embedded chunks rather than starting from zero.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

CHUNK_WORDS = 450
STRIDE = 380
EMBED_BATCH = 50
QDRANT_BATCH = 200
DIM = 768
MODEL = "nomic-embed-text"
COLLECTION = "theology_passages"


def log(msg: str, log_path: Path) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def chunk_text(text: str, meta: dict) -> list[dict]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        window = words[i : i + CHUNK_WORDS]
        chunk = {"text": " ".join(window), **meta}
        chunk["id"] = hashlib.md5(chunk["text"].encode()).hexdigest()
        chunks.append(chunk)
        if len(window) < CHUNK_WORDS:
            break
        i += STRIDE
    return chunks


def embed_batch(texts: list[str], ollama_url: str) -> list[list[float]]:
    prefixed = ["search_document: " + t for t in texts]
    body = json.dumps(
        {"model": MODEL, "input": prefixed, "keep_alive": "10m"}
    ).encode()
    req = urllib.request.Request(
        f"{ollama_url}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["embeddings"]


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def load_existing_ids(chunks_path: Path) -> set:
    ids: set = set()
    if not chunks_path.exists():
        return ids
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            try:
                ids.add(json.loads(line)["id"])
            except Exception:
                pass
    return ids


def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection '{COLLECTION}'")


def index_source(
    label: str,
    source_dir: Path,
    source_key: str,
    out_dir: Path,
    ollama_url: str,
    client: QdrantClient,
    existing_ids: set,
    log_path: Path,
) -> int:
    txt_files = sorted(source_dir.glob("*.txt"))
    log(f"=== {label} — {len(txt_files):,} files ===", log_path)
    if not txt_files:
        log(f"  No .txt files in {source_dir} — skipping.", log_path)
        return 0

    chunks_path = out_dir / "chunks.jsonl"
    offs_path = out_dir / "chunks.offsets.npy"

    log("  Chunking...", log_path)
    new_chunks = []
    for i, f in enumerate(txt_files):
        try:
            text = f.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        title = f.stem.replace("_", " ").strip()
        meta = {"title": title, "source": source_key, "file": f.name}
        for c in chunk_text(text, meta):
            if c["id"] not in existing_ids:
                existing_ids.add(c["id"])
                new_chunks.append(c)
        if (i + 1) % 1000 == 0:
            log(f"  [{i+1:,}/{len(txt_files):,}] {len(new_chunks):,} new chunks", log_path)

    log(f"  Chunking done: {len(new_chunks):,} new chunks", log_path)
    if not new_chunks:
        log("  Nothing new — skipping.", log_path)
        return 0

    # Embed with numpy checkpoint (avoids ~7x memory inflation from .tolist())
    ckpt_path = out_dir / f"embed_ckpt_{source_key}.npy"
    all_vecs: np.ndarray | None = None
    resume_from = 0
    if ckpt_path.exists():
        try:
            saved = np.load(str(ckpt_path))
            if saved.ndim == 2 and saved.shape[1] == DIM and saved.shape[0] <= len(new_chunks):
                all_vecs = saved
                resume_from = len(saved)
                log(f"  Resuming from checkpoint: {resume_from:,} vectors ({saved.nbytes // 1024 // 1024} MB)", log_path)
        except Exception:
            pass

    log(f"  Embedding {len(new_chunks):,} chunks via {MODEL} (from {resume_from:,})...", log_path)
    t0 = time.time()
    buf: list[np.ndarray] = []

    for i in range(resume_from, len(new_chunks), EMBED_BATCH):
        batch = new_chunks[i : i + EMBED_BATCH]
        for attempt in range(3):
            try:
                vecs = embed_batch([c["text"] for c in batch], ollama_url)
                buf.append(
                    np.array([normalize(np.array(v, dtype=np.float32)) for v in vecs], dtype=np.float32)
                )
                break
            except Exception as e:
                if attempt < 2:
                    log(f"  WARNING batch {i // EMBED_BATCH} attempt {attempt+1} failed: {e} — retrying", log_path)
                    time.sleep(5)
                else:
                    log(f"  WARNING batch {i // EMBED_BATCH} all retries failed — filling zeros", log_path)
                    buf.append(np.zeros((len(batch), DIM), dtype=np.float32))

        done = min(i + EMBED_BATCH, len(new_chunks))
        if done % 2000 == 0 or done >= len(new_chunks):
            new_np = np.vstack(buf) if buf else np.empty((0, DIM), dtype=np.float32)
            buf = []
            all_vecs = np.vstack([all_vecs, new_np]) if all_vecs is not None else new_np
            np.save(str(ckpt_path), all_vecs)
            elapsed = time.time() - t0
            total_done = len(all_vecs)
            rate = (total_done - resume_from) / elapsed if elapsed > 0 else 0
            eta = (len(new_chunks) - total_done) / rate / 60 if rate > 0 else 0
            log(f"  [{total_done:,}/{len(new_chunks):,}] {rate:.0f}/s ETA {eta:.1f}min", log_path)

    if buf:
        new_np = np.vstack(buf)
        all_vecs = np.vstack([all_vecs, new_np]) if all_vecs is not None else new_np

    log(f"  Embedding complete: {len(all_vecs):,} vectors", log_path)
    if ckpt_path.exists():
        ckpt_path.unlink()

    # Append to chunks.jsonl + update offsets
    log("  Appending to chunks.jsonl and offsets.npy...", log_path)
    offsets = np.load(str(offs_path)) if offs_path.exists() else np.array([], dtype=np.int64)
    file_size = chunks_path.stat().st_size if chunks_path.exists() else 0
    new_offsets = []
    with open(chunks_path, "ab") as fout:
        for c in new_chunks:
            new_offsets.append(file_size)
            line = (json.dumps(c, ensure_ascii=False) + "\n").encode("utf-8")
            fout.write(line)
            file_size += len(line)
    updated_offs = np.concatenate([offsets, np.array(new_offsets, dtype=np.int64)])
    np.save(str(offs_path), updated_offs)
    log(f"  offsets: {len(offsets):,} → {len(updated_offs):,}", log_path)

    # Upsert to Qdrant
    log("  Upserting to Qdrant...", log_path)
    start_id = len(offsets)
    points, upserted = [], 0
    total_vecs = len(all_vecs)
    for i in range(total_vecs):
        points.append(PointStruct(id=start_id + i, vector=all_vecs[i].tolist(), payload={}))
        if len(points) >= QDRANT_BATCH:
            client.upsert(collection_name=COLLECTION, points=points)
            upserted += len(points)
            points = []
            if upserted % 10000 == 0:
                log(f"  Qdrant upserted {upserted:,}/{total_vecs:,}", log_path)
    if points:
        client.upsert(collection_name=COLLECTION, points=points)
        upserted += len(points)

    total = client.count(collection_name=COLLECTION).count
    log(f"  Qdrant upsert done: {upserted:,} points → collection total {total:,}", log_path)
    log(f"=== {label} DONE ===", log_path)
    return upserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Build theology vector search index")
    parser.add_argument(
        "--source-dirs", nargs="+", required=True,
        help="Directories containing .txt source files (e.g. data/cathen_pages data/wiki_pages)"
    )
    parser.add_argument(
        "--out-dir", default="out",
        help="Output directory for chunks.jsonl, offsets.npy, and log (default: out)"
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--qdrant-host", default="localhost",
        help="Qdrant host (default: localhost)"
    )
    parser.add_argument(
        "--qdrant-port", type=int, default=6333,
        help="Qdrant port (default: 6333)"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "build_index.log"

    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, prefer_grpc=True, timeout=120)
    ensure_collection(client)

    log(f"Starting index build — out_dir={out_dir}", log_path)
    log(f"  sources: {args.source_dirs}", log_path)
    log(f"  ollama: {args.ollama_url}  qdrant: {args.qdrant_host}:{args.qdrant_port}", log_path)

    log("Loading existing chunk IDs for dedup...", log_path)
    existing_ids = load_existing_ids(out_dir / "chunks.jsonl")
    log(f"  {len(existing_ids):,} existing chunks", log_path)

    for source_dir_str in args.source_dirs:
        source_dir = Path(source_dir_str)
        source_key = source_dir.name
        label = source_key.replace("_", " ").title()
        index_source(label, source_dir, source_key, out_dir, args.ollama_url, client, existing_ids, log_path)

    log("All sources indexed. Restart the MCP server to reload offsets.", log_path)


if __name__ == "__main__":
    main()
