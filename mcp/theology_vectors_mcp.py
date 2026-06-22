"""
theology_vectors_mcp.py
MCP server: semantic passage search over the full text of a theology corpus.

Set THEOLOGY_OUT_DIR to the directory containing chunks.jsonl and
chunks.offsets.npy (produced by pipeline/build_index.py).

Example mcporter call:
    mcporter call theology-vectors.semantic_search query="logos divine word stoics" k=8
"""

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

OUT = Path(os.environ.get("THEOLOGY_OUT_DIR", str(Path.home() / "theology-out")))
CHUNKS = OUT / "chunks.jsonl"
OFFS = OUT / "chunks.offsets.npy"
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
COLLECTION = "theology_passages"
EMB_MODEL = "nomic-embed-text"
TOP_VEC = 40

mcp = FastMCP("theology-vectors")

_offs = np.load(str(OFFS))
_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=True, timeout=120)
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")


def _rec(line_no: int) -> dict:
    with open(CHUNKS, "rb") as f:
        f.seek(int(_offs[line_no]))
        return json.loads(f.readline())


def _embed_query(q: str) -> list[float]:
    body = json.dumps(
        {"model": EMB_MODEL, "input": ["search_query: " + q], "keep_alive": "2m"}
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        v = np.asarray(json.loads(r.read())["embeddings"][0], dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


@mcp.tool()
def semantic_search(query: str, k: int = 8) -> str:
    """Search the full text of the theology corpus for passages relevant to QUERY.

    Returns top-K reranked passages with book title and relevance score.
    Use this for 'what does the corpus say about X', exact-wording lookups,
    and synonym/paraphrase matches.

    Call with key=value syntax (not --args):
        mcporter call theology-vectors.semantic_search query="logos divine word" k=8

    Args:
        query: a natural-language question or topic
        k: number of passages to return (default 8, max 20)
    """
    k = max(1, min(int(k), 20))
    if not query or len(query.strip()) < 3:
        return "ERROR: query is too short or empty."

    qv = _embed_query(query)
    hits = _client.query_points(COLLECTION, query=qv, limit=TOP_VEC, with_payload=False).points
    cands = [_rec(h.id) for h in hits]
    if not cands:
        return "No passages found."

    scores = _reranker.predict([(query, c.get("text", "")) for c in cands])
    ranked = sorted(zip(scores, cands), key=lambda x: -x[0])

    out, seen = [], set()
    for s, rec in ranked:
        title = rec.get("title", "")
        text = rec.get("text", "")
        key = (title, text[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(f"### {title}  (relevance {s:.1f})\n{text}")
        if len(out) >= k:
            break
    return "\n\n---\n\n".join(out)


@mcp.tool()
def corpus_stats() -> str:
    """Return basic stats about the semantic passage index."""
    count = _client.count(COLLECTION, exact=False).count
    offs_count = len(_offs)
    return (
        f"Theology passage index: {count:,} vectors in Qdrant, "
        f"{offs_count:,} passages in chunks.jsonl. "
        f"Embedder: {EMB_MODEL} (768-dim), "
        f"reranker: cross-encoder/ms-marco-MiniLM-L-6-v2. "
        f"Backend: Qdrant at {QDRANT_HOST}:{QDRANT_PORT}. "
        f"Index directory: {OUT}"
    )


if __name__ == "__main__":
    mcp.run()
