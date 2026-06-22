# Theology Vector Search

Semantic passage search over a large theology corpus, exposed as an MCP server for AI assistants (Claude, Finn, or any mcporter-compatible agent).

Built on:
- **[Qdrant](https://qdrant.tech/)** — local vector database (~10M+ passages)
- **[Ollama](https://ollama.com/)** + `nomic-embed-text` — local GPU embedding
- **[sentence-transformers](https://sbert.net/)** — cross-encoder reranking
- **[FastMCP](https://github.com/jlowin/fastmcp)** — MCP server wrapper
- **[mcporter](https://github.com/mcporter/mcporter)** — routes tool calls from AI assistants

Once installed, your AI assistant can answer theology questions grounded in actual primary-source passages rather than from memory.

## Corpus sources

| Source | Articles | Notes |
|---|---|---|
| Catholic Encyclopedia (1907–1913) | ~3,700 | Public domain; `download_cathen.py` fetches from newadvent.org |
| Wikipedia theology articles | ~300 | Curated list in `config/wiki_titles.txt`; `download_wiki.py` fetches via API |
| Project Gutenberg books | 77,000+ | Optional; find theology books at gutenberg.org |

The pre-built index (10M+ passages) is too large for git. See **Pre-built index** below for options.

---

## Hardware requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | None (CPU-only, slow) | NVIDIA RTX 3080+ |
| RAM | 16 GB | 32 GB |
| Disk | 10 GB (index only) | 60 GB (index + sources) |
| OS | Linux / macOS / Windows | Any |

Building from source takes ~30 min on an RTX 5080 for cathen+wiki, longer for larger corpora.

---

## Quick start

### 1. Install dependencies

```bash
# Python deps
pip install -r requirements.txt

# Qdrant (Docker)
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant

# Ollama + embedding model
# Install Ollama from https://ollama.com
ollama pull nomic-embed-text

# mcporter (Node.js)
npm install -g mcporter
```

### 2. Download source texts

```bash
# Catholic Encyclopedia (~3,700 articles, ~1 hour at 1 req/sec)
python pipeline/download_cathen.py --out-dir data/cathen_pages

# Wikipedia theology articles (~300 articles, ~5 minutes)
python pipeline/download_wiki.py --out-dir data/wiki_pages
```

### 3. Build the index

```bash
python pipeline/build_index.py \
    --source-dirs data/cathen_pages data/wiki_pages \
    --out-dir out \
    --qdrant-host localhost --qdrant-port 6333

# Add more sources any time — the pipeline deduplicates automatically:
python pipeline/build_index.py \
    --source-dirs data/my_other_texts \
    --out-dir out
```

The pipeline resumes safely on crash. A checkpoint file (`embed_ckpt_<source>.npy`) is saved every 2,000 chunks so you never re-embed from scratch.

### 4. Configure the MCP server

Copy `config/mcporter.example.json` to `~/.mcporter/mcporter.json` and replace the placeholder paths with absolute paths on your machine:

```json
{
  "servers": {
    "theology-vectors": {
      "command": "python",
      "args": ["/home/you/theology-vector-search/mcp/theology_vectors_mcp.py"],
      "env": {
        "THEOLOGY_OUT_DIR": "/home/you/theology-vector-search/out",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "OLLAMA_URL": "http://localhost:11434",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8"
      },
      "lifecycle": "keep-alive"
    }
  }
}
```

### 5. Query

```bash
# Test the server directly
mcporter call theology-vectors.semantic_search query="logos divine word stoics heraclitus" k=5

# Check index stats
mcporter call theology-vectors.corpus_stats
```

---

## Pre-built index

The built index (`out/chunks.jsonl`, `out/chunks.offsets.npy`) is too large for git. Two options:

**Option A — build locally** (free, takes time, needs GPU for speed):
Follow the Quick start steps above.

**Option B — download pre-built** (fast, ~30 GB download):
A pre-built index for cathen+wiki is available on Hugging Face:
*(link to be added after upload)*

To use a downloaded index, place `chunks.jsonl` and `chunks.offsets.npy` in your `--out-dir`, then restore the Qdrant collection:

```bash
# If you have a Qdrant snapshot (.snapshot file):
curl -X POST 'http://localhost:6333/collections/theology_passages/snapshots/upload' \
    -H 'Content-Type: multipart/form-data' \
    -F 'snapshot=@theology_passages.snapshot'

# If you only have the pre-built .jsonl and want to re-upsert vectors,
# just re-run build_index.py — it skips already-seen chunks and only upserts missing vectors.
```

---

## Adding your own texts

Put `.txt` files in any directory and point `build_index.py` at it:

```bash
python pipeline/build_index.py \
    --source-dirs data/my_books \
    --out-dir out
```

The pipeline chunks (450 words, 380-word stride), embeds, and upserts into Qdrant. Existing chunks are skipped by MD5 dedup, so you can safely re-run with the same source directories after adding new files.

---

## TOOLS.md for AI assistants

If you use an AI assistant (e.g. via openclaw/Finn), add this to your assistant's `TOOLS.md` so it knows how to call the search:

```markdown
### Theology corpus

Use `key=value` syntax — NOT `--args` (breaks on Windows):
    mcporter call theology-vectors.semantic_search query="your question here" k=8

Returns top-k reranked passages with book title and relevance score.
Always cite the book titles in your reply. Use this for any theology,
biblical studies, church history, or religious philosophy question.
```

---

## Project structure

```
theology-vector-search/
├── pipeline/
│   ├── build_index.py       # chunk → embed → Qdrant (main pipeline)
│   ├── download_cathen.py   # fetch Catholic Encyclopedia from newadvent.org
│   └── download_wiki.py     # fetch curated Wikipedia theology articles
├── mcp/
│   └── theology_vectors_mcp.py  # FastMCP server (THEOLOGY_OUT_DIR env var)
├── config/
│   ├── mcporter.example.json    # mcporter server config template
│   └── wiki_titles.txt          # curated Wikipedia article list (~300 titles)
├── requirements.txt
└── README.md
```

---

## License

Code: MIT.

Source texts:
- **Catholic Encyclopedia** — public domain (published 1907–1913)
- **Wikipedia** — CC BY-SA 4.0 (attribution required; see each article's history page)
- **Project Gutenberg** — public domain (verify per book at gutenberg.org)
