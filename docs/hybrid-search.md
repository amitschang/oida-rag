# Hybrid content search

The base retrieval in oida-rag (`search_documents`, `get_related`) matches
document **metadata** — titles, Bates numbers, authors, custodians, topics, and
the relationship graph. It never looks at what a document actually *says*.

The **hybrid search index** closes that gap. It reads the OCR / plain-text
contents of the archive's artifacts and lets you find documents by their text,
combining two complementary kinds of matching:

- **Keyword (full-text) search** — exact term matching, great for names, codes,
  and phrases you already know.
- **Semantic (vector) search** — embedding similarity, great for finding text
  that *means* the same thing even when the wording differs.

The two rankings are fused with **Reciprocal Rank Fusion (RRF)**, so a document
that ranks well in *either* method surfaces, and documents that rank well in
*both* surface near the top.

Everything runs locally and embedded: there is no external search service. The
index lives in a [LanceDB](https://lancedb.github.io/lancedb/) database on disk
(default directory `oida-lance/`) — the *same* database that holds the
document/artifact metadata tables (`documents`, `artifacts`); the hybrid index
just adds a `chunks` table beside them. Embeddings come from a local
[Ollama](https://ollama.com) model.

## How it works

The hybrid index is the second of two ingest phases. A metadata `ingest` runs
first, loading the parquet into the LanceDB `documents` and `artifacts` tables
(see the project README). The `--full-text` phase below builds on top of that:
it reads the artifact list from the `artifacts` table, so a metadata ingest must
have completed before it can run.

### 1. Source text

The build enumerates every artifact whose contents are plain text — files with
a `text/plain` media type or a `.ocr` extension — by querying the `artifacts`
table in the LanceDB index. Each such artifact is then read from `artifact_root`
on disk.

### 2. Chunking

Long documents are split into overlapping, byte-bounded **chunks** before
embedding. Two settings control this:

- `chunk_bytes` (default `2048`) — the target size of each chunk.
- `chunk_overlap` (default `256`) — how many bytes adjacent chunks share, so a
  match that straddles a boundary is not lost.

Chunking keeps each embedded passage focused (which improves semantic recall)
and respects UTF-8 character boundaries.

### 3. Embedding

Each chunk's text is sent to Ollama's `/api/embed` endpoint using the configured
`embed_model` (default `nomic-embed-text`). The resulting vector is stored
alongside the chunk.

### 4. Storage and indexing

Chunks are written to a single LanceDB table, `chunks`, with the columns:

| Column | Type | Meaning |
|--------|------|---------|
| `doc_id` | `Utf8` | The document the chunk belongs to. |
| `chunk_idx` | `Int32` | The chunk's position within its artifact. |
| `artifact_name` | `Utf8` | The source text file. |
| `text` | `Utf8` | The chunk text (also used for the result snippet). |
| `vector` | `FixedSizeList<Float32, dim>` | The chunk embedding. |

Chunk batches are buffered in memory and flushed to LanceDB once they reach
`write_buffer_bytes`, so each write lands as one large fragment rather than one
per embed call. After loading (and an optional compaction pass, see
`compact_on_build`), the build creates:

- a **full-text index** on `text`, and
- a **vector index** on `vector` (skipped below 256 chunks, where exact flat
  search is already fast and ANN training has too few rows; if ANN index
  creation fails the build logs a warning and falls back to flat search).

### 5. Querying

At query time the index runs a vector search and a full-text search over the
same table, fuses the two rankings with RRF, and then **collapses chunk hits to
one hit per document** (keeping each document's best-ranked chunk). Each result
carries the fused relevance score, the source artifact name, and a short text
snippet. Document metadata (Bates number, title, date, etc.) is hydrated from
the `documents` table in the same LanceDB index so results look like the other
tools' output.

## Embedding-model consistency

A vector index is only meaningful if queries are embedded with the **same model**
that produced the stored vectors. Mixing models (or even silently changing a
model's weights behind the same tag) yields quietly wrong results.

The index defends against this by recording, at build time, a single-row `_meta`
table containing the embed model name, the vector dimension, the model's content
**digest**, the build timestamp, and the chunking settings. The `_meta` row is
the source of truth:

1. **The stored model wins.** Queries embed with the model name from `_meta`,
   never with whatever `embed_model` happens to be in the current config. A
   mismatch between the index and config cannot produce wrong results.
2. **Dimension check.** The query vector's length is asserted against the stored
   dimension.
3. **Digest check.** Before serving, the index fetches the model's current
   digest from Ollama (`/api/tags`) and compares it to the stored digest. If the
   model changed under the same tag, the query fails with a clear error telling
   you to rebuild — rather than returning garbage.

If you intentionally change the embedding model, rebuild the index (see below):
`ingest --full-text` refuses to overwrite an existing `chunks` table unless you
pass `--force`, which drops and rebuilds it from scratch.

## Building and inspecting the index

### Prerequisites

- An embedding model pulled into Ollama:
  ```sh
  ollama pull nomic-embed-text
  ```
- `artifact_root` (or an S3 source) configured and pointing at the artifact
  files (the build needs the actual text, not just the metadata).
- The metadata index already built (`oida-cli ingest --force`), since the
  full-text build reads the artifact list from the LanceDB `artifacts` table.
  Passing `--full-text` to the same command builds both phases at once.

### Commands

The hybrid index is built by the `ingest` subcommand's `--full-text` flag. By
default `ingest` performs an *incremental* update from the stored watermark and
re-embeds only new/changed documents; pass `--force` to rebuild metadata and the
index from scratch.

```sh
# Fresh full build: rebuild metadata AND the hybrid index from a full Solr scan
oida-cli ingest --force --full-text

# Incremental: sync metadata, then re-embed only new/changed documents
oida-cli ingest --full-text

# Build with a specific embedding model (global flag, overrides config)
oida-cli ingest --force --full-text --embed-model mxbai-embed-large

# Show statistics about the ingested index (metadata + hybrid, if built)
oida-cli stats
```

> `--force` rebuilds both the metadata tables and the chunks index from a full
> Solr scan. Without it, `ingest --full-text` updates metadata in place and
> re-embeds only the documents that are new or changed since the last run, so it
> can be re-run cheaply after the source archive changes.

`stats` reports the document, artifact, and chunk counts, and — when the hybrid
index is built — the embed model and its digest, the vector dimension, and the
chunking settings recorded at build time.

> Building embeds every chunk via Ollama, so the time scales with corpus size
> and your embedding throughput. It is a one-time cost; queries are fast.

## Using it

Once the index exists, the MCP server picks it up automatically on startup and
exposes the `hybrid_search` tool. If the index is absent, the server still runs
and all the metadata tools work — `hybrid_search` simply reports that the index
needs to be built.

`hybrid_search` takes a `query` (natural-language or keyword) and an optional
`limit` (default 10, max 50), and returns ranked documents with a matching
snippet and the source artifact. In a chat session you can just ask a
content-oriented question and the model will reach for it.

## Configuration

All settings live in `oida.toml` (see `oida.toml.example`) and fall back to the
defaults shown:

| Config key | Default | Meaning |
|------------|---------|---------|
| `parquet_path` | `oida-index.parquet` | Source parquet the metadata ingest reads. Also `OIDA_PARQUET`. |
| `lance_path` | `oida-lance` | Directory holding the LanceDB index (metadata **and** hybrid `chunks`). Also `OIDA_LANCE`. |
| `artifact_root` | _(unset)_ | Directory of on-disk artifact files. Required to build the hybrid index. Also `OIDA_ARTIFACT_ROOT`. |
| `embed_model` | `nomic-embed-text` | Ollama model used to embed text **when building**. Queries use the model stored in the index. |
| `chunk_bytes` | `2048` | Target chunk size in bytes. |
| `chunk_overlap` | `256` | Overlap in bytes between adjacent chunks. |
| `write_buffer_bytes` | `1073741824` (1 GiB) | In-memory buffer before embedded chunks flush to LanceDB as one fragment. |
| `compact_on_build` | `true` | Compact the `chunks` table before indexing, merging small fragments. |
| `ingest_buffer_bytes` | `536870912` (512 MiB) | In-memory buffer before the metadata ingest flushes a `documents`/`artifacts` fragment. |

The embedding requests reuse the existing `ollama_host` setting. Each
build-relevant key also has a CLI flag and `OIDA_*` env var (see
`oida-cli --help`).

## Where it lives in the code

- `crates/oida-core/src/ingest.rs` — the parquet → LanceDB metadata ingest
  (`documents`/`artifacts` tables) that the hybrid build sits on top of.
- `crates/oida-core/src/index.rs` — the metadata index handle, including
  `text_artifacts()`, which lists the artifacts the hybrid build reads.
- `crates/oida-core/src/hybrid.rs` — the LanceDB hybrid build/query module
  (chunking, embedding orchestration, RRF query, `_meta` consistency guards).
- `crates/oida-core/src/embed.rs` — the minimal Ollama embedding client
  (`/api/embed`) and digest lookup (`/api/tags`).
- `crates/oida-cli/src/main.rs` — the `ingest [--full-text] [--force]` and
  `stats` subcommands.
- `crates/oida-mcp-server/src/tools.rs` — the `hybrid_search` MCP tool.
