# oida-rag

A small, learning-oriented Rust application for querying and understanding the
**OIDA** archive — a large parquet index describing mixed-modality artifacts
(OCR'd PDFs, emails, images) and the network of connections between documents.

A local [Ollama](https://ollama.com) model answers questions by calling tools
exposed over the [Model Context Protocol](https://modelcontextprotocol.io)
(MCP). The MCP server gives the model grounded access to the index and the
on-disk artifacts; you interact through a simple CLI chat.

## Architecture

Three crates in a Cargo workspace:

| Crate | Role |
|-------|------|
| `oida-core` | Domain logic: config, the LanceDB-backed index (search + relationship graph), and artifact access. Transport-agnostic. |
| `oida-mcp-server` | An MCP server (via the official `rmcp` SDK) exposing the index as tools over **stdio**. |
| `oida-cli` | An MCP **client** that spawns the server as a child process and drives an Ollama tool-calling loop with a REPL. |

```
 you ──► oida-cli ──►    local LLM     ────┐
            |                              │
            | ◄── (Ollama tool calling) ◄──┘
            │
            └── MCP ──► oida-mcp-server ◄── LanceDB index ◄── parquet
                              ▲
                    artifact files on disk
```

### How retrieval works (v1)

The 2.7 GB parquet has one row per document (~7.6M) with that document's
artifacts inline as a `list<struct>` column. Scanning it per query is not
interactive, so we **ingest** it into a persistent embedded **LanceDB index**
(default `oida-lance/`) using DataFusion: a document-level `documents` table
(with an FTS-only `search_text` column) plus a thin `artifacts` table exploded
from the inline list, both scalar/FTS indexed. All tools query that index.

Retrieval is **metadata + keyword + graph** (no embeddings):

- **Keyword search** — case-insensitive matching over title, Bates number,
  authors, custodian, topic, and description, ranked by how many query terms
  match, with provenance (which fields matched).
- **Graph traversal** — documents reference each other by Bates number through
  `attachment`, `related`, and `men` (mentions), and share email threads via
  `conversation`. The graph tool resolves these into neighbor documents.

On top of this, an optional **hybrid search index** searches the *contents* of
documents (OCR text) by combining keyword (full-text) and semantic (vector)
matching, fused with Reciprocal Rank Fusion. It is built on demand with
`oida-cli ingest --full-text` and exposed via the `hybrid_search` tool. See
[docs/hybrid-search.md](docs/hybrid-search.md) for how it works and how to build
it.

### MCP tools

| Tool | Purpose |
|------|---------|
| `search_documents` | Keyword-search documents with optional media-type / custodian filters and pagination. |
| `get_document` | Full metadata for one document (by id or Bates number) plus its artifact list. |
| `get_artifact_text` | Read an artifact's OCR text from disk. Returns a status (`text_loaded`, `artifact_file_missing`, `unsupported_artifact_type`, `artifact_root_not_configured`). |
| `get_related` | Traverse the document relationship graph (attachments, mentions, related, conversation). |
| `hybrid_search` | Search document *contents* (OCR text) with hybrid keyword + semantic matching (RRF). Requires the [hybrid index](docs/hybrid-search.md) to be built. |

## Prerequisites

- Rust (edition 2024).
- [Ollama](https://ollama.com) running locally with a tool-capable model, e.g.
  `ollama pull qwen2.5-coder:latest`.
- The `oida-index.parquet` file in the working directory (or set `parquet_path`).
- Optionally, the artifact files on disk (set `artifact_root`). Without them,
  artifact-text tools degrade gracefully (and the hybrid index cannot be built).

## Usage

```sh
# Build everything
cargo build --release

# (Optional) configure
cp oida.toml.example oida.toml   # then edit

# Build the metadata index from Solr once (required before chatting). Add
# --full-text to also build the hybrid semantic index over the OCR artifacts.
cargo run --release -p oida-cli -- ingest --force

# Chat interactively (the CLI spawns the server for you)
cargo run --release -p oida-cli -- chat

# Or ask a single question and exit
cargo run --release -p oida-cli -- chat --once "Find weekly retail reports and give me a document id and Bates number."
```

REPL commands: `/reset` clears the conversation, `/exit` (or Ctrl-D) quits.

### Hybrid content search (optional)

To search what documents *say* (not just their metadata), build the hybrid
keyword + semantic index over the OCR text. It needs an embedding model and the
artifact files on disk:

```sh
ollama pull nomic-embed-text                       # default embedding model
cargo run --release -p oida-cli -- ingest --force --full-text   # build metadata + the index
cargo run --release -p oida-cli -- stats                # inspect it
```

Once built, the MCP server exposes the `hybrid_search` tool automatically. See
[docs/hybrid-search.md](docs/hybrid-search.md) for the full design, the
embedding-model consistency guarantees, and tuning options.

### Configuration

Settings resolve from defaults → `oida.toml` → environment variables → CLI flags.

| Setting | Config key | Env | CLI flag |
|---------|-----------|-----|----------|
| Parquet path | `parquet_path` | `OIDA_PARQUET` | `--parquet-path` |
| LanceDB index path | `lance_path` | `OIDA_LANCE` | `--lance-path` |
| Artifact directory | `artifact_root` | `OIDA_ARTIFACT_ROOT` | `--artifact-root` |
| Ollama host | `ollama_host` | `OIDA_OLLAMA_HOST` | `--ollama-host` |
| Ollama model | `ollama_model` | `OIDA_MODEL` | `--model` |

See `oida.toml.example` for the full set, including the ingest/hybrid-search
settings (`embed_model`, `chunk_bytes`, `chunk_overlap`, `write_buffer_bytes`,
`compact_on_build`, `ingest_buffer_bytes`) described in
[docs/hybrid-search.md](docs/hybrid-search.md).

## Notes

- Some models (including `qwen2.5-coder`) emit tool calls as JSON text rather
  than via Ollama's native `tool_calls` field. The CLI handles both: it parses
  tool calls out of message content as a fallback.
- The agent loop has guardrails: a max number of tool-call rounds per turn,
  duplicate-call detection, and truncation of oversized tool results.
- `cargo run -p oida-core --example smoke -- "your query"` runs the core
  search/document/graph paths directly against the LanceDB index (no LLM),
  useful for validating the index.
