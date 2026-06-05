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
| `oida-core` | Domain logic: config, the DuckDB-backed index (search + relationship graph), and artifact access. Transport-agnostic. |
| `oida-mcp-server` | An MCP server (via the official `rmcp` SDK) exposing the index as tools over **stdio**. |
| `oida-cli` | An MCP **client** that spawns the server as a child process and drives an Ollama tool-calling loop with a REPL. |

```
 you ──► oida-cli ──►    local LLM     ────┐
            |                              │
            | ◄── (Ollama tool calling) ◄──┘
            │
            └── MCP ──► oida-mcp-server ◄── DuckDB cache ◄── parquet index
                              ▲
                    artifact files on disk
```

### How retrieval works (v1)

The 2.7 GB parquet has ~24M artifact rows / ~7.6M documents. Scanning it per
query is not interactive, so on first run we build a persistent **DuckDB cache**
(`oida.duckdb`): a deduplicated, document-level `documents` table plus a thin
`artifacts` table, both indexed. All tools query that cache.

Retrieval is **metadata + keyword + graph** (no embeddings):

- **Keyword search** — case-insensitive matching over title, Bates number,
  authors, custodian, topic, and description, ranked by how many query terms
  match, with provenance (which fields matched).
- **Graph traversal** — documents reference each other by Bates number through
  `attachment`, `related`, and `men` (mentions), and share email threads via
  `conversation`. The graph tool resolves these into neighbor documents.

### MCP tools

| Tool | Purpose |
|------|---------|
| `search_documents` | Keyword-search documents with optional media-type / custodian filters and pagination. |
| `get_document` | Full metadata for one document (by id or Bates number) plus its artifact list. |
| `get_artifact_text` | Read an artifact's OCR text from disk. Returns a status (`text_loaded`, `artifact_file_missing`, `unsupported_artifact_type`, `artifact_root_not_configured`). |
| `get_related` | Traverse the document relationship graph (attachments, mentions, related, conversation). |

## Prerequisites

- Rust (edition 2024).
- [Ollama](https://ollama.com) running locally with a tool-capable model, e.g.
  `ollama pull qwen2.5-coder:latest`.
- The `oida-index-by-artifact.parquet` file in the working directory (or set
  `parquet_path`).
- Optionally, the artifact files on disk (set `artifact_root`). Without them,
  artifact-text tools degrade gracefully.

The first build compiles a bundled DuckDB and takes a few minutes.

## Usage

```sh
# Build everything
cargo build --release

# (Optional) configure
cp oida.toml.example oida.toml   # then edit

# Build the DuckDB cache once (otherwise it builds on first server start).
# This deduplicates ~24M rows and indexes the result; it takes several minutes.
cargo run --release -p oida-mcp-server -- build-cache

# Chat interactively (the CLI spawns the server for you)
cargo run --release -p oida-cli

# Or ask a single question and exit
cargo run --release -p oida-cli -- --once "Find weekly retail reports and give me a document id and Bates number."
```

REPL commands: `/reset` clears the conversation, `/exit` (or Ctrl-D) quits.

### Configuration

Settings resolve from defaults → `oida.toml` → environment variables → CLI flags.

| Setting | Config key | Env | CLI flag |
|---------|-----------|-----|----------|
| Parquet path | `parquet_path` | `OIDA_PARQUET` | — |
| Cache path | `cache_path` | `OIDA_CACHE` | — |
| Artifact directory | `artifact_root` | `OIDA_ARTIFACT_ROOT` | `--artifact-root` |
| Ollama host | `ollama_host` | `OIDA_OLLAMA_HOST` | `--ollama-host` |
| Ollama model | `ollama_model` | `OIDA_MODEL` | `--model` |

See `oida.toml.example` for the full set.

## Notes

- Some models (including `qwen2.5-coder`) emit tool calls as JSON text rather
  than via Ollama's native `tool_calls` field. The CLI handles both: it parses
  tool calls out of message content as a fallback.
- The agent loop has guardrails: a max number of tool-call rounds per turn,
  duplicate-call detection, and truncation of oversized tool results.
- `cargo run -p oida-core --example smoke -- "your query"` runs the core
  search/document/graph paths directly against the cache (no LLM), useful for
  validating the index.
