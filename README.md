# oida-rag

Tools for building and exploring a self-contained, retrieval-ready archive of
the **OIDA** corpus — a large collection of mixed-modality litigation
documents (OCR'd PDFs, emails, images) and the network of connections between
them.

The project does two things:

1. **Builds and maintains a LanceDB archive.** A single embedded LanceDB
   dataset holds everything for a document: metadata, the relationship graph,
   the full-text (OCR) search index with embeddings, and — optionally — the
   raw artifact bytes themselves, all in file-efficient columnar storage and
   indexed for rapid retrieval. The `oida-cli ingest` family of commands builds
   this dataset from a Solr source and keeps it in sync.

2. **Serves it to a research-assistant agent.** An MCP server exposes the
   dataset as grounded tools, and a CLI chat agent (driven by a local LLM)
   uses those tools to answer questions, follow document relationships, and
   read artifact contents.

## Architecture

The project is split into a **corpus-agnostic framework** and a thin **domain
crate** that adapts it to OIDA. Two crates in a Cargo workspace:

| Crate | Role |
|-------|------|
| `corpus-index` | The reusable engine, knowing nothing about any particular corpus: the LanceDB store (scalar + full-text/BM25 + vector indexes), the hybrid (RRF) search engine, the raw-artifact store and layered retrieval resolver, read-only SQL, and the ingest / incremental-apply drivers. Transport-agnostic by default; MCP server tools, the chat agent, and CLI helpers are opt-in cargo features (`mcp`, `chat`, `cli`). |
| `oida` | The OIDA domain layer over `corpus-index`: the Solr `SourceProvider`, the document schema and relationship-graph queries, config slices — **plus the two OIDA binaries**, `oida-cli` and `oida-mcp-server`. It re-exports the framework so the binaries compose against a single `oida::` surface. |

A new corpus is therefore a single small crate: implement a `SourceProvider`
and a document row type against `corpus-index`, and reuse the search engine,
MCP tools, and chat agent unchanged.

The two binaries live in the `oida` crate (`crates/oida/src/bin/`):

- **`oida-cli`** — the **dataset management** front-end (`ingest`, `stats`)
  *and* an MCP **client** that spawns the server and drives a local LLM
  tool-calling chat loop.
- **`oida-mcp-server`** — an MCP server (via the official `rmcp` SDK) exposing
  the dataset as tools over **stdio**.

```
 build / maintain                         query
 ─────────────────                        ─────
   Solr  ─┐                                you
          │                                 │
 artifacts├─► oida-cli ingest ─► LanceDB ◄─ oida-mcp-server ◄─ oida-cli chat ◄─► local LLM
 (disk/S3)┘                      dataset                                  (Ollama tool calling)
```

Both binaries live in the `oida` package, so run them with
`cargo run -p oida --bin oida-cli` (and `--bin oida-mcp-server`).

### Writing a domain crate

Everything corpus-specific lives in the `oida` crate; the framework supplies the
rest. Adapting `corpus-index` to a different corpus means writing these elements
— using [crates/oida/](crates/oida/) as the worked example:

1. **A source provider** — implement [`SourceProvider`](crates/corpus-index/src/provider.rs)
   to stream the corpus as pages of Arrow batches, and declare a
   `DocumentsContract` (the `documents` schema, its FTS column, and the scalar
   columns to index). The framework's `build_metadata` / `apply` drivers own
   everything downstream (table writes, indexes, watermark, incremental diffing).
   See [solr_provider.rs](crates/oida/src/solr_provider.rs) and the Solr mapping
   in [solr_map.rs](crates/oida/src/solr_map.rs).

2. **Document row types** — define your document structs and implement
   [`DocumentRow`](crates/corpus-index/src/row.rs) (column list + decode from an
   Arrow row) for each. A lean search projection additionally implements
   `SearchableRow` to declare which fields are scored and how artifact types are
   read. The generic `Index::get<D>` / `search<D>` / `documents_where<D>` are
   parameterized over these. See [model.rs](crates/oida/src/model.rs).

3. **Domain queries** — corpus concepts with no generic meaning (here: Bates
   numbers, conversation threads, the reference graph) go in an *extension trait*
   on the framework's `Index` — a trait, not inherent methods, because the orphan
   rule forbids `impl`ing on a foreign type. Each is built from the generic query
   primitives. See [`CorpusQueries`](crates/oida/src/queries.rs).

4. **Config slices** — add your domain and app config structs and `serde(flatten)`
   them alongside the framework's `CoreConfig` into one aggregate, so the on-disk
   TOML stays flat while framework drivers still take only `&CoreConfig`. Corpus
   branding (system prompt, assistant label) is config, not code. See
   [config.rs](crates/oida/src/config.rs).

5. **The MCP server** — implement the one-method
   [`CorpusBackend`](crates/corpus-index/src/mcp.rs) trait (index / hybrid /
   artifacts accessors) to inherit the generic tools (`search_documents`,
   `hybrid_search`, `get_artifact_text`, `run_sql`, `describe_schema`), then merge
   any domain-only tools with `ToolRouter`'s `+`. Tool descriptions naming corpus
   fields are passed as parameters. See
   [oida-mcp-server/tools.rs](crates/oida/src/bin/oida-mcp-server/tools.rs).

6. **The chat agent** — nothing to write: the CLI calls the generic
   `corpus_index::chat::run(ChatOptions { .. })` with the prompt, label, model,
   and server binary pulled from config. See
   [oida-cli.rs](crates/oida/src/bin/oida-cli.rs).

The performance-critical paths (the per-chunk embed loop, the per-document
mapping) stay monomorphic inside the provider impl; genericity is only at the I/O
boundaries and cold result-materialization.

---

## 1. The LanceDB dataset

Everything lives in one embedded LanceDB dataset (default `oida-lance/`). It is
built from a **Solr** source rather than scanned at query time, so retrieval is
interactive. The dataset is composed of a few tables:

| Table | Contents |
|-------|----------|
| `documents` | One row per document: metadata (title, Bates number, authors, custodian, topic, dates, …), the relationship fields, and an FTS-indexed `search_text` column. |
| `artifacts` | One row per artifact (exploded from each document), with name, media type, md5, and size. Scalar/FTS indexed and joinable on `id`. |
| `chunks` | The hybrid full-text index: OCR text split into overlapping chunks, each with its embedding vector. Built on demand. |
| `raw_artifacts` | Optional blob store of the original (non-text) artifact bytes — PDFs, images, spreadsheets — for self-contained point lookups. Built on demand. |

### Building and maintaining it

All management flows through `oida-cli ingest`:

```sh
# First build: drop and rebuild documents/artifacts from a full Solr scan.
cargo run --release -p oida --bin oida-cli -- ingest --force

# Add the hybrid full-text (OCR) index and/or the raw-artifact blob store.
# These are separate passes over the already-ingested artifacts, so they
# compose with --force or with an incremental update.
cargo run --release -p oida --bin oida-cli -- ingest --force --full-text --store-raw

# Incremental sync (the default with no mode flag, or --update): upsert
# new/changed docs from the stored watermark, delete redactions, invalidate
# stale chunks/raw rows, then resume the requested derived stores.
cargo run --release -p oida --bin oida-cli -- ingest --full-text --store-raw

# Preview the incremental delta without writing anything.
cargo run --release -p oida --bin oida-cli -- ingest --dry-run

# Inspect one full Solr document to understand the source schema.
cargo run --release -p oida --bin oida-cli -- ingest --sample-doc

# Report row counts, archive sizes, and full-text index metadata.
cargo run --release -p oida --bin oida-cli -- stats
```

Key properties:

- **Incremental by default.** A plain `ingest` syncs from the stored watermark;
  `--force` does a full Solr re-ingest. Changed and redacted documents have
  their stale chunks/raw rows invalidated so a follow-up `--full-text` /
  `--store-raw` re-processes exactly what changed.
- **Pluggable artifact source.** The full-text and raw-store passes read
  artifact bytes from a local directory (`artifact_root`) or from **S3** /
  S3-compatible stores (`s3_bucket`, …). Without a source, metadata-only
  ingest still works and artifact-text tools degrade gracefully.
- **Embeddings via an OpenAI-compatible API.** The full-text build embeds
  chunks against `embed_host` (e.g. a local Ollama or a vLLM sidecar). A
  comma-separated list of replicas is balanced client-side by least
  connections. The embedding model name is recorded in the dataset, and queries
  always use that recorded model so search can never disagree with the stored
  vectors. See [docs/hybrid-search.md](docs/hybrid-search.md).

---

## 2. The research assistant (MCP server + chat)

Once the dataset is built, chat with it. The CLI spawns the MCP server as a
child process and drives a local Ollama tool-calling loop against it:

```sh
# Interactive REPL
cargo run --release -p oida --bin oida-cli -- chat

# One-shot question
cargo run --release -p oida --bin oida-cli -- chat --once \
  "Find weekly retail reports and give me a document id and Bates number."
```

REPL commands: `/reset` clears the conversation, `/exit` (or Ctrl-D) quits.

### MCP tools

| Tool | Purpose |
|------|---------|
| `search_documents` | Keyword-search document metadata (title, Bates number, authors, custodian, topic, description) with optional filters and pagination, ranked with match provenance. |
| `hybrid_search` | Search document *contents* (OCR text) with hybrid keyword + semantic matching fused by Reciprocal Rank Fusion. Requires the full-text index. |
| `get_document` | Full metadata for one document (by id or Bates number) plus its artifact list. |
| `get_artifact_text` | Read an artifact's OCR text. Returns a status (`text_loaded`, `artifact_file_missing`, `unsupported_artifact_type`, `artifact_root_not_configured`). |
| `get_related` | Traverse the relationship graph — attachments, mentions, related references, and shared email conversation — returning typed edges with resolved neighbors. |
| `run_sql` | Run a single read-only DataFusion SQL query against the `documents`/`artifacts`/`chunks` tables for ad-hoc counting, grouping, and filtering. |
| `describe_schema` | List the columns and Arrow types of the tables, to write correct `run_sql` queries. |

The MCP server starts even when the full-text index or artifact source is
absent — the corresponding tools simply report that they are unavailable.

---

## Prerequisites

- Rust (edition 2024).
- A Solr source for the corpus (default query `industrycode:OPIOIDS`); set
  `solr_url`.
- For chat: [Ollama](https://ollama.com) running locally with a tool-capable
  model, e.g. `ollama pull qwen2.5-coder:latest`.
- For the full-text index: an OpenAI-compatible embedding endpoint and a model
  (e.g. `ollama pull nomic-embed-text`, or a vLLM sidecar).
- For artifact text / raw storage: the artifact files on disk
  (`artifact_root`) or in S3 (`s3_bucket`).

## Configuration

Settings resolve from defaults → `oida.toml` → environment variables → CLI
flags. Copy `oida.toml.example` to `oida.toml` and edit.

| Setting | Config key | Env | CLI flag |
|---------|-----------|-----|----------|
| LanceDB dataset path | `lance_path` | `OIDA_LANCE` | `--lance-path` |
| Solr core URL | `solr_url` | `OIDA_SOLR_URL` | `--solr-url` |
| Solr query | `solr_query` | `OIDA_SOLR_QUERY` | `--solr-query` |
| Artifact directory | `artifact_root` | `OIDA_ARTIFACT_ROOT` | `--artifact-root` |
| S3 bucket / region / endpoint / prefix | `s3_*` | `OIDA_S3_*` | `--s3-*` |
| Embedding host (OpenAI-compatible) | `embed_host` | `OIDA_EMBED_HOST` | `--embed-host` |
| Embedding model | `embed_model` | `OIDA_EMBED_MODEL` | `--embed-model` |
| Ollama host (chat) | `ollama_host` | `OIDA_OLLAMA_HOST` | `--ollama-host` |
| Ollama model (chat) | `ollama_model` | `OIDA_MODEL` | `--model` |

See `oida.toml.example` for the full set, including chunking and build-tuning
options (`chunk_bytes`, `chunk_overlap`, `write_buffer_bytes`,
`ingest_buffer_bytes`, `embed_concurrency`, `read_concurrency`, `embed_batch`,
`embed_lookahead`, `compact_on_build`, `raw_file_bytes`) described in
[docs/hybrid-search.md](docs/hybrid-search.md).

## Notes

- Some models (including `qwen2.5-coder`) emit tool calls as JSON text rather
  than via Ollama's native `tool_calls` field. The CLI parses both.
- The agent loop has guardrails: a max number of tool-call rounds per turn,
  duplicate-call detection, and truncation of oversized tool results.
- `cargo run -p oida --example smoke -- "your query"` exercises the core
  search/document/graph paths directly against the dataset (no LLM), useful for
  validating it.
</content>
</invoke>
