# Codebase Index — Pi Extension Design Doc

## Goal

Build a Pi extension that provides Cursor-like codebase intelligence for local projects.

## Architecture

```text
Workspace files → Tree-sitter chunking → content_hash
                                         │
                    ┌────────────────────┤
                    ▼                    ▼
            PostgreSQL (Docker)    OpenRouter API
            ┌─────────────────┐    Qwen3-Embedding-0.6B
            │ pg_trgm (regex) │    $0.01/M tokens
            │ tsvector (kw)   │
            │ pgvector (sem)  │    DeepInfra API
            │ JSONB (meta)    │    Qwen3-Reranker-0.6B
            │ content_hash    │    $0.01/M tokens
            └────────┬────────┘
                     │
              hybrid_search()
              (RRF fusion of all 3)
                     │
              codebase_search tool
              (single Pi registerTool)
                     │
                   Pi LLM
```

## Tech Stack

- **Database**: PostgreSQL with pg_trgm + tsvector + pgvector + pgvectorscale
- **Chunking**: Tree-sitter (AST-aware, splits at function/class boundaries)
- **Embeddings**: Qwen3-Embedding-0.6B via OpenRouter ($0.01/M, 1024d, 32K context)
- **Reranking**: Qwen3-Reranker-0.6B via DeepInfra ($0.01/M)
- **Change detection**: content_hash per chunk (Merkle tree equivalent)
- **File watching**: chokidar for incremental re-indexing

## Tool Design

Single `codebase_search` tool registered via `pi.registerTool()`:

```typescript
codebase_search({
  query: string,           // regex, keyword, or natural language
  mode?: "auto" | "regex" | "keyword" | "semantic",
  language?: string,       // filter by language
  maxResults?: number      // default 10
})
```

- `mode: "auto"` routes query to best strategy (like Cursor)
- Does NOT override `code_search` (that's pi-web-access for internet code search)
- `codebase_search` = local project, `code_search` = internet

## PG Schema

```sql
CREATE TABLE code_chunks (
    id          BIGSERIAL PRIMARY KEY,
    file_path   TEXT NOT NULL,
    language    TEXT,
    symbol_type TEXT,          -- 'function', 'class', 'method', 'module', 'block'
    symbol_name TEXT,
    start_line  INT,
    end_line    INT,
    content     TEXT NOT NULL,
    content_hash TEXT NOT NULL, -- for incremental updates
    embedding   vector(1024),  -- Qwen3-Embedding dimensions
    metadata    JSONB DEFAULT '{}',
    indexed_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Regex search (trigram)
CREATE INDEX idx_trgm ON code_chunks USING gin (content gin_trgm_ops);
-- Keyword search (full-text)
CREATE INDEX idx_fts ON code_chunks USING gin (to_tsvector('english', content));
-- Semantic search (vector)
CREATE INDEX idx_vec ON code_chunks USING diskann (embedding vector_cosine_ops);
-- Metadata filter
CREATE INDEX idx_meta ON code_chunks USING gin (metadata);
-- Change detection
CREATE INDEX idx_hash ON code_chunks (content_hash);
```

## Extension File Structure

```text
~/.pi/agent/extensions/codebase-index/
    ├── index.ts              # Pi extension entry point
    ├── db.ts                 # PostgreSQL connection + schema management
    ├── indexer.ts            # File discovery + Tree-sitter chunking + hashing
    ├── embed.ts              # Embedding generation via OpenRouter
    ├── rerank.ts             # Reranking via DeepInfra
    ├── search.ts             # hybrid_search: trgm + tsvector + pgvector → RRF → rerank
    ├── watcher.ts            # File watcher for incremental re-indexing
    ├── package.json
    └── docker-compose.yml    # PG with pgvector + pg_trgm + pgvectorscale
```

## API Keys Needed

- `OPENROUTER_API_KEY` — for embeddings
- `DEEPINFRA_API_KEY` — for reranking

## Cost Estimate

| Operation | Tokens | Cost |
| --- | --- | --- |
| Initial index (50K LOC) | ~2M | $0.02 |
| Daily incremental sync (20 files) | ~40K | $0.0004 |
| Search query | ~100 | $0.000001 |
| Rerank 20 results | ~30K | $0.0003 |
| **Monthly total** | | **< $0.05** |

## Cursor Features We're Replicating

| Cursor Feature | Our Implementation |
| --- | --- |
| Merkle tree change detection | content_hash per chunk |
| Sparse n-gram / Instant Grep | pg_trgm GIN index |
| Semantic search (embeddings) | pgvector HNSW/DiskANN |
| Agentic search routing | mode: "auto" with heuristic |
| Explore subagent | Single tool (Pi limitation) |
| Hybrid search + reranking | RRF fusion → Qwen3-Reranker |

## Pi Extension Patterns to Use

- `pi.registerTool()` for codebase_search
- `pi.on("session_start")` to trigger indexing
- `pi.on("session_shutdown")` for cleanup
- `promptSnippet` + `promptGuidelines` to steer the LLM
- `withFileMutationQueue()` not needed (read-only tool)

## Reference: User's Previous Work

- `SilviuSavu/toolshed` — Rust MCP tool registry with Sourcegraph, code search, dep-docs
- `SilviuSavu/bridge` — VS Code MCP bridge with LSP/DAP, symbol resolution
- `dep-docs` tool — Qdrant + OpenRouter + DeepInfra + Tree-sitter (pattern proven)
- `deepsearch` module — Agentic LLM-driven code research with tool_calls
