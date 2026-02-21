# Hammy

> Codebase Intelligence Engine — deep structural and historical context for AI coding agents

Hammy is a specialized intelligence layer that gives AI coding agents a high-fidelity "brain" for understanding complex, multi-language codebases. It parses source code into a queryable property graph, tracks version control history, and exposes everything through an MCP server and CrewAI agent tools.

## Features

- **Multi-language AST parsing** — PHP, JavaScript, TypeScript, Python, Go (Tree-sitter based)
- **Cross-language bridge detection** — links `fetch('/api/users')` in JS to `#[Route('/api/users')]` in PHP
- **Call graph tracking** — indexes function call sites so you can find all callers of any symbol
- **Semantic + hybrid search** — dense vector search (Qdrant) combined with BM25 via Reciprocal Rank Fusion
- **Structural search** — filter symbols by visibility, async, param count, return type, file, complexity
- **Impact analysis** — N-hop call graph traversal to find the blast radius of any change
- **PR diff analysis** — parse a unified diff, extract changed symbols, compute their blast radius
- **Hotspot scoring** — rank symbols by `log(callers) × log(churn)` to surface high-risk code
- **LLM enrichment** — auto-generate summaries for every indexed function and class
- **Brain / memory layer** — agents can store and recall context across sessions in Qdrant
- **Watch mode** — incremental re-indexing on file change (inotify/FSEvents via watchfiles)
- **VCS integration** — Git log, blame, churn metrics; Mercurial scaffolded
- **MCP server** — expose all tools over the Model Context Protocol for IDE/LLM client use
- **Smart ignore system** — four-layer filtering: defaults → .gitignore → .hgignore → .hammyignore

## Architecture

Hammy represents codebases as a property graph:

- **Nodes** — files, classes, functions, methods, endpoints
- **Edges** — calls, imports, bridges between languages, API endpoint links
- **Metadata** — line numbers, complexity, visibility, churn rate, LLM summaries

A multi-agent system built on CrewAI answers queries by coordinating:

- **The Explorer** — maps code structure, searches symbols, traces call graphs
- **The Historian** — analyzes VCS history for authorship, churn, and temporal patterns
- **The Dispatcher** — coordinates agents and synthesizes context packs

## Installation

Requires [uv](https://github.com/astral-sh/uv) and Docker (for Qdrant).

```bash
git clone https://github.com/yourusername/hammy.git
cd hammy
uv sync --extra dev

# Start Qdrant
docker compose up -d

# Install as a CLI tool
uv tool install --editable .
```

## Quick Start

```bash
# Initialize Hammy config in your project
hammy init /path/to/your/project

# Index the codebase
cd /path/to/your/project
hammy index

# Query with the AI agent
hammy query "Where is the payment processing logic and who calls it?"

# Check what was indexed
hammy status

# Start the MCP server (for IDE / LLM client integration)
hammy serve

# Watch for file changes and re-index incrementally
hammy watch
```

## MCP Tools

When running `hammy serve`, all tools are available to any MCP client (Cursor, Claude Desktop, etc.):

| Tool | Description |
|---|---|
| `ast_query` | Parse a file and return its full symbol tree |
| `search_symbols` | Keyword search over symbol names |
| `search_code_hybrid` | BM25 + dense hybrid search with RRF fusion |
| `lookup_symbol` | Fetch full detail for a specific symbol by name |
| `find_usages` | Find all call sites for a symbol name |
| `impact_analysis` | N-hop caller/callee traversal for a symbol |
| `structural_search` | Filter by visibility, async, param count, return type, complexity |
| `hotspot_score` | Rank symbols by caller count × churn rate |
| `pr_diff` | Parse a unified diff and compute blast radius of each changed symbol |
| `find_bridges` | Find cross-language endpoint connections |
| `list_files` | List indexed files with node counts |
| `index_status` | Overview of total nodes, edges, and languages |
| `git_log` | Recent commit history |
| `git_blame` | Blame for a file |
| `file_churn` | Commit frequency per file over a time window |
| `search_commits` | Semantic search over commit messages |
| `store_context` | Save agent context to memory (requires Qdrant) |
| `recall_context` | Retrieve relevant saved context |
| `list_context` | List all stored memory entries |

## Configuration

```yaml
# config/hammy.yaml
project:
  name: "my-project"
  root: "."

parsing:
  languages:
    - php
    - javascript
    - typescript
    - python
    - go
  max_file_size_kb: 500

qdrant:
  host: "localhost"
  port: 6333
  embedding_model: "all-MiniLM-L6-v2"

vcs:
  max_commits: 5000
  churn_window_days: 90
```

`.hammyignore` accepts standard gitignore syntax and is merged with `.gitignore`/`.hgignore`.

## Use Cases

- *"Where is `getRenew` called and what breaks if I change it?"* → `find_usages` + `impact_analysis`
- *"What are the riskiest files to touch in this PR?"* → `pr_diff` with blast radius + hotspot scores
- *"What endpoints does this React component call?"* → `find_bridges` maps fetch → PHP Route
- *"Who owns the payment logic and when was it last changed?"* → `git_blame` + `file_churn`
- *"Find all async functions that take more than 3 parameters"* → `structural_search`

## Project Structure

```
hammy/
├── src/hammy/
│   ├── cli.py              # Typer CLI (init, index, query, status, serve, watch)
│   ├── config.py           # Pydantic settings loader
│   ├── ignore.py           # Four-layer ignore system
│   ├── watcher.py          # watchfiles-based incremental re-indexer
│   ├── agents/             # CrewAI Explorer and Historian agents
│   ├── core/               # Crew orchestration and context pack generation
│   ├── indexer/            # File walking, parsing pipeline, incremental indexing
│   ├── mcp/                # MCP server (mcp-python)
│   ├── schema/             # Pydantic models (Node, Edge, ContextPack)
│   └── tools/
│       ├── languages/      # Tree-sitter extractors: php, js, ts, python, go
│       ├── ast_tools.py    # AST query tool
│       ├── bridge.py       # Cross-language bridge resolver
│       ├── diff_analysis.py# PR diff parser and blast radius
│       ├── hotspot.py      # Hotspot scoring
│       ├── hybrid_search.py# BM25 + dense RRF fusion
│       ├── parser.py       # ParserFactory dispatcher
│       ├── qdrant_tools.py # Qdrant embed, upsert, search, delete
│       └── vcs.py          # Git/Mercurial wrapper
├── config/                 # Default configuration files
├── tests/                  # 302 passing tests with fixtures
└── docker-compose.yml      # Qdrant service
```

## Development

```bash
# Run the full test suite (Qdrant must be running)
uv run pytest

# Run with coverage
uv run pytest --cov=hammy --cov-report=term-missing

# Run a specific module
uv run pytest tests/test_parser.py
```

## Requirements

- Python 3.11+
- Docker (for Qdrant)
- Git (for VCS analysis)
- LLM API key (OpenAI, Anthropic, or any LiteLLM-compatible provider) for agent queries and enrichment

## Built With

- [Tree-sitter](https://tree-sitter.github.io/) — incremental, multi-language AST parsing
- [Qdrant](https://qdrant.tech/) — vector database for semantic search and memory
- [CrewAI](https://github.com/joaomdmoura/crewAI) — multi-agent orchestration
- [mcp-python](https://github.com/modelcontextprotocol/python-sdk) — Model Context Protocol server
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25Plus for hybrid search
- [watchfiles](https://github.com/samuelcolvin/watchfiles) — fast filesystem watching
- [uv](https://github.com/astral-sh/uv) — Python package management
