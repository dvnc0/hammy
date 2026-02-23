# Hammy

> Codebase Intelligence Engine — deep structural and historical context for AI coding agents

<p align="center">
  <img src="hammy_resized.jpg" alt="Hammy">
</p>

Hammy gives AI coding agents a high-fidelity map of your codebase. It parses source code into a queryable call graph, tracks version control history, and exposes everything through an MCP server — so agents spend less time guessing and more time doing.

**Best on:** legacy monoliths, team codebases, anything where "just grep it" stops working.

---

## Why Hammy?

Most coding agents navigate by reading files and hoping for the best. Hammy gives them:

- **A call graph** — not just "this function exists" but "these 12 things call it, and here's what it calls"
- **Risk signals** — which code is high-churn AND heavily depended on (the landmines)
- **Blast radius** — before changing anything, know exactly what breaks
- **Orientation** — understand a 200-file module in one call instead of reading every file

---

## Features

- **Multi-language AST parsing** — PHP, JavaScript, TypeScript, Python, Go (Tree-sitter based)
- **Full call expression indexing** — stores `$this->resolve(PaymentService::class)` not just `resolve`, enabling argument-level filtering
- **Cross-language bridge detection** — links `fetch('/api/users')` in JS to the backend handler
- **Semantic + hybrid search** — dense vector search (Qdrant) + BM25, merged via Reciprocal Rank Fusion
- **Structural search** — filter by visibility, async, param count, return type, complexity score
- **Impact analysis** — N-hop call graph traversal: know the full blast radius before touching anything
- **PR diff analysis** — parse a unified diff (or auto-diff uncommitted changes), get HIGH/MED/LOW risk per changed symbol
- **Hotspot scoring** — `log(callers) × log(churn)`: surfaces code that's both heavily depended on AND frequently modified
- **LLM enrichment** — auto-generate plain-English summaries for every indexed function and class
- **Brain / memory layer** — agents store and recall research findings across sessions (invaluable on large codebases)
- **Watch mode** — incremental re-indexing on file change
- **VCS integration** — Git log, blame, churn; Mercurial scaffolded
- **MCP server** — all tools available to any MCP client (Cursor, Claude Desktop, VS Code, etc.)
- **Smart ignore** — four-layer filtering: defaults → .gitignore → .hgignore → .hammyignore

---

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

---

## Quick Start

```bash
# Point Hammy at your project
hammy init /path/to/your/project

# Index the codebase
cd /path/to/your/project
hammy index

# Start the MCP server (connect from your IDE)
hammy serve

# Query with the AI agent directly
hammy query "Where is the payment processing logic and who calls it?"

# Watch for changes and re-index incrementally
hammy watch
```

---

## MCP Tools

All tools are available via `hammy serve` to any MCP client. Grouped by what you're trying to do:

### Orientation — understand unfamiliar code fast

#### `explain_symbol` ⭐
**The single most useful tool.** One call returns everything about a symbol: full definition (file, line, params, return type, visibility, async, LLM summary), direct callers, direct callees, sibling symbols in the same file, and recent commits. Replaces `lookup_symbol` + `find_usages` + `impact_analysis` + `ast_query` in a single round trip.

```
explain_symbol("PaymentService")
→ definition, 8 callers, 3 callees, 12 siblings, last 5 commits
```

#### `module_summary`
**Orient yourself on a directory without opening a single file.** Groups all symbols under a path into a structured table of contents — classes with nested methods first, then functions — across every file in the directory. Use this before diving into an unfamiliar module.

```
module_summary("app/Services/Payment/")
→ 3 files, 47 symbols, organized by class hierarchy
```

#### `ast_query`
Parse any file and see its full symbol tree: every class, method, function, endpoint, and import with line numbers, visibility, and LLM summaries. Filter by type (`classes`, `functions`, `methods`, `endpoints`, `imports`).

#### `list_files`
List every indexed file with its language. Good first call on an unfamiliar project to understand scope before searching.

---

### Search — find what you're looking for

#### `search_symbols`
Keyword search over symbol names, ranked by match quality (exact → prefix → substring → summary). Use when you know roughly what you're looking for but not the exact name.

#### `lookup_symbol`
You know the exact name — get the full definition immediately: file, line range, params, return type, visibility, async flag, and LLM summary. Falls back to word-boundary match if no exact hit.

#### `lookup_symbols_batch`
Look up multiple symbols in one call. Pass a comma-separated list, get all definitions back at once. Eliminates the `lookup_symbol` loop after a search result.

```
lookup_symbols_batch("UserController, PaymentService, getRenew")
→ 3 full definitions in one call
```

#### `search_code_hybrid`
Combines BM25 (exact identifiers) with semantic embeddings (conceptual matches), merged via RRF. Use when your query mixes exact terms and concepts: `"sendPersonalInvite email logic"`. Requires Qdrant.

#### `structural_search`
Find symbols by shape, not name. Useful for refactoring sprints and code reviews.

```
structural_search(node_type="method", visibility="public", min_params=4)
→ all public methods with 4+ parameters

structural_search(min_complexity=15, file_filter="Services/")
→ high-complexity methods in the services layer
```

Parameters: `node_type`, `language`, `visibility`, `async_only`, `min_params`, `max_params`, `return_type`, `name_pattern`, `file_filter`, `min_complexity`, `limit`

---

### Call Graph — trace dependencies

#### `find_usages`
Find every call site for a function or method. Word-boundary matched so `save` won't match `saveAll`. Now with `argument_filter` to narrow by what's passed in — critical for dependency-injection heavy codebases.

```
find_usages("resolve", argument_filter="PaymentService")
→ only calls to resolve() that pass PaymentService, not the other 40
```

Parameters: `symbol_name`, `file_filter`, `argument_filter`

#### `impact_analysis`
**"If I change this, what breaks?"** Traverses the call graph N hops deep. Use `direction="callers"` before any refactor to map the full dependency chain. Use `direction="callees"` to see what a function depends on. Use `direction="both"` for the full neighbourhood.

```
impact_analysis("charge", depth=3, direction="callers")
→ everything downstream that will break
```

#### `find_bridges`
Finds cross-language endpoint connections — e.g. links a `fetch('/api/v1/users')` in React to the backend route handler. Useful when tracing frontend→backend flows.

---

### Risk — know before you touch

#### `hotspot_score` ⭐
**Mandatory pre-work before any significant change.** Scores each symbol by `log(callers) × log(churn)`. High score = heavily depended on AND frequently modified = highest risk. Near zero = safe to change. Run this before touching any unfamiliar code.

```
hotspot_score(file_filter="app/Services/", top_n=10)
→ ranked list of landmines in the services layer
```

#### `pr_diff`
**"What's the risk of this PR?"** Parses a diff, identifies every changed symbol, and rates each one LOW/MED/HIGH based on caller count. Accepts raw diff text, a base ref, or `working_tree=True` to automatically diff your uncommitted changes.

```
pr_diff(working_tree=True)              # analyse uncommitted changes
pr_diff(base_ref="main")               # compare branch against main
pr_diff(diff_text="<paste from GitHub>") # analyse a PR diff
```

---

### VCS History — understand context and ownership

#### `git_log`
Recent commit history for a file or the whole repo. Shows what changed, when, and by whom.

#### `git_blame`
Line-by-line authorship. Use when you need to understand intent, know who to ask about a tricky section, or check whether a suspicious line is recent or ancient.

#### `file_churn`
Commit frequency per file over a time window. High churn = actively changing or repeatedly fixed. Run before diving into a module to know whether you're on stable ground or in a churn zone.

---

### Semantic Memory — retain research across sessions *(requires Qdrant)*

#### `store_context`
Save a research finding to persistent memory with a key, tags, and source files. Sub-agents and future sessions can retrieve it instantly instead of re-researching.

```
store_context(
  key="auth-flow-research",
  content="Authentication touches 40 files. Entry point is AuthController::login...",
  tags="auth,sprint-42"
)
```

#### `recall_context`
Retrieve stored research by exact key or semantic query.

#### `list_context`
List all stored memory entries with their tags and timestamps.

---

### Housekeeping

#### `index_status`
Quick orientation: total symbols, files, edges, and languages indexed. Call first on an unfamiliar project, or to confirm the index is populated before searching.

#### `reindex`
Refresh the in-memory symbol index after editing files. Pass `update_qdrant=true` to also refresh semantic embeddings. Pass `enrich=true` to generate LLM summaries for new symbols.

---

## Common Workflows

**Before touching unfamiliar code:**
```
hotspot_score(file_filter="app/Services/Payment/")   # find the landmines
explain_symbol("PaymentService")                      # understand the entry point
impact_analysis("charge", depth=3)                    # map the blast radius
```

**Exploring an unfamiliar module:**
```
module_summary("app/Services/Payment/")              # table of contents
lookup_symbols_batch("PaymentService, Webhook, StripeClient")  # drill into key symbols
```

**Before merging a PR:**
```
pr_diff(working_tree=True)                           # risk-rate your changes
pr_diff(base_ref="main")                             # or compare against main
```

**Tracking DI dependencies:**
```
find_usages("resolve", argument_filter="PaymentService")  # who injects PaymentService
find_usages("make", argument_filter="UserRepository")     # who creates UserRepository
```

**Planning a refactoring sprint:**
```
structural_search(min_complexity=15, node_type="method")  # find complexity hotspots
structural_search(min_params=5, visibility="public")      # candidates for parameter objects
```

---

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

---

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
├── tests/                  # 317 passing tests with fixtures
└── docker-compose.yml      # Qdrant service
```

---

## Development

```bash
# Run the full test suite
uv run pytest

# Run with coverage
uv run pytest --cov=hammy --cov-report=term-missing

# Run a specific module
uv run pytest tests/test_parser.py
```

---

## Requirements

- Python 3.11+
- Docker (for Qdrant)
- Git (for VCS analysis)
- LLM API key (OpenAI, Anthropic, or any LiteLLM-compatible provider) for agent queries and enrichment

---

## Built With

- [Tree-sitter](https://tree-sitter.github.io/) — incremental, multi-language AST parsing
- [Qdrant](https://qdrant.tech/) — vector database for semantic search and memory
- [CrewAI](https://github.com/joaomdmoura/crewAI) — multi-agent orchestration
- [mcp-python](https://github.com/modelcontextprotocol/python-sdk) — Model Context Protocol server
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25Plus for hybrid search
- [watchfiles](https://github.com/samuelcolvin/watchfiles) — fast filesystem watching
- [uv](https://github.com/astral-sh/uv) — Python package management
