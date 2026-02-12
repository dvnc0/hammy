# Hammy

> A Codebase Intelligence Engine for AI coding agents

**STATUS: Work in Progress**

Hammy is a specialized intelligence asset designed to provide deep structural and historical context about codebases to AI coding agents. It operates as a high-fidelity "brain" that external tools (IDE extensions, CLIs, MCP clients) can query to understand the *How*, *Where*, and *Why* of complex, multi-language codebases.

## Overview

Hammy uses a multi-agent architecture built on CrewAI to analyze codebases through multiple lenses:

- **The Explorer**: Maps code structure using Tree-sitter AST parsing, tracking dependencies and cross-language bridges
- **The Historian**: Analyzes version control history (Git/Mercurial) to provide temporal context, authorship, and change patterns
- **The Dispatcher**: Coordinates agents, breaks down queries, and synthesizes comprehensive "context packs"

### Key Features

- **Multi-language AST parsing**: Currently supports PHP and JavaScript with extensible architecture for additional languages
- **Cross-language bridge detection**: Links frontend API calls to backend endpoints across language boundaries
- **Version control intelligence**: Extracts commit history, blame information, and code churn metrics
- **Vector search**: Semantic code search powered by Qdrant vector database
- **Smart ignore system**: Four-layer ignore system (defaults, .gitignore, .hgignore, .hammyignore) prevents indexing irrelevant files
- **CLI interface**: Simple commands to index codebases and query for insights

## Architecture

Hammy represents codebases as a property graph using a Universal JSON Schema (UJS):

- **Nodes**: Represent code entities (files, classes, functions, interfaces)
- **Edges**: Represent relationships (calls, imports, bridges between languages)
- **Metadata**: Includes line numbers, complexity scores, and historical heat

This unified representation enables AI agents to "walk" between a JavaScript frontend and PHP backend as if they were a single codebase.

## Installation

Hammy uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Clone the repository
git clone https://github.com/yourusername/hammy.git
cd hammy

# Install dependencies
uv sync --extra dev

# Start Qdrant vector database
docker compose up -d
```

## Quick Start

```bash
# Initialize Hammy in your project
hammy init /path/to/your/project

# Index your codebase
cd /path/to/your/project
hammy index

# Query the codebase
hammy query "What does the UserController do?"

# Check index status
hammy status
```

## Configuration

Hammy uses YAML configuration files located in `config/`:

- **hammy.yaml**: Project settings, parsing options, Qdrant connection, VCS limits
- **agents.yaml**: CrewAI agent definitions (role, goal, backstory, LLM provider)
- **tasks.yaml**: Agent task definitions and workflows
- **.hammyignore**: Custom ignore patterns (gitignore syntax)

### Example Configuration

```yaml
# config/hammy.yaml
project:
  name: "my-project"
  root: "."

parsing:
  languages:
    - php
    - javascript
  max_file_size_kb: 500

qdrant:
  host: "localhost"
  port: 6333
  embedding_model: "all-MiniLM-L6-v2"

vcs:
  max_commits: 5000
  churn_window_days: 90
```

## Current Status

### Completed (Phases 0-6)

- Project scaffolding with uv package management
- Four-layer ignore system with .gitignore/.hgignore support
- Tree-sitter based AST parsing for PHP and JavaScript
- VCS integration (Git fully implemented, Mercurial scaffolded)
- Qdrant vector database integration
- CrewAI agent system with Explorer and Historian
- Cross-language bridge detection
- CLI interface with init, index, query, and status commands
- 106 passing tests

### Planned (Phase 7+)

- Model Context Protocol (MCP) server implementation
- Additional language support (Python, TypeScript, Java, etc.)
- Enhanced bridge detection strategies
- IDE extension integrations
- Performance optimizations for large codebases
- Context pack caching and incremental updates

## Development

### Running Tests

```bash
# Ensure Qdrant is running
docker compose up -d

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=hammy --cov-report=term-missing

# Run specific test file
uv run pytest tests/test_parser.py
```

### Project Structure

```
hammy/
├── src/hammy/
│   ├── cli.py              # Typer CLI interface
│   ├── config.py           # Pydantic Settings config loader
│   ├── ignore.py           # Multi-layer ignore system
│   ├── agents/             # CrewAI agent definitions
│   ├── core/               # Crew orchestration and context packs
│   ├── indexer/            # File walking and indexing pipelines
│   ├── schema/             # Pydantic models (Node, Edge, ContextPack)
│   └── tools/              # Tree-sitter, VCS, Qdrant, Bridge tools
├── config/                 # Default configuration files
├── tests/                  # Test suite with fixtures
└── docker-compose.yml      # Qdrant service definition
```

## How It Works

1. **Indexing**: Hammy walks your codebase respecting ignore patterns, parses files using Tree-sitter, and stores structured representations in Qdrant
2. **VCS Analysis**: Commits, blame info, and churn metrics are extracted and indexed for temporal context
3. **Bridge Detection**: Cross-language connections are identified (e.g., `fetch('/api/users')` linked to PHP `#[Route('/api/users')]`)
4. **Query Processing**: The Dispatcher coordinates Explorer and Historian agents to answer questions about your codebase
5. **Context Pack Generation**: Results are synthesized into Markdown documents optimized for LLM consumption

## Use Cases

- "Why does the user profile load so slowly?" → Finds the bottleneck and its change history
- "What endpoints does this React component call?" → Maps frontend-to-backend connections
- "Who owns the payment processing logic?" → Provides authorship and change patterns
- "Where is the User type defined?" → Finds definitions across multiple languages

## Requirements

- Python 3.11+
- Docker (for Qdrant)
- Git (for repository analysis)
- LLM API access (configurable: OpenAI, Anthropic, etc.)

## Contributing

Hammy is in active development. Contributions, issues, and feature requests are welcome!

## Acknowledgments

Built with:
- [CrewAI](https://github.com/joaomdmoura/crewAI) - Multi-agent orchestration
- [Tree-sitter](https://tree-sitter.github.io/) - Incremental parsing
- [Qdrant](https://qdrant.tech/) - Vector database
- [uv](https://github.com/astral-sh/uv) - Python package management
