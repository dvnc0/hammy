"""Hammy CLI — command-line interface for codebase intelligence."""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="hammy",
    help="Hammy: Codebase Intelligence Engine",
    no_args_is_help=True,
)
console = Console()


@app.command()
def init(
    path: Path = typer.Argument(
        Path("."),
        help="Project root directory to initialize.",
    ),
) -> None:
    """Initialize Hammy configuration in a project directory."""
    path = path.resolve()

    config_dir = path / "config"
    config_dir.mkdir(exist_ok=True)

    # Copy default config files
    package_config = Path(__file__).parent.parent.parent / "config"

    for filename in ("hammy.yaml", "agents.yaml", "tasks.yaml"):
        src = package_config / filename
        dest = config_dir / filename
        if dest.exists():
            console.print(f"  [yellow]exists[/yellow]  {dest.relative_to(path)}")
        elif src.exists():
            shutil.copy2(src, dest)
            console.print(f"  [green]created[/green] {dest.relative_to(path)}")
        else:
            console.print(f"  [red]missing[/red] template: {filename}")

    # Create .hammyignore if it doesn't exist
    hammyignore = path / ".hammyignore"
    if hammyignore.exists():
        console.print(f"  [yellow]exists[/yellow]  .hammyignore")
    else:
        hammyignore.write_text(
            "# Hammy custom ignore patterns\n"
            "# Uses .gitignore syntax\n"
            "*.min.js\n"
            "*.min.css\n"
            "*.map\n"
            "dist/\n"
            "build/\n"
        )
        console.print(f"  [green]created[/green] .hammyignore")

    console.print(f"\n[bold green]Hammy initialized in {path}[/bold green]")
    console.print("Edit config/agents.yaml to set your LLM provider, then run: hammy index")


@app.command()
def index(
    path: Path = typer.Argument(
        Path("."),
        help="Project root directory to index.",
    ),
    no_qdrant: bool = typer.Option(
        False,
        "--no-qdrant",
        help="Parse only, don't store in Qdrant.",
    ),
    no_commits: bool = typer.Option(
        False,
        "--no-commits",
        help="Skip commit history indexing.",
    ),
) -> None:
    """Index a codebase — parse files, extract symbols, store in Qdrant."""
    from dotenv import load_dotenv

    from hammy.config import HammyConfig
    from hammy.indexer.code_indexer import index_codebase
    from hammy.indexer.commit_indexer import index_commits
    from hammy.tools.qdrant_tools import QdrantManager

    path = path.resolve()
    load_dotenv(path / ".env")
    config = HammyConfig.load(path)

    qdrant = None
    if not no_qdrant:
        try:
            qdrant = QdrantManager(config.qdrant)
            qdrant.ensure_collections()
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Qdrant not available ({e})")
            console.print("Continuing without Qdrant. Use --no-qdrant to suppress this warning.")
            qdrant = None

    with console.status("[bold blue]Indexing codebase..."):
        result, nodes, edges = index_codebase(
            config,
            qdrant=qdrant,
            store_in_qdrant=qdrant is not None,
        )

    # Display results
    table = Table(title="Code Indexing Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Files processed", str(result.files_processed))
    table.add_row("Files skipped", str(result.files_skipped))
    table.add_row("Nodes extracted", str(result.nodes_extracted))
    table.add_row("Edges extracted", str(result.edges_extracted))
    table.add_row("Nodes indexed (Qdrant)", str(result.nodes_indexed))
    console.print(table)

    if result.errors:
        console.print(f"\n[yellow]Errors ({len(result.errors)}):[/yellow]")
        for err in result.errors[:10]:
            console.print(f"  - {err}")

    # Index commits
    if not no_commits:
        try:
            with console.status("[bold blue]Indexing commit history..."):
                commit_result = index_commits(config, qdrant=qdrant)

            console.print(f"\nCommits: {commit_result.commits_processed} processed, "
                         f"{commit_result.commits_indexed} indexed")
        except Exception as e:
            console.print(f"[yellow]Commit indexing skipped:[/yellow] {e}")

    # Show bridge summary
    from hammy.tools.bridge import resolve_bridges
    bridges = resolve_bridges(nodes, edges)
    if bridges:
        console.print(f"\n[bold]Cross-language bridges found: {len(bridges)}[/bold]")
        for b in bridges[:5]:
            console.print(f"  - {b.metadata.context} ({b.metadata.confidence:.0%})")


@app.command()
def query(
    question: str = typer.Argument(
        ...,
        help="Natural language question about the codebase.",
    ),
    path: Path = typer.Option(
        Path("."),
        "--path", "-p",
        help="Project root directory.",
    ),
) -> None:
    """Query the codebase using AI agents."""
    from dotenv import load_dotenv

    from hammy.config import HammyConfig
    from hammy.core.crew import HammyCrew
    from hammy.indexer.code_indexer import index_codebase
    from hammy.tools.qdrant_tools import QdrantManager

    path = path.resolve()

    # Load .env from project root so API keys are available
    load_dotenv(path / ".env")

    config = HammyConfig.load(path)

    # Check that LLM is configured
    agents_yaml = path / "config" / "agents.yaml"
    if agents_yaml.exists():
        import yaml
        with open(agents_yaml) as f:
            agents_config = yaml.safe_load(f) or {}
        for agent_name, agent_cfg in agents_config.items():
            if agent_cfg.get("llm") is None:
                console.print(
                    f"[red]Error:[/red] No LLM configured for '{agent_name}' agent.\n"
                    f"Edit config/agents.yaml and set the 'llm' field."
                )
                raise typer.Exit(1)

    # Quick index (parse only, no Qdrant storage)
    with console.status("[bold blue]Parsing codebase..."):
        _, nodes, edges = index_codebase(config, store_in_qdrant=False)

    console.print(f"Parsed {len(nodes)} symbols from codebase.\n")

    # Set up Qdrant if available
    qdrant = None
    try:
        qdrant = QdrantManager(config.qdrant)
    except Exception:
        pass

    # Create crew with full context (no filtering)
    try:
        crew = HammyCrew(config, nodes, edges, qdrant=qdrant)
        
        with console.status("[bold blue]Analyzing..."):
            result = crew.query(question)
        
        console.print(result)
    except Exception as e:
        console.print(f"[red]Crew analysis failed:[/red] {e}\n")
        console.print("[yellow]Falling back to simple search...[/yellow]\n")
        
        # Simple fallback: keyword search over parsed nodes
        keywords = question.lower().split()
        matched = [
            n for n in nodes
            if any(kw in n.name.lower() or (n.summary and kw in n.summary.lower()) for kw in keywords)
        ]
        if not matched:
            matched = nodes[:20]

        console.print("[bold]Relevant code entities found:[/bold]\n")
        for node in matched[:20]:
            console.print(f"  • {node.type.value}: [cyan]{node.name}[/cyan]")
            console.print(f"    Location: {node.loc.file}:{node.loc.lines[0]}-{node.loc.lines[1]}")
            if node.summary:
                console.print(f"    {node.summary}")
            console.print()


@app.command()
def status(
    path: Path = typer.Argument(
        Path("."),
        help="Project root directory.",
    ),
) -> None:
    """Show Hammy index status and statistics."""
    from hammy.config import HammyConfig
    from hammy.tools.qdrant_tools import QdrantManager

    path = path.resolve()
    config = HammyConfig.load(path)

    table = Table(title="Hammy Status")
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("Project root", config.project.root)
    table.add_row("Languages", ", ".join(config.parsing.languages))
    table.add_row("Max file size", f"{config.parsing.max_file_size_kb} KB")
    table.add_row("Qdrant", f"{config.qdrant.host}:{config.qdrant.port}")
    table.add_row("Embedding model", config.qdrant.embedding_model)
    console.print(table)

    # Check Qdrant
    try:
        qdrant = QdrantManager(config.qdrant)
        stats = qdrant.get_stats()

        qtable = Table(title="Qdrant Collections")
        qtable.add_column("Collection", style="bold")
        qtable.add_column("Points", justify="right")
        for name, count in stats.items():
            qtable.add_row(name, str(count))
        console.print(qtable)
    except Exception as e:
        console.print(f"[yellow]Qdrant not available:[/yellow] {e}")

    # Check VCS
    from hammy.tools.vcs import VCSWrapper
    try:
        vcs = VCSWrapper(Path(config.project.root))
        console.print(f"VCS: {vcs.vcs_type.value} detected")
    except ValueError:
        console.print("[yellow]No VCS detected[/yellow]")


@app.command()
def serve(
    path: Path = typer.Argument(
        Path("."),
        help="Project root directory.",
    ),
    transport: str = typer.Option(
        "stdio",
        "--transport", "-t",
        help="Transport mode: 'stdio' (default) or 'sse'.",
    ),
) -> None:
    """Start the Hammy MCP server for AI tool integration."""
    from dotenv import load_dotenv

    from hammy.config import HammyConfig
    from hammy.mcp.server import create_mcp_server

    path = path.resolve()
    load_dotenv(path / ".env")
    config = HammyConfig.load(path)

    console.print(f"[bold blue]Starting Hammy MCP server[/bold blue]")
    console.print(f"  Project: {config.project.name}")
    console.print(f"  Root: {config.project.root}")
    console.print(f"  Transport: {transport}")

    mcp_server = create_mcp_server(project_root=path, config=config)

    if transport not in ("stdio", "sse"):
        console.print(f"[red]Unknown transport: {transport}[/red]")
        raise typer.Exit(1)

    mcp_server.run(transport=transport)


if __name__ == "__main__":
    app()
