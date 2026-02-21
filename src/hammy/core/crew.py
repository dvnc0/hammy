"""CrewAI crew definition for Hammy.

Wires together the Dispatcher, Historian, and Explorer agents with their
tools and defines the query workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from crewai import Agent, Crew, Process, Task

from hammy.agents.explorer import make_explorer_tools
from hammy.agents.historian import make_historian_tools
from hammy.config import HammyConfig
from hammy.core.context_pack import generate_context_pack_markdown
from hammy.schema.models import ContextPack, Edge, Node
from hammy.tools.bridge import resolve_bridges
from hammy.tools.parser import ParserFactory
from hammy.tools.qdrant_tools import QdrantManager
from hammy.tools.vcs import VCSWrapper


class HammyCrew:
    """Manages the CrewAI agent setup and query execution."""

    def __init__(
        self,
        config: HammyConfig,
        nodes: list[Node],
        edges: list[Edge],
        *,
        qdrant: QdrantManager | None = None,
        vcs: VCSWrapper | None = None,
    ):
        self.config = config
        self.nodes = nodes
        self.edges = edges
        self.project_root = Path(config.project.root).resolve()

        # Load agent configs
        agents_config = self._load_agents_config()

        # Set up tools
        parser_factory = ParserFactory(config.parsing.languages)

        if vcs is None:
            try:
                vcs = VCSWrapper(self.project_root)
            except ValueError:
                vcs = None

        explorer_tools = make_explorer_tools(
            self.project_root, parser_factory, nodes, edges, qdrant
        )

        historian_tools = []
        if vcs is not None:
            historian_tools = make_historian_tools(vcs, qdrant)

        # Create agents
        explorer_cfg = agents_config.get("explorer", {})
        self.explorer = Agent(
            role=explorer_cfg.get("role", "Codebase Explorer"),
            goal=explorer_cfg.get("goal", "Map structural relationships in the codebase."),
            backstory=explorer_cfg.get("backstory", "You are an expert at code structure analysis."),
            tools=explorer_tools,
            llm=explorer_cfg.get("llm"),
            max_iter=explorer_cfg.get("max_iter", 5),
            allow_delegation=False,
            verbose=explorer_cfg.get("verbose", False),
        )

        historian_cfg = agents_config.get("historian", {})
        self.historian = Agent(
            role=historian_cfg.get("role", "Codebase Historian"),
            goal=historian_cfg.get("goal", "Provide temporal context about code changes."),
            backstory=historian_cfg.get("backstory", "You are an expert at reading VCS history."),
            tools=historian_tools,
            llm=historian_cfg.get("llm"),
            max_iter=historian_cfg.get("max_iter", 5),
            allow_delegation=False,
            verbose=historian_cfg.get("verbose", False),
        )

        dispatcher_cfg = agents_config.get("dispatcher", {})
        self.dispatcher = Agent(
            role=dispatcher_cfg.get("role", "Codebase Intelligence Dispatcher"),
            goal=dispatcher_cfg.get("goal", "Analyze queries and synthesize results from Explorer and Historian."),
            backstory=dispatcher_cfg.get("backstory", "You are an expert at understanding developer needs."),
            tools=[],
            llm=dispatcher_cfg.get("llm"),
            max_iter=dispatcher_cfg.get("max_iter", 10),
            allow_delegation=False,
            verbose=dispatcher_cfg.get("verbose", True),
        )

    def query(self, question: str) -> str:
        """Run a query through the crew and return a context pack as Markdown.

        Args:
            question: Natural language question about the codebase.

        Returns:
            Markdown-formatted context pack.
        """
        # Task 1: Explorer analyzes code structure
        explore_task = Task(
            description=(
                f"Find code entities related to: '{question}'\n\n"
                "Use AST Query to find relevant files, classes, and methods. "
                "List what you find with file paths and line numbers. "
                "Keep it factual - just report what exists in the code."
            ),
            expected_output=(
                "A list of relevant code entities with their locations and basic descriptions."
            ),
            agent=self.explorer,
        )

        # Task 2: Historian analyzes VCS history (only if VCS tools are available)
        tasks = [explore_task]
        agents = [self.explorer]
        
        if self.historian.tools:  # Only add historian if VCS tools are available
            history_task = Task(
                description=(
                    f"Research VCS history related to: '{question}'\n\n"
                    "Based on the Explorer's findings, use Git Log and File Churn tools "
                    "to check the commit history. Report who changed these files, when, "
                    "and how often they change. Keep it brief."
                ),
                expected_output=(
                    "A summary of relevant commits, authors, and change frequency."
                ),
                agent=self.historian,
            )
            tasks.append(history_task)
            agents.append(self.historian)

            # Task 3: Dispatcher synthesizes findings
            synthesize_task = Task(
                description=(
                    f"Answer: '{question}'\n\n"
                    "Combine the Explorer's code structure findings with the Historian's "
                    "change history. Provide a complete answer with both what the code does "
                    "and its evolution over time. Highlight any concerns like high churn."
                ),
                expected_output=(
                    "A comprehensive answer combining code structure and history."
                ),
                agent=self.dispatcher,
                context=[explore_task, history_task],
            )
            tasks.append(synthesize_task)
            agents.append(self.dispatcher)

        # Create and run crew
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        return str(result)

    def _load_agents_config(self) -> dict[str, Any]:
        """Load agent configuration from agents.yaml."""
        config_path = self.project_root / "config" / "agents.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f) or {}

        # Also check relative to the hammy package
        package_config = Path(__file__).parent.parent.parent.parent / "config" / "agents.yaml"
        if package_config.exists():
            with open(package_config) as f:
                return yaml.safe_load(f) or {}

        return {}
