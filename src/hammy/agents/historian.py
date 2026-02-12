"""Historian agent tools — VCS history, blame, churn analysis.

These are CrewAI @tool functions that wrap the VCS wrapper and
Qdrant commit search for use by the Historian agent.
"""

from __future__ import annotations

from crewai.tools import tool

from hammy.tools.qdrant_tools import QdrantManager
from hammy.tools.vcs import VCSWrapper


def make_historian_tools(
    vcs: VCSWrapper,
    qdrant: QdrantManager | None = None,
) -> list:
    """Create Historian agent tools bound to the current VCS context."""

    @tool("Git Log")
    def vcs_log(file_path: str = "", limit: int = 20) -> str:
        """Get commit history for a file or the entire repository.

        Args:
            file_path: Optional path to filter commits (empty for all).
            limit: Maximum number of commits to return.
        """
        path = file_path if file_path else None
        commits = vcs.log(path=path, limit=limit)

        if not commits:
            return "No commits found."

        lines = []
        for c in commits:
            date = c.date.strftime("%Y-%m-%d")
            files = ", ".join(c.files_changed[:5])
            if len(c.files_changed) > 5:
                files += f" (+{len(c.files_changed) - 5} more)"
            lines.append(
                f"[{c.revision[:8]}] {date} by {c.author}: {c.message}"
            )
            if files:
                lines.append(f"  files: {files}")

        return "\n".join(lines)

    @tool("Git Blame")
    def vcs_blame(file_path: str) -> str:
        """Get line-by-line authorship for a file. Shows who last modified each line.

        Args:
            file_path: Path to the file to blame.
        """
        try:
            blame_lines = vcs.blame(file_path)
        except RuntimeError as e:
            return f"Error: {e}"

        if not blame_lines:
            return f"No blame data for {file_path}."

        lines = []
        for bl in blame_lines:
            lines.append(f"L{bl.line_number:4d} | {bl.revision} | {bl.author:15s} | {bl.content}")

        return "\n".join(lines)

    @tool("File Churn Analysis")
    def vcs_churn(window_days: int = 90) -> str:
        """Analyze which files change most frequently (high churn = potential hotspots).

        Args:
            window_days: How many days back to analyze (default: 90).
        """
        churn = vcs.churn(window_days=window_days)

        if not churn:
            return "No changes found in the specified window."

        lines = [f"File churn in last {window_days} days:\n"]
        for file_path, count in list(churn.items())[:30]:
            bar = "█" * min(count, 20)
            lines.append(f"  {count:4d} changes | {bar} | {file_path}")

        return "\n".join(lines)

    @tool("Search Commit History")
    def search_commits(query: str, limit: int = 10) -> str:
        """Semantic search through commit messages. Finds commits related to a topic.

        Args:
            query: Natural language description of what you're looking for.
            limit: Maximum results to return.
        """
        if qdrant is None:
            return "Commit search not available (Qdrant not configured)."

        results = qdrant.search_commits(query, limit=limit)

        if not results:
            return f"No commits matching '{query}' found."

        lines = []
        for r in results:
            score = r.get("score", 0)
            lines.append(
                f"[{r['revision'][:8]}] (relevance: {score:.2f}) "
                f"by {r['author']}: {r['message']}"
            )
            files = r.get("files_changed", [])
            if files:
                lines.append(f"  files: {', '.join(files[:5])}")

        return "\n".join(lines)

    return [vcs_log, vcs_blame, vcs_churn, search_commits]
