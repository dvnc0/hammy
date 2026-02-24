"""Commit indexing pipeline — ingests VCS commit history into Qdrant."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hammy.config import HammyConfig
from hammy.tools.qdrant_tools import QdrantManager
from hammy.tools.vcs import VCSWrapper


@dataclass
class CommitIndexResult:
    """Results from a commit indexing run."""

    commits_processed: int = 0
    commits_indexed: int = 0


def index_commits(
    config: HammyConfig,
    *,
    qdrant: QdrantManager | None = None,
) -> CommitIndexResult:
    """Index VCS commit history into Qdrant.

    Args:
        config: Hammy configuration.
        qdrant: Optional QdrantManager instance.

    Returns:
        CommitIndexResult with stats.
    """
    project_root = Path(config.project.root).resolve()
    vcs = VCSWrapper(project_root)

    commits = vcs.log(limit=config.vcs.max_commits)
    result = CommitIndexResult(commits_processed=len(commits))

    if not commits:
        return result

    commit_dicts = [
        {
            "revision": c.revision,
            "author": c.author,
            "date": c.date.isoformat(),
            "message": c.message,
            "files_changed": c.files_changed,
        }
        for c in commits
    ]

    if qdrant is None:
        qdrant = QdrantManager(config.qdrant, project_name=config.project.name)

    qdrant.ensure_collections()
    result.commits_indexed = qdrant.upsert_commits(commit_dicts)

    return result
