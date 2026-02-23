"""Unified VCS wrapper for Git and Mercurial.

Auto-detects the VCS type from the project root and provides a consistent
interface for log, blame, churn, and diff operations.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path


class VCSType(str, Enum):
    GIT = "git"
    MERCURIAL = "hg"


@dataclass
class CommitInfo:
    """A single commit from version control."""

    revision: str
    author: str
    date: datetime
    message: str
    files_changed: list[str] = field(default_factory=list)


@dataclass
class BlameLine:
    """A single line of blame output."""

    line_number: int
    revision: str
    author: str
    content: str


class VCSWrapper:
    """Unified interface for Git and Mercurial operations."""

    def __init__(self, project_root: Path, vcs_type: VCSType | None = None):
        self.project_root = project_root.resolve()
        self.vcs_type = vcs_type or self.detect(project_root)

    @staticmethod
    def detect(project_root: Path) -> VCSType:
        """Auto-detect which VCS is used in the project root."""
        if (project_root / ".git").exists():
            return VCSType.GIT
        if (project_root / ".hg").exists():
            return VCSType.MERCURIAL
        raise ValueError(
            f"No VCS detected in {project_root}. "
            "Expected .git/ or .hg/ directory."
        )

    def log(
        self,
        path: str | None = None,
        limit: int = 50,
    ) -> list[CommitInfo]:
        """Get commit history, optionally filtered to a specific path."""
        if self.vcs_type == VCSType.GIT:
            return self._git_log(path, limit)
        else:
            return self._hg_log(path, limit)

    def blame(self, path: str) -> list[BlameLine]:
        """Get line-by-line authorship for a file."""
        if self.vcs_type == VCSType.GIT:
            return self._git_blame(path)
        else:
            return self._hg_blame(path)

    def churn(self, path: str | None = None, window_days: int = 90) -> dict[str, int]:
        """Get change frequency per file within a time window.

        Returns a dict of {file_path: change_count}.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        commits = self.log(path, limit=10000)

        churn: dict[str, int] = {}
        for commit in commits:
            if commit.date < cutoff:
                continue
            for f in commit.files_changed:
                churn[f] = churn.get(f, 0) + 1

        return dict(sorted(churn.items(), key=lambda x: x[1], reverse=True))

    def diff(self, rev1: str, rev2: str) -> str:
        """Get the diff between two revisions."""
        if self.vcs_type == VCSType.GIT:
            return self._run(["git", "diff", rev1, rev2])
        else:
            return self._run(["hg", "diff", "-r", rev1, "-r", rev2])

    def diff_working_tree(self, base_ref: str = "HEAD") -> str:
        """Diff working tree against a ref (equivalent to git diff <ref>)."""
        if self.vcs_type == VCSType.GIT:
            return self._run(["git", "diff", base_ref])
        else:
            return self._run(["hg", "diff", "-r", base_ref])

    # --- Git Implementation ---

    def _git_log(self, path: str | None, limit: int) -> list[CommitInfo]:
        # Use a record separator to clearly delineate commits
        sep = "---HAMMY_SEP---"
        record_sep = "---HAMMY_RECORD---"
        fmt = f"{record_sep}%H{sep}%an{sep}%aI{sep}%s"

        cmd = ["git", "log", f"--format={fmt}", f"-n{limit}", "--name-only"]
        if path:
            cmd.extend(["--", path])

        output = self._run(cmd)
        if not output.strip():
            return []

        commits: list[CommitInfo] = []
        # Split by record separator to get individual commits
        records = output.split(record_sep)
        for record in records:
            record = record.strip()
            if not record:
                continue

            lines = record.split("\n")
            parts = lines[0].split(sep)
            if len(parts) < 4:
                continue

            date = datetime.fromisoformat(parts[2])
            files = [l for l in lines[1:] if l.strip()]

            commits.append(CommitInfo(
                revision=parts[0],
                author=parts[1],
                date=date,
                message=parts[3],
                files_changed=files,
            ))

        return commits

    def _git_blame(self, path: str) -> list[BlameLine]:
        output = self._run(["git", "blame", "--porcelain", path])
        if not output.strip():
            return []

        lines: list[BlameLine] = []
        current_rev = ""
        current_author = ""
        current_line_no = 0

        for line in output.split("\n"):
            if not line:
                continue

            # Porcelain format: first line of each group starts with a hash
            parts = line.split()
            if len(parts) >= 3 and len(parts[0]) == 40:
                current_rev = parts[0]
                current_line_no = int(parts[2])
            elif line.startswith("author "):
                current_author = line[7:]
            elif line.startswith("\t"):
                lines.append(BlameLine(
                    line_number=current_line_no,
                    revision=current_rev[:8],
                    author=current_author,
                    content=line[1:],
                ))

        return lines

    # --- Mercurial Implementation ---

    def _hg_log(self, path: str | None, limit: int) -> list[CommitInfo]:
        sep = "---HAMMY_SEP---"
        template = f"{{node|short}}{sep}{{author|user}}{sep}{{date|isodatesec}}{sep}{{desc|firstline}}{sep}{{files}}\n"

        cmd = ["hg", "log", f"--template={template}", f"--limit={limit}"]
        if path:
            cmd.append(path)

        output = self._run(cmd)
        if not output.strip():
            return []

        commits: list[CommitInfo] = []
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split(sep)
            if len(parts) < 5:
                continue

            try:
                date = datetime.fromisoformat(parts[2].strip())
            except ValueError:
                date = datetime.now(timezone.utc)

            files = [f.strip() for f in parts[4].split() if f.strip()]

            commits.append(CommitInfo(
                revision=parts[0],
                author=parts[1],
                date=date,
                message=parts[3],
                files_changed=files,
            ))

        return commits

    def _hg_blame(self, path: str) -> list[BlameLine]:
        output = self._run(["hg", "annotate", "-u", "-c", path])
        if not output.strip():
            return []

        lines: list[BlameLine] = []
        for i, line in enumerate(output.split("\n"), 1):
            if not line:
                continue
            # Format: "user rev: content"
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue
            header = parts[0].strip().rsplit(" ", 1)
            if len(header) < 2:
                continue

            lines.append(BlameLine(
                line_number=i,
                revision=header[1],
                author=header[0],
                content=parts[1].lstrip(),
            ))

        return lines

    def _run(self, cmd: list[str]) -> str:
        """Run a VCS command and return its stdout."""
        result = subprocess.run(
            cmd,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"VCS command failed: {' '.join(cmd)}\n"
                f"stderr: {result.stderr}"
            )
        return result.stdout
