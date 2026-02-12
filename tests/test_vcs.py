"""Tests for the VCS wrapper.

Creates temporary Git repositories as test fixtures.
"""

import subprocess
from pathlib import Path

import pytest

from hammy.tools.vcs import VCSType, VCSWrapper


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary Git repo with a few commits."""

    def run(*args: str) -> None:
        subprocess.run(
            args, cwd=tmp_path, capture_output=True, text=True, check=True
        )

    run("git", "init")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test User")

    # Commit 1: initial file
    (tmp_path / "app.php").write_text("<?php echo 'hello';")
    run("git", "add", "app.php")
    run("git", "commit", "-m", "Initial commit")

    # Commit 2: add JS file
    (tmp_path / "app.js").write_text("console.log('hello');")
    run("git", "add", "app.js")
    run("git", "commit", "-m", "Add JavaScript file")

    # Commit 3: modify PHP file
    (tmp_path / "app.php").write_text("<?php echo 'world';")
    run("git", "add", "app.php")
    run("git", "commit", "-m", "Update PHP greeting")

    return tmp_path


class TestVCSDetection:
    def test_detect_git(self, git_repo: Path):
        assert VCSWrapper.detect(git_repo) == VCSType.GIT

    def test_detect_hg(self, tmp_path: Path):
        (tmp_path / ".hg").mkdir()
        assert VCSWrapper.detect(tmp_path) == VCSType.MERCURIAL

    def test_detect_none_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="No VCS detected"):
            VCSWrapper.detect(tmp_path)

    def test_wrapper_auto_detects(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        assert wrapper.vcs_type == VCSType.GIT


class TestGitLog:
    def test_log_returns_commits(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        commits = wrapper.log()
        assert len(commits) == 3

    def test_log_order(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        commits = wrapper.log()
        # Most recent first
        assert commits[0].message == "Update PHP greeting"
        assert commits[1].message == "Add JavaScript file"
        assert commits[2].message == "Initial commit"

    def test_log_has_authors(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        commits = wrapper.log()
        assert all(c.author == "Test User" for c in commits)

    def test_log_has_dates(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        commits = wrapper.log()
        assert all(c.date is not None for c in commits)

    def test_log_has_files_changed(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        commits = wrapper.log()
        assert "app.php" in commits[0].files_changed
        assert "app.js" in commits[1].files_changed

    def test_log_limit(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        commits = wrapper.log(limit=1)
        assert len(commits) == 1

    def test_log_path_filter(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        commits = wrapper.log(path="app.js")
        assert len(commits) == 1
        assert commits[0].message == "Add JavaScript file"


class TestGitBlame:
    def test_blame_returns_lines(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        lines = wrapper.blame("app.php")
        assert len(lines) == 1
        assert lines[0].content == "<?php echo 'world';"
        assert lines[0].author == "Test User"
        assert lines[0].line_number == 1

    def test_blame_multi_line(self, git_repo: Path):
        # Add a multi-line file
        (git_repo / "multi.php").write_text("line1\nline2\nline3\n")
        subprocess.run(
            ["git", "add", "multi.php"],
            cwd=git_repo, capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Add multi-line file"],
            cwd=git_repo, capture_output=True, check=True,
        )

        wrapper = VCSWrapper(git_repo)
        lines = wrapper.blame("multi.php")
        assert len(lines) == 3
        assert lines[0].content == "line1"
        assert lines[2].content == "line3"


class TestChurn:
    def test_churn_counts(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        churn = wrapper.churn(window_days=365)
        # app.php was in 2 commits (initial + update)
        assert churn.get("app.php", 0) == 2
        # app.js was in 1 commit
        assert churn.get("app.js", 0) == 1

    def test_churn_sorted_descending(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        churn = wrapper.churn(window_days=365)
        values = list(churn.values())
        assert values == sorted(values, reverse=True)


class TestDiff:
    def test_diff_between_revisions(self, git_repo: Path):
        wrapper = VCSWrapper(git_repo)
        commits = wrapper.log()
        diff = wrapper.diff(commits[2].revision, commits[0].revision)
        assert "hello" in diff or "world" in diff
