"""Configuration loading for Hammy."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class IgnoreConfig(BaseModel):
    """Settings for the ignore system."""

    use_gitignore: bool = True
    use_hgignore: bool = True
    use_hammyignore: bool = True
    extra_patterns: list[str] = Field(default_factory=list)


class ParsingConfig(BaseModel):
    """Settings for AST parsing."""

    languages: list[str] = Field(
        default_factory=lambda: ["php", "javascript", "python", "typescript", "go"]
    )
    max_file_size_kb: int = 500


class QdrantConfig(BaseModel):
    """Settings for the Qdrant vector database."""

    host: str = "localhost"
    port: int = 6333
    collection_prefix: str = "hammy"
    embedding_model: str = "all-MiniLM-L6-v2"


class VCSConfig(BaseModel):
    """Settings for version control integration."""

    max_commits: int = 5000
    churn_window_days: int = 90


class EnrichmentConfig(BaseModel):
    """Settings for LLM-powered symbol summarization."""

    enabled: bool = False
    provider: str = "anthropic"
    model: str = "claude-haiku-4-5-20251001"
    batch_size: int = 10
    skip_if_summary: bool = True
    max_symbols: int = 0  # 0 = no limit


class ProjectConfig(BaseModel):
    """Top-level project settings."""

    name: str = "my-project"
    root: str = "."


class HammyConfig(BaseSettings):
    """Main Hammy configuration, loaded from YAML + environment variables."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    ignore: IgnoreConfig = Field(default_factory=IgnoreConfig)
    parsing: ParsingConfig = Field(default_factory=ParsingConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    vcs: VCSConfig = Field(default_factory=VCSConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "HammyConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        return cls(**data)

    @classmethod
    def load(cls, project_root: Path | None = None) -> "HammyConfig":
        """Load configuration, searching for config/hammy.yaml relative to project root."""
        if project_root is None:
            project_root = Path.cwd()

        config_path = project_root / "config" / "hammy.yaml"
        if config_path.exists():
            config = cls.from_yaml(config_path)
        else:
            config = cls()

        # Resolve project root to absolute path
        root = Path(config.project.root)
        if not root.is_absolute():
            root = project_root / root
        config.project.root = str(root.resolve())

        return config
