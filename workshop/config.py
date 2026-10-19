"""Centralized configuration and environment settings for Panorama Video Maker."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is the root directory containing pyproject.toml
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            str(ROOT_DIR / ".env"),
            str(Path.home() / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base paths
    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    inputs_dir: Path = ROOT_DIR / "inputs"
    outputs_dir: Path = ROOT_DIR / "outputs"
    web_dir: Path = ROOT_DIR / "web"
    bridges_dir: Path = ROOT_DIR / "inputs" / "bridges"

    # Database
    database_url: str | None = None

    # AI API Keys
    openai_api_key: str | None = None
    google_generative_ai_api_key: str | None = None

    # Processing Defaults
    default_imagen_model: str = "gpt-image-2.5-sunburst-2026-09-08"
    default_gif_encoder: str = "auto"

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        self.data_dir.mkdir(parents=True, exist_ok=True)
        db_path = self.data_dir / "studio.db"
        return f"sqlite:///{db_path}"

    def ensure_directories(self) -> None:
        """Create standard application directories if they do not exist."""
        for directory in [self.data_dir, self.inputs_dir, self.outputs_dir, self.bridges_dir]:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
