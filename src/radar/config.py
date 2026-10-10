"""Runtime settings (from environment / `.env`) and YAML config loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class Settings(BaseSettings):
    """All runtime settings."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RADAR_", extra="ignore")

    edinet_api_key: SecretStr | None = None
    jquants_api_key: SecretStr | None = None

    data_dir: Path = Path("data")
    edinet_base_url: str = "https://api.edinet-fsa.go.jp/api/v2"
    # J-Quants API V2 uses an `x-api-key` header. Confirm the base URL in the official V2 docs.
    jquants_base_url: str = "https://api.jquants.com/v2"

    requests_per_second: float = 0.5
    http_timeout_s: float = 60.0

    @property
    def raw_dir(self) -> Path:
        """Immutable copies of everything downloaded (bronze layer)."""
        return self.data_dir / "raw"

    @property
    def staging_dir(self) -> Path:
        """Parsed, normalised tables as Parquet (silver layer)."""
        return self.data_dir / "staging"

    @property
    def marts_dir(self) -> Path:
        """Features, labels, scores and backtest outputs (gold layer)."""
        return self.data_dir / "marts"

    @property
    def manual_dir(self) -> Path:
        """Files downloaded by hand (e.g. JPX Excel lists) before they are automated."""
        return self.data_dir / "manual"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "radar.duckdb"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_yaml(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load `config/<name>.yaml` as a dict."""
    path = config_dir / f"{name}.yaml"
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data
