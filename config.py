"""Configuration loader (S4 / NFR5).

Every tunable parameter -- k, the similarity threshold, and all model names --
is read from config.yaml through this module. Nothing else in the codebase
hardcodes those three values.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"

load_dotenv(ROOT / ".env")


class Config:
    """Dotted-path access over the parsed YAML, with path resolution."""

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        """Resolve a config value as a path relative to the project root."""
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"config key not found: {dotted}")
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    def require(self, dotted: str) -> Any:
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"required config key missing: {dotted}")
        return value

    @property
    def api_key(self) -> str:
        """Read the generator API key from the env var named in config."""
        var = self.require("generation.api_key_env")
        key = os.environ.get(var, "").strip()
        if not key:
            raise RuntimeError(
                f"Generation backend API key not found. Set {var} in the "
                f"environment or in the project .env file (see .env.example)."
            )
        return key

    def as_dict(self) -> dict[str, Any]:
        return self._data


_cached: Config | None = None


def load(reload: bool = False) -> Config:
    global _cached
    if _cached is None or reload:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            _cached = Config(yaml.safe_load(fh), ROOT)
    return _cached
