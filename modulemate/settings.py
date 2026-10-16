"""Operator configuration. No provider key is stored in the application database."""

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    ai_base_url: str = "https://api.openai.com/v1"
    ai_model: str = ""
    ai_api_key: str = ""

    @property
    def ai_ready(self) -> bool:
        return bool(self.ai_model and (self.ai_api_key or self.provider_is_local))

    @property
    def provider_is_local(self) -> bool:
        return urlparse(self.ai_base_url).hostname in {"localhost", "127.0.0.1", "::1"}

    @property
    def provider_host(self) -> str:
        return urlparse(self.ai_base_url).netloc


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    url = os.getenv("MODULEMATE_AI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    parsed = urlparse(url)
    if (
        parsed.scheme not in ("https", "http")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("Configure a valid AI base URL without embedded credentials.")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Remote AI providers must use HTTPS.")
    return Settings(
        data_dir=Path(os.getenv("MODULEMATE_DATA_DIR", str(ROOT / "data"))).expanduser().resolve(),
        ai_base_url=url,
        ai_model=os.getenv("MODULEMATE_AI_MODEL", "").strip(),
        ai_api_key=os.getenv("MODULEMATE_AI_API_KEY", "").strip(),
    )
