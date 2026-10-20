"""Central config, loaded once from environment/.env. Never hardcode secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv("DB_PATH", "data/f1.db")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    f1_api_key: str = os.getenv("F1_API_KEY", "")
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    fastf1_cache_dir: str = os.getenv("FASTF1_CACHE_DIR", "data/raw/fastf1_cache")


settings = Settings()
