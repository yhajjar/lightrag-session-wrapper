from __future__ import annotations

from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    lightrag_url: str = Field(default="http://lightrag:9621", alias="LIGHTRAG_URL")
    lightrag_api_key: str | None = Field(default=None, alias="LIGHTRAG_API_KEY")
    wrapper_port: int = Field(default=8000, alias="WRAPPER_PORT")
    session_max_age_hours: int = Field(default=24, alias="SESSION_MAX_AGE_HOURS")
    session_cleanup_interval: int = Field(
        default=3600, alias="SESSION_CLEANUP_INTERVAL"
    )
    workers: int = Field(default=2, alias="WORKERS")
    log_level: str = Field(default="info", alias="LOG_LEVEL")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    upload_status_timeout: float = Field(
        default=15.0, alias="UPLOAD_STATUS_TIMEOUT"
    )
    upload_status_poll_interval: float = Field(
        default=2.0, alias="UPLOAD_STATUS_POLL_INTERVAL"
    )
    upload_status_background_timeout: float = Field(
        default=180.0, alias="UPLOAD_STATUS_BACKGROUND_TIMEOUT"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def cors_origin_list(self) -> List[str]:
        """Return CORS origins as a list, handling '*' wildcard."""
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin]


settings = Settings()
