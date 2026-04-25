from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from ai_service_kit.utils import mask_secret
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="rag-api", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    mock_mode: bool = Field(default=False, alias="MOCK_MODE")
    provider_type: str = Field(default="openai", alias="PROVIDER_TYPE")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="text-embedding-3-small", alias="OPENAI_MODEL")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="models/embedding-001", alias="GEMINI_MODEL")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    claude_model: str = Field(default="claude-3-5-haiku-latest", alias="CLAUDE_MODEL")
    vectorstore_backend: str = Field(default="chroma", alias="VECTORSTORE_BACKEND")
    default_collection_name: str = Field(default="documents_v2", alias="DEFAULT_COLLECTION_NAME")
    chunk_size: int = Field(default=500, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    default_k: int = Field(default=10, alias="DEFAULT_K")
    default_rerank_k: int = Field(default=5, alias="DEFAULT_RERANK_K")
    enable_cors: bool = Field(default=True, alias="ENABLE_CORS")
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"],
        alias="CORS_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        populate_by_name=True,
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [item.strip() for item in value if item.strip()]
        return [item.strip() for item in value.split(",") if item.strip()]

    def provider_config(self) -> dict[str, Any]:
        provider_name = self.provider_type.strip().lower()
        if provider_name == "openai":
            return {
                "api_key": self.openai_api_key,
                "model": self.openai_model,
            }
        if provider_name == "gemini":
            return {
                "api_key": self.gemini_api_key,
                "model": self.gemini_model,
            }
        if provider_name == "claude":
            return {
                "api_key": self.anthropic_api_key,
                "model": self.claude_model,
            }
        return {}

    def selected_provider_model(self) -> str | None:
        return self.provider_config().get("model")

    def selected_provider_api_key(self) -> str | None:
        return self.provider_config().get("api_key")

    def operational_settings(self) -> dict[str, Any]:
        return {
            "app_env": self.app_env,
            "app_debug": self.app_debug,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "log_level": self.log_level,
            "mock_mode": self.mock_mode,
            "provider_type": self.provider_type,
            "provider_model": self.selected_provider_model(),
            "vectorstore_backend": self.vectorstore_backend,
            "default_collection_name": self.default_collection_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "default_k": self.default_k,
            "default_rerank_k": self.default_rerank_k,
            "enable_cors": self.enable_cors,
            "cors_origins": list(self.cors_origins),
        }

    def masked_secrets(self) -> dict[str, str | None]:
        return {
            "openai_api_key": mask_secret(self.openai_api_key),
            "gemini_api_key": mask_secret(self.gemini_api_key),
            "anthropic_api_key": mask_secret(self.anthropic_api_key),
        }

    def masked_debug_config(self) -> Mapping[str, Any]:
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "app_env": self.app_env,
            "app_debug": self.app_debug,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "log_level": self.log_level,
            "mock_mode": self.mock_mode,
            "provider_type": self.provider_type,
            "openai_api_key": self.masked_secrets()["openai_api_key"],
            "openai_model": self.openai_model,
            "gemini_api_key": self.masked_secrets()["gemini_api_key"],
            "gemini_model": self.gemini_model,
            "anthropic_api_key": self.masked_secrets()["anthropic_api_key"],
            "claude_model": self.claude_model,
            "vectorstore_backend": self.vectorstore_backend,
            "default_collection_name": self.default_collection_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "default_k": self.default_k,
            "default_rerank_k": self.default_rerank_k,
            "enable_cors": self.enable_cors,
            "cors_origins": self.cors_origins,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()