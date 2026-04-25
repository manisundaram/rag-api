import pytest
from app.config import Settings, get_settings


def test_settings_defaults():
    settings = Settings()
    assert settings.app_name == "rag-api"
    assert settings.app_version == "0.1.0"
    assert settings.api_port == 8001
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.default_k == 10
    assert settings.default_rerank_k == 5


def test_provider_config():
    settings = Settings(provider_type="openai", openai_api_key="test-key")
    config = settings.provider_config()
    assert config["api_key"] == "test-key"
    assert config["model"] == settings.openai_model


def test_masked_secrets():
    settings = Settings(
        openai_api_key="sk-1234567890",
        gemini_api_key="AIza1234567890"
    )
    masked = settings.masked_secrets()
    assert masked["openai_api_key"] == "sk-...7890"
    assert "1234567890" not in str(masked)


def test_get_settings_singleton():
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2


def test_cors_origins_parsing():
    settings = Settings(cors_origins="http://localhost:3000,http://localhost:8080")
    assert len(settings.cors_origins) == 2
    assert "http://localhost:3000" in settings.cors_origins
    assert "http://localhost:8080" in settings.cors_origins