from __future__ import annotations

import asyncio
from typing import Any

from ai_service_kit.providers import ProviderFactory, BaseEmbeddingProvider

from .config import Settings
from .embedding_providers import OpenAIEmbeddingProvider, GeminiEmbeddingProvider


class EmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._provider: BaseEmbeddingProvider | None = None
        self._setup_providers()

    def _setup_providers(self) -> None:
        """Register our custom providers with ai-service-kit."""
        factory = ProviderFactory()
        
        # Register our providers with the factory's registry
        factory.register_provider("openai", OpenAIEmbeddingProvider)
        factory.register_provider("gemini", GeminiEmbeddingProvider)
        
        self._factory = factory

    def _get_provider(self) -> BaseEmbeddingProvider:
        """Lazy initialization of provider."""
        if self._provider is None:
            self._provider = self._factory.create_provider(
                provider_name=self.settings.provider_type,
                config=self.settings.provider_config()
            )
        return self._provider

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        provider = self._get_provider()
        result = await provider.embed([text], model=self.settings.selected_provider_model())
        return list(result.embeddings[0])

    async def embed_batch(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Generate embeddings for multiple texts in batches."""
        provider = self._get_provider()
        
        # Process in batches to avoid overwhelming the API
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = await provider.embed(batch, model=self.settings.selected_provider_model())
            all_embeddings.extend([list(emb) for emb in result.embeddings])
        
        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model."""
        provider = self._get_provider()
        return provider.get_embedding_dimension(self.settings.selected_provider_model())