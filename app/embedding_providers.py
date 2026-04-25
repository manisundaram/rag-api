"""Concrete embedding provider implementations for ai-service-kit."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping, Sequence

from ai_service_kit.providers import (
    BaseEmbeddingProvider, 
    EmbeddingResult, 
    EmbeddingUsage,
    EmbeddingConfigError,
    EmbeddingAPIError
)


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider implementation."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__(config)
        self._client = None

    def validate_config(self) -> bool:
        """Validate provider configuration and raise on invalid state."""
        api_key = self.config.get("api_key")
        if not api_key:
            raise EmbeddingConfigError(
                "OpenAI API key is required",
                provider=self.get_provider_name()
            )
        return True

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.AsyncOpenAI(api_key=self.config["api_key"])
            except ImportError as e:
                raise EmbeddingConfigError(
                    "OpenAI package not installed. Run: pip install openai",
                    provider=self.get_provider_name()
                ) from e
        return self._client

    async def embed(
        self,
        texts: Sequence[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Generate embeddings for the provided texts."""
        client = self._get_client()
        model = model or self.config.get("model", "text-embedding-3-small")
        
        try:
            response = await client.embeddings.create(
                model=model,
                input=list(texts)
            )
            
            embeddings = [item.embedding for item in response.data]
            usage = EmbeddingUsage(
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens
            )
            
            return EmbeddingResult.from_vectors(
                embeddings=embeddings,
                model=model,
                usage=usage,
                provider=self.get_provider_name(),
                dimension=len(embeddings[0]) if embeddings else self.get_embedding_dimension(model)
            )
            
        except Exception as e:
            raise EmbeddingAPIError(
                f"OpenAI API request failed: {str(e)}",
                provider=self.get_provider_name()
            ) from e

    def get_available_models(self) -> list[str]:
        """Return the available embedding model identifiers."""
        return [
            "text-embedding-3-small",
            "text-embedding-3-large", 
            "text-embedding-ada-002"
        ]

    def get_embedding_dimension(self, model: str | None = None) -> int:
        """Return the embedding vector dimension for the given model."""
        model = model or "text-embedding-3-small"
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dimensions.get(model, 1536)

    def get_max_input_tokens(self) -> int:
        """Return the maximum supported input token count."""
        return 8192


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """Google Gemini embedding provider implementation."""

    def __init__(self, config: Mapping[str, Any] | None = None):
        super().__init__(config)
        self._configured = False

    def validate_config(self) -> bool:
        """Validate provider configuration and raise on invalid state."""
        api_key = self.config.get("api_key")
        if not api_key:
            raise EmbeddingConfigError(
                "Gemini API key is required",
                provider=self.get_provider_name()
            )
        return True

    def _ensure_configured(self):
        """Ensure Gemini is configured."""
        if not self._configured:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config["api_key"])
                self._configured = True
            except ImportError as e:
                raise EmbeddingConfigError(
                    "Google Generative AI package not installed. Run: pip install google-generativeai",
                    provider=self.get_provider_name()
                ) from e

    async def embed(
        self,
        texts: Sequence[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> EmbeddingResult:
        """Generate embeddings for the provided texts."""
        self._ensure_configured()
        
        try:
            import google.generativeai as genai
            
            model = model or self.config.get("model", "models/embedding-001")
            
            embeddings = []
            total_tokens = 0
            
            for text in texts:
                response = genai.embed_content(
                    model=model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(response['embedding'])
                # Rough token estimation for Gemini
                total_tokens += len(text.split())
            
            usage = EmbeddingUsage(
                prompt_tokens=total_tokens,
                total_tokens=total_tokens
            )
            
            return EmbeddingResult.from_vectors(
                embeddings=embeddings,
                model=model,
                usage=usage,
                provider=self.get_provider_name(),
                dimension=self.get_embedding_dimension(model)
            )
            
        except Exception as e:
            raise EmbeddingAPIError(
                f"Gemini API request failed: {str(e)}",
                provider=self.get_provider_name()
            ) from e

    def get_available_models(self) -> list[str]:
        """Return the available embedding model identifiers."""
        return [
            "models/embedding-001",
            "models/text-embedding-004"
        ]

    def get_embedding_dimension(self, model: str | None = None) -> int:
        """Return the embedding vector dimension for the given model."""
        return 768  # Gemini embeddings are 768-dimensional

    def get_max_input_tokens(self) -> int:
        """Return the maximum supported input token count."""
        return 2048  # Conservative estimate for Gemini