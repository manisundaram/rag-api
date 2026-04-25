from __future__ import annotations

import asyncio
from typing import Any

from .config import Settings


class EmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    def _get_client(self):
        """Get the appropriate embedding client based on provider type."""
        if self._client is None:
            provider_type = self.settings.provider_type.lower()
            
            if provider_type == "openai":
                import openai
                self._client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)
            elif provider_type == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.settings.gemini_api_key)
                self._client = genai
            else:
                raise ValueError(f"Unsupported embedding provider: {provider_type}")
        
        return self._client

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        provider_type = self.settings.provider_type.lower()
        client = self._get_client()
        
        if provider_type == "openai":
            response = await client.embeddings.create(
                model=self.settings.openai_model,
                input=text
            )
            return response.data[0].embedding
        elif provider_type == "gemini":
            response = client.embed_content(
                model=self.settings.gemini_model,
                content=text,
                task_type="retrieval_document"
            )
            return response['embedding']
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")

    async def embed_batch(self, texts: list[str], batch_size: int = 10) -> list[list[float]]:
        """Generate embeddings for multiple texts in batches."""
        provider_type = self.settings.provider_type.lower()
        client = self._get_client()
        
        all_embeddings = []
        
        if provider_type == "openai":
            # Process in batches to avoid overwhelming the API
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = await client.embeddings.create(
                    model=self.settings.openai_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
        elif provider_type == "gemini":
            # Process individually for Gemini
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = []
                for text in batch:
                    response = client.embed_content(
                        model=self.settings.gemini_model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    batch_embeddings.append(response['embedding'])
                all_embeddings.extend(batch_embeddings)
        else:
            raise ValueError(f"Unsupported provider: {provider_type}")
        
        return all_embeddings

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current model."""
        provider_type = self.settings.provider_type.lower()
        model = self.settings.selected_provider_model()
        
        # Common embedding dimensions
        dimension_map = {
            "openai": {
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072,
                "text-embedding-ada-002": 1536,
            },
            "gemini": {
                "models/embedding-001": 768,
                "models/gemini-embedding-001": 768,
            }
        }
        
        if provider_type in dimension_map and model in dimension_map[provider_type]:
            return dimension_map[provider_type][model]
        
        # Default dimensions
        if provider_type == "openai":
            return 1536
        elif provider_type == "gemini":
            return 768
        else:
            return 1536  # Default fallback