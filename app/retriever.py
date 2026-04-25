from __future__ import annotations

import re
from typing import Any

from .config import Settings
from .embeddings import EmbeddingProvider
from .vectorstore import VectorStore


class DocumentRetriever:
    def __init__(
        self,
        settings: Settings,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore
    ) -> None:
        self.settings = settings
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    async def retrieve_top_k(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        use_hybrid: bool = True
    ) -> list[dict[str, Any]]:
        """Retrieve top-k documents using vector similarity and optional hybrid search."""
        k = k or self.settings.default_k
        
        if use_hybrid:
            return await self._hybrid_search(query, k, filters)
        else:
            return await self._vector_search(query, k, filters)

    async def _vector_search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Pure vector similarity search."""
        query_embedding = await self.embedding_provider.embed_text(query)
        
        # Debug: Log actual embedding dimensions
        print(f"DEBUG: Generated query embedding dimensions: {len(query_embedding)}")
        print(f"DEBUG: First few embedding values: {query_embedding[:5]}")
        
        return await self.vector_store.query_documents(
            query_embedding=query_embedding,
            k=k,
            filters=filters
        )

    async def _hybrid_search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> list[dict[str, Any]]:
        """Hybrid search combining vector similarity and keyword search."""
        # Get more results from each method to have better fusion
        expanded_k = min(k * 2, 50)
        
        # Vector search
        vector_results = await self._vector_search(query, expanded_k, filters)
        
        # Keyword search (BM25-style)
        keyword_results = await self._keyword_search(query, expanded_k, filters)
        
        # Merge and rerank results
        merged_results = self._merge_search_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            vector_weight=vector_weight,
            keyword_weight=keyword_weight,
            top_k=k
        )
        
        return merged_results

    async def _keyword_search(
        self,
        query: str,
        k: int,
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Keyword-based search (simplified BM25 implementation)."""
        # For now, use ChromaDB's text search which has some keyword capabilities
        # In a full implementation, you would integrate with a proper BM25 index
        return await self.vector_store.query_documents_by_text(
            query_text=query,
            k=k,
            filters=filters,
            embedding_provider=self.embedding_provider  # Pass embedding provider
        )

    def _merge_search_results(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        vector_weight: float,
        keyword_weight: float,
        top_k: int
    ) -> list[dict[str, Any]]:
        """Merge and rerank results from vector and keyword search."""
        # Create a map of document text to results for deduplication
        result_map: dict[str, dict[str, Any]] = {}
        
        # Add vector results
        for i, result in enumerate(vector_results):
            text = result["text"]
            vector_score = result["score"]
            vector_rank = i + 1
            
            result_map[text] = {
                "text": text,
                "metadata": result["metadata"],
                "vector_score": vector_score,
                "vector_rank": vector_rank,
                "keyword_score": 0.0,
                "keyword_rank": len(keyword_results) + 1,  # Worst rank for items not in keyword results
            }
        
        # Add keyword results and update existing ones
        for i, result in enumerate(keyword_results):
            text = result["text"]
            keyword_score = result["score"]
            keyword_rank = i + 1
            
            if text in result_map:
                result_map[text]["keyword_score"] = keyword_score
                result_map[text]["keyword_rank"] = keyword_rank
            else:
                result_map[text] = {
                    "text": text,
                    "metadata": result["metadata"],
                    "vector_score": 0.0,
                    "vector_rank": len(vector_results) + 1,
                    "keyword_score": keyword_score,
                    "keyword_rank": keyword_rank,
                }
        
        # Calculate combined scores using RRF (Reciprocal Rank Fusion)
        combined_results = []
        for result in result_map.values():
            # RRF score: 1/(rank + k) where k=60 is common
            rrf_vector = 1.0 / (result["vector_rank"] + 60)
            rrf_keyword = 1.0 / (result["keyword_rank"] + 60)
            
            combined_score = (
                vector_weight * rrf_vector +
                keyword_weight * rrf_keyword
            )
            
            combined_results.append({
                "text": result["text"],
                "score": combined_score,
                "metadata": {
                    **result["metadata"],
                    "vector_score": result["vector_score"],
                    "keyword_score": result["keyword_score"],
                    "vector_rank": result["vector_rank"],
                    "keyword_rank": result["keyword_rank"],
                }
            })
        
        # Sort by combined score and return top-k
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top_k]

    def _calculate_bm25_score(self, query: str, document: str) -> float:
        """Simple BM25 score calculation (stub implementation)."""
        # This is a simplified implementation
        # A full implementation would require document frequency statistics
        query_terms = self._tokenize(query.lower())
        doc_terms = self._tokenize(document.lower())
        
        doc_length = len(doc_terms)
        avg_doc_length = 100  # Placeholder average document length
        
        k1 = 1.2
        b = 0.75
        
        score = 0.0
        for term in query_terms:
            term_freq = doc_terms.count(term)
            if term_freq > 0:
                # Simplified BM25 formula without IDF calculation
                numerator = term_freq * (k1 + 1)
                denominator = term_freq + k1 * (1 - b + b * (doc_length / avg_doc_length))
                score += numerator / denominator
        
        return score

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    async def retrieve_by_metadata(
        self,
        filters: dict[str, Any],
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Retrieve documents by metadata filters only."""
        return await self.vector_store.search_by_metadata(filters, limit)

    async def get_similar_documents(
        self,
        document_text: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        exclude_self: bool = True
    ) -> list[dict[str, Any]]:
        """Find documents similar to a given document."""
        k = k or self.settings.default_k
        
        results = await self._vector_search(document_text, k + 1, filters)
        
        if exclude_self:
            # Remove results that are too similar (might be the same document)
            filtered_results = []
            for result in results:
                if result["score"] < 0.99:  # Threshold to exclude near-duplicates
                    filtered_results.append(result)
                if len(filtered_results) >= k:
                    break
            return filtered_results[:k]
        
        return results[:k]