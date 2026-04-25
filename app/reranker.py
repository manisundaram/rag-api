from __future__ import annotations

import re
from typing import Any

from .config import Settings
from .embeddings import EmbeddingProvider


class DocumentReranker:
    def __init__(
        self,
        settings: Settings,
        embedding_provider: EmbeddingProvider | None = None
    ) -> None:
        self.settings = settings
        self.embedding_provider = embedding_provider
        self._provider = None

    async def _get_provider(self):
        """Create LLM provider for LLM-based scoring."""
        if self._provider is None:
            provider_type = self.settings.provider_type.lower()
            
            if provider_type == "openai":
                import openai
                self._provider = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)
            elif provider_type == "claude":
                import anthropic
                self._provider = anthropic.AsyncAnthropic(api_key=self.settings.anthropic_api_key)
            elif provider_type == "gemini":
                import google.generativeai as genai
                genai.configure(api_key=self.settings.gemini_api_key)
                self._provider = genai
            else:
                raise ValueError(f"Unsupported LLM provider: {provider_type}")
        
        return self._provider

    async def rerank_chunks(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int | None = None,
        method: str = "llm_scoring"
    ) -> list[dict[str, Any]]:
        """Rerank retrieved chunks using the specified method."""
        if not chunks:
            return chunks
        
        top_k = top_k or self.settings.default_rerank_k
        top_k = min(top_k, len(chunks))
        
        if method == "llm_scoring":
            return await self._llm_rerank(query, chunks, top_k)
        elif method == "embedding_similarity":
            return await self._embedding_rerank(query, chunks, top_k)
        elif method == "keyword_overlap":
            return await self._keyword_rerank(query, chunks, top_k)
        else:
            raise ValueError(f"Unknown reranking method: {method}")

    async def _llm_rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int
    ) -> list[dict[str, Any]]:
        """Rerank using LLM-based relevance scoring."""
        provider = await self._get_provider()
        
        scoring_prompt = self._build_scoring_prompt(query, chunks)
        
        try:
            provider_type = self.settings.provider_type.lower()
            
            if provider_type == "openai":
                response = await provider.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": scoring_prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                response_text = response.choices[0].message.content
            elif provider_type == "claude":
                response = await provider.messages.create(
                    model=self.settings.claude_model,
                    messages=[{"role": "user", "content": scoring_prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                response_text = response.content[0].text
            elif provider_type == "gemini":
                import google.generativeai as genai
                model = genai.GenerativeModel('gemini-pro')
                response = await model.generate_content_async(
                    scoring_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=500,
                        temperature=0.1
                    )
                )
                response_text = response.text
            else:
                raise ValueError(f"Unsupported provider: {provider_type}")
                
            scores = self._parse_llm_scores(response_text, len(chunks))
        except Exception:
            # Fallback to original scores if LLM scoring fails
            scores = [chunk.get("score", 0.0) for chunk in chunks]
        
        # Apply new scores and sort
        reranked_chunks = []
        for i, chunk in enumerate(chunks):
            new_score = scores[i] if i < len(scores) else chunk.get("score", 0.0)
            reranked_chunk = {
                **chunk,
                "score": new_score,
                "metadata": {
                    **chunk.get("metadata", {}),
                    "original_score": chunk.get("score", 0.0),
                    "rerank_method": "llm_scoring",
                }
            }
            reranked_chunks.append(reranked_chunk)
        
        reranked_chunks.sort(key=lambda x: x["score"], reverse=True)
        return reranked_chunks[:top_k]

    def _build_scoring_prompt(self, query: str, chunks: list[dict[str, Any]]) -> str:
        """Build prompt for LLM-based relevance scoring."""
        prompt = f"""You are a relevance scoring assistant. Given a query and a list of text chunks, rate the relevance of each chunk to the query on a scale of 0.0 to 1.0, where 1.0 is most relevant.

Query: {query}

Text chunks to score:
"""
        
        for i, chunk in enumerate(chunks):
            prompt += f"\n{i+1}. {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}\n"
        
        prompt += """\nProvide scores in this exact format:
1: 0.85
2: 0.72
3: 0.91
...

Only provide the numbered scores, no explanation."""
        
        return prompt

    def _parse_llm_scores(self, response: str, expected_count: int) -> list[float]:
        """Parse LLM response to extract relevance scores."""
        scores = []
        
        # Look for patterns like "1: 0.85" or "1. 0.85"
        pattern = r'(\d+)[:.]?\s*([0-9.]+)'
        matches = re.findall(pattern, response)
        
        # Create a map of index to score
        score_map = {}
        for match in matches:
            try:
                index = int(match[0]) - 1  # Convert to 0-based index
                score = float(match[1])
                if 0 <= score <= 1.0:
                    score_map[index] = score
            except (ValueError, IndexError):
                continue
        
        # Build final scores list with fallbacks
        for i in range(expected_count):
            if i in score_map:
                scores.append(score_map[i])
            else:
                scores.append(0.5)  # Default medium score
        
        return scores

    async def _embedding_rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int
    ) -> list[dict[str, Any]]:
        """Rerank using embedding similarity refinement."""
        if not self.embedding_provider:
            # Fallback to original scores
            return sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)[:top_k]
        
        query_embedding = await self.embedding_provider.embed_text(query)
        chunk_texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = await self.embedding_provider.embed_batch(chunk_texts)
        
        reranked_chunks = []
        for i, chunk in enumerate(chunks):
            similarity = self._cosine_similarity(query_embedding, chunk_embeddings[i])
            
            # Combine original score with new similarity
            original_score = chunk.get("score", 0.0)
            combined_score = 0.6 * similarity + 0.4 * original_score
            
            reranked_chunk = {
                **chunk,
                "score": combined_score,
                "metadata": {
                    **chunk.get("metadata", {}),
                    "original_score": original_score,
                    "embedding_similarity": similarity,
                    "rerank_method": "embedding_similarity",
                }
            }
            reranked_chunks.append(reranked_chunk)
        
        reranked_chunks.sort(key=lambda x: x["score"], reverse=True)
        return reranked_chunks[:top_k]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)

    async def _keyword_rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int
    ) -> list[dict[str, Any]]:
        """Rerank using keyword overlap and term frequency."""
        query_terms = set(self._tokenize(query.lower()))
        
        reranked_chunks = []
        for chunk in chunks:
            chunk_terms = set(self._tokenize(chunk["text"].lower()))
            
            # Calculate keyword overlap
            overlap = len(query_terms & chunk_terms)
            overlap_ratio = overlap / len(query_terms) if query_terms else 0.0
            
            # Calculate term frequency score
            tf_score = self._calculate_tf_score(query_terms, chunk["text"].lower())
            
            # Combine with original score
            original_score = chunk.get("score", 0.0)
            keyword_score = 0.4 * overlap_ratio + 0.3 * tf_score + 0.3 * original_score
            
            reranked_chunk = {
                **chunk,
                "score": keyword_score,
                "metadata": {
                    **chunk.get("metadata", {}),
                    "original_score": original_score,
                    "keyword_overlap": overlap_ratio,
                    "tf_score": tf_score,
                    "rerank_method": "keyword_overlap",
                }
            }
            reranked_chunks.append(reranked_chunk)
        
        reranked_chunks.sort(key=lambda x: x["score"], reverse=True)
        return reranked_chunks[:top_k]

    def _calculate_tf_score(self, query_terms: set[str], text: str) -> float:
        """Calculate term frequency score."""
        text_terms = self._tokenize(text)
        total_terms = len(text_terms)
        
        if total_terms == 0:
            return 0.0
        
        tf_score = 0.0
        for term in query_terms:
            term_count = text_terms.count(term)
            tf_score += term_count / total_terms
        
        return tf_score

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()