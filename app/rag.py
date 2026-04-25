from __future__ import annotations

import logging
import traceback
from typing import Any

from .chunker import TextChunker
from .config import Settings
from .embeddings import EmbeddingProvider
from .models import QueryResponse, RetrievedChunk
from .reranker import DocumentReranker
from .retriever import DocumentRetriever
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, settings: Settings) -> None:
        logger.info("🚀 Initializing RAGPipeline...")
        
        try:
            logger.info("  Creating TextChunker...")
            self.settings = settings
            self.chunker = TextChunker(settings)
            logger.info("  TextChunker created")
            
            logger.info("  Creating EmbeddingProvider...")
            self.embedding_provider = EmbeddingProvider(settings)
            logger.info("  EmbeddingProvider created")
            
            logger.info("  Creating VectorStore...")
            # 🔴 SET BREAKPOINT HERE - This is likely where it crashes
            self.vector_store = VectorStore(settings)
            logger.info("  VectorStore created")
            
            logger.info("  Creating DocumentRetriever...")
            self.retriever = DocumentRetriever(settings, self.embedding_provider, self.vector_store)
            logger.info("  DocumentRetriever created")
            
            logger.info("  Creating DocumentReranker...")
            self.reranker = DocumentReranker(settings, self.embedding_provider)
            logger.info("  DocumentReranker created")
            
            self._llm_provider = None
            logger.info("RAGPipeline initialization complete!")
            
        except Exception as e:
            logger.error(f"RAGPipeline initialization FAILED: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _get_llm_provider(self):
        """Create LLM provider for answer generation (fallback to direct API calls)."""
        if self._llm_provider is None:
            # Since ai-service-kit only provides embedding providers,
            # we'll implement direct LLM API calls
            self._llm_provider = self._create_llm_client()
        return self._llm_provider

    def _create_llm_client(self):
        """Create appropriate LLM client based on provider type."""
        provider_type = self.settings.provider_type.lower()
        
        if provider_type == "openai":
            import openai
            return openai.AsyncOpenAI(api_key=self.settings.openai_api_key)
        elif provider_type == "claude":
            import anthropic
            return anthropic.AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        elif provider_type == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.settings.gemini_api_key)
            return genai  # Return the module itself
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}")

    async def index_documents(
        self,
        documents: list[str],
        metadata: list[dict[str, Any]] | dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Index documents for retrieval.
        
        Args:
            documents: List of text documents to index
            metadata: Either:
                - list of dicts: per-document metadata (must match len(documents))
                - dict: global metadata applied to all documents  
                - None: no additional metadata
        """
        # Step 1: Chunk documents with flexible metadata support
        chunks = self.chunker.chunk_documents(documents, metadata)
        
        # Step 2: Generate embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.embedding_provider.embed_batch(chunk_texts)
        
        # Step 3: Store in vector store
        doc_ids = await self.vector_store.add_documents_with_embeddings(
            chunks=chunks,
            embeddings=embeddings,
            metadata=None  # metadata is already in each chunk
        )
        
        return {
            "indexed_documents": len(documents),
            "created_chunks": len(chunks),
            "document_ids": doc_ids,
            "status": "success",
            "metadata_format": "per_document" if isinstance(metadata, list) else "global" if metadata else "none"
        }

    async def query(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        use_hybrid: bool = True,
        rerank_method: str = "llm_scoring",
        include_sources: bool = True
    ) -> QueryResponse:
        """Full RAG pipeline: retrieve, rerank, and generate answer."""
        k = k or self.settings.default_k
        rerank_k = self.settings.default_rerank_k
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = await self.retriever.retrieve_top_k(
            query=query,
            k=k,
            filters=filters,
            use_hybrid=use_hybrid
        )
        
        if not retrieved_chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[]
            )
        
        # Step 2: Rerank chunks
        reranked_chunks = await self.reranker.rerank_chunks(
            query=query,
            chunks=retrieved_chunks,
            top_k=min(rerank_k, len(retrieved_chunks)),
            method=rerank_method
        )
        
        # Step 3: Assemble context and generate answer
        context = self._assemble_context(reranked_chunks, max_tokens=4000)
        answer = await self._generate_answer(query, context)
        
        # Step 4: Format sources
        sources = [
            RetrievedChunk(
                text=chunk["text"],
                score=chunk["score"],
                metadata=chunk.get("metadata", {})
            )
            for chunk in reranked_chunks
        ] if include_sources else []
        
        return QueryResponse(answer=answer, sources=sources)

    async def retrieve_sources(
        self,
        query: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None,
        use_hybrid: bool = True
    ) -> list[RetrievedChunk]:
        """Retrieve and return source chunks without generating an answer."""
        k = k or self.settings.default_k
        
        retrieved_chunks = await self.retriever.retrieve_top_k(
            query=query,
            k=k,
            filters=filters,
            use_hybrid=use_hybrid
        )
        
        return [
            RetrievedChunk(
                text=chunk["text"],
                score=chunk["score"],
                metadata=chunk.get("metadata", {})
            )
            for chunk in retrieved_chunks
        ]

    def _assemble_context(self, chunks: list[dict[str, Any]], max_tokens: int = 4000) -> str:
        """Assemble context from chunks, respecting token budget."""
        context_parts = []
        estimated_tokens = 0
        
        # Rough estimation: 4 characters per token
        chars_per_token = 4
        
        for chunk in chunks:
            chunk_text = chunk["text"]
            chunk_tokens = len(chunk_text) // chars_per_token
            
            if estimated_tokens + chunk_tokens > max_tokens:
                break
            
            # Add source reference for better provenance
            source_info = chunk.get("metadata", {})
            doc_index = source_info.get("document_index", "unknown")
            chunk_index = source_info.get("chunk_index", "unknown")
            
            formatted_chunk = f"[Source {doc_index}.{chunk_index}] {chunk_text}"
            context_parts.append(formatted_chunk)
            estimated_tokens += chunk_tokens
        
        return "\n\n".join(context_parts)

    async def _generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM with context."""
        provider = await self._get_llm_provider()
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
Use only the information from the context to answer the question. 
If the context doesn't contain enough information to answer the question, say so clearly.
Provide a clear, concise, and accurate answer."""
        
        user_prompt = f"""Context:
{context}

Question: {query}

Answer:"""
        
        try:
            provider_type = self.settings.provider_type.lower()
            
            if provider_type == "openai":
                response = await provider.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
                
            elif provider_type == "claude":
                response = await provider.messages.create(
                    model=self.settings.claude_model,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                return response.content[0].text.strip()
                
            elif provider_type == "gemini":
                import google.generativeai as genai
                model = genai.GenerativeModel('gemini-pro')
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = await model.generate_content_async(
                    combined_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=500,
                        temperature=0.1
                    )
                )
                return response.text.strip()
            
            else:
                return "I encountered an error: Unsupported LLM provider"
            
        except Exception as e:
            return f"I encountered an error while generating the answer: {str(e)}"

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the indexed documents."""
        return await self.vector_store.get_collection_stats()

    async def clear_index(self) -> bool:
        """Clear all indexed documents."""
        return await self.vector_store.delete_collection()

    async def find_similar_documents(
        self,
        document_text: str,
        k: int | None = None,
        filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        """Find documents similar to a given document."""
        k = k or self.settings.default_k
        
        similar_chunks = await self.retriever.get_similar_documents(
            document_text=document_text,
            k=k,
            filters=filters
        )
        
        return [
            RetrievedChunk(
                text=chunk["text"],
                score=chunk["score"],
                metadata=chunk.get("metadata", {})
            )
            for chunk in similar_chunks
        ]

    async def search_by_metadata(
        self,
        filters: dict[str, Any],
        limit: int = 100
    ) -> list[RetrievedChunk]:
        """Search documents by metadata only."""
        chunks = await self.retriever.retrieve_by_metadata(filters, limit)
        
        return [
            RetrievedChunk(
                text=chunk["text"],
                score=chunk["score"],
                metadata=chunk.get("metadata", {})
            )
            for chunk in chunks
        ]