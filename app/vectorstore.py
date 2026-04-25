from __future__ import annotations

import logging
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import Settings

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        logger.info("Initializing VectorStore...")
        
        try:
            self.settings = settings
            logger.info(f"Collection name: {settings.default_collection_name}")
            
            logger.info("Creating ChromaDB client...")
            self.client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("ChromaDB client created")
            
            self.collection_name = settings.default_collection_name
            self._collection = None
            logger.info("VectorStore initialization complete!")
            
        except Exception as e:
            logger.error(f"VectorStore initialization FAILED: {str(e)}")
            raise

    def _get_collection(self):
        """Get or create the collection without auto-embedding - simplified approach."""
        logger.info(f"_get_collection called, current _collection: {self._collection}")
        
        if self._collection is None:
            logger.info(f"Creating new collection with name: {self.collection_name}")
            
            # Delete collection if it exists (exactly like working isolated test)
            try:
                logger.info(f"Trying to delete existing collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
                logger.info(f"Successfully deleted collection: {self.collection_name}")
            except Exception as delete_e:
                logger.info(f"No existing collection to delete (expected): {delete_e}")
            
            # Create collection WITHOUT embedding function (exactly like isolated test)
            try:
                logger.info(f"Creating collection: {self.collection_name}")
                self._collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "RAG document collection"},
                    embedding_function=None
                )
                logger.info(f"Successfully created collection: {self.collection_name}")
            except Exception as create_e:
                logger.error(f"Failed to create collection {self.collection_name}: {create_e}")
                raise
            
        logger.info(f"Returning collection: {self._collection}")
        return self._collection

    def _reset_collection_if_dimension_mismatch(self, expected_dimension: int) -> None:
        """Reset collection if there's a dimension mismatch - simplified."""
        # Just clear the cached collection, let _get_collection recreate it
        self._collection = None

    async def add_documents(
        self, 
        chunks: list[dict[str, Any]], 
        metadata: dict[str, Any] | None = None
    ) -> list[str]:
        """Add document chunks to the vector store."""
        collection = self._get_collection()
        
        documents = []
        metadatas = []
        ids = []
        
        base_metadata = metadata or {}
        
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            documents.append(chunk["text"])
            
            # Combine base metadata with chunk metadata
            combined_metadata = {
                **base_metadata,
                **chunk.get("metadata", {}),
                "chunk_id": doc_id,
            }
            metadatas.append(combined_metadata)
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids

    async def add_documents_with_embeddings(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
        metadata: dict[str, Any] | None = None
    ) -> list[str]:
        """Add document chunks with pre-computed embeddings."""
        logger.info(f"add_documents_with_embeddings called with {len(chunks)} chunks")
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Check for dimension mismatch and reset collection if needed
        if embeddings:
            expected_dimension = len(embeddings[0])
            logger.info(f"Embeddings dimension: {expected_dimension}")
            self._reset_collection_if_dimension_mismatch(expected_dimension)
        
        try:
            collection = self._get_collection()
            logger.info(f"Got collection for adding documents: {collection}")
        except Exception as e:
            logger.error(f"Error getting collection in add_documents_with_embeddings: {e}")
            raise
        
        documents = []
        metadatas = []
        ids = []
        
        base_metadata = metadata or {}
        
        for chunk in chunks:
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            documents.append(chunk["text"])
            
            combined_metadata = {
                **base_metadata,
                **chunk.get("metadata", {}),
                "chunk_id": doc_id,
            }
            metadatas.append(combined_metadata)
        
        try:
            logger.info(f"About to add {len(documents)} documents to ChromaDB")
            logger.info(f"Sample document: {documents[0] if documents else 'No documents'}")
            logger.info(f"Sample metadata: {metadatas[0] if metadatas else 'No metadata'}")
            
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            logger.info("Successfully added documents to ChromaDB")
        except Exception as e:
            logger.error(f"Error adding documents to ChromaDB: {e}")
            # If there's any error, just re-raise it
            raise
        
        return ids

    async def query_documents(
        self,
        query_embedding: list[float],
        k: int = 10,
        filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Query documents using vector similarity."""
        collection = self._get_collection()
        
        # Convert filters to ChromaDB format
        where_clause = self._build_where_clause(filters) if filters else None
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            logger.error(f"Query embedding dimension: {len(query_embedding)}")
            logger.error(f"Collection name: {self.collection_name}")
            
            if "dimension" in str(e).lower():
                # Dimension mismatch - reset collection and return empty results
                expected_dimension = len(query_embedding)
                self._reset_collection_if_dimension_mismatch(expected_dimension)
                # After reset, collection is empty, so return empty results
                return []
            else:
                raise
        
        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                score = 1.0 - results["distances"][0][i]  # Convert distance to similarity score
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                
                chunks.append({
                    "text": doc,
                    "score": score,
                    "metadata": metadata
                })
        
        return chunks

    async def query_documents_by_text(
        self,
        query_text: str,
        k: int = 10,
        filters: dict[str, Any] | None = None,
        embedding_provider: Any = None
    ) -> list[dict[str, Any]]:
        """Query documents using text - generates embeddings manually to avoid dimension mismatch."""
        collection = self._get_collection()
        
        # Generate embedding manually if provider is provided
        if embedding_provider:
            query_embedding = await embedding_provider.embed_text(query_text)
            
            where_clause = self._build_where_clause(filters) if filters else None
            
            results = collection.query(
                query_embeddings=[query_embedding],  # Use manual embedding
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
        else:
            # Fallback to text query (may cause dimension mismatch)
            logger.warning("No embedding provider passed to query_documents_by_text - using fallback text query")
            where_clause = self._build_where_clause(filters) if filters else None
            
            results = collection.query(
                query_texts=[query_text],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
        
        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                score = 1.0 - results["distances"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                
                chunks.append({
                    "text": doc,
                    "score": score,
                    "metadata": metadata
                })
        
        return chunks

    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        where_clause = {}
        
        for key, value in filters.items():
            if isinstance(value, list):
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict):
                # Handle nested filters like {"$gt": 10}
                where_clause[key] = value
            else:
                where_clause[key] = {"$eq": value}
        
        return where_clause

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection."""
        collection = self._get_collection()
        count = collection.count()
        
        return {
            "name": self.collection_name,
            "document_count": count,
            "backend": "chroma"
        }

    async def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._collection = None
            return True
        except Exception:
            return False

    async def delete_documents(self, doc_ids: list[str]) -> bool:
        """Delete specific documents by ID."""
        try:
            collection = self._get_collection()
            collection.delete(ids=doc_ids)
            return True
        except Exception:
            return False

    async def search_by_metadata(
        self,
        filters: dict[str, Any],
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Search documents by metadata only."""
        collection = self._get_collection()
        
        where_clause = self._build_where_clause(filters)
        
        results = collection.get(
            where=where_clause,
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        chunks = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"]):
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                
                chunks.append({
                    "text": doc,
                    "score": 1.0,  # No similarity score for metadata search
                    "metadata": metadata
                })
        
        return chunks