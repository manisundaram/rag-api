from __future__ import annotations

import re
from typing import Any

from .config import Settings


class TextChunker:
    def __init__(self, settings: Settings) -> None:
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

    def chunk_text(
        self, 
        text: str, 
        chunk_size: int | None = None, 
        overlap: int | None = None
    ) -> list[dict[str, Any]]:
        """Chunk text into overlapping segments."""
        size = chunk_size or self.chunk_size
        overlap_size = overlap or self.chunk_overlap
        
        if len(text) <= size:
            return [{"text": text, "chunk_index": 0, "start_pos": 0, "end_pos": len(text)}]
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + size
            
            # If we're not at the end, try to break at word boundary
            if end < len(text):
                # Look for the last space before the end position
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    "start_pos": start,
                    "end_pos": end,
                })
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - overlap_size)
        
        return chunks

    def chunk_documents(
        self, 
        documents: list[str], 
        metadata: list[dict[str, Any]] | dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Chunk multiple documents with metadata.
        
        Args:
            documents: List of text documents to chunk
            metadata: Either:
                - list of dicts: per-document metadata (must match len(documents))
                - dict: global metadata applied to all documents  
                - None: no additional metadata
        """
        all_chunks = []
        
        # Handle different metadata formats
        if metadata is None:
            doc_metadatas = [{}] * len(documents)
        elif isinstance(metadata, list):
            if len(metadata) != len(documents):
                raise ValueError(f"Metadata list length ({len(metadata)}) must match documents length ({len(documents)})")
            doc_metadatas = metadata
        else:
            # Single dict - apply to all documents
            doc_metadatas = [metadata] * len(documents)
        
        for doc_index, (document, doc_metadata) in enumerate(zip(documents, doc_metadatas)):
            chunks = self.chunk_text(document)
            
            for chunk in chunks:
                chunk_metadata = {
                    **doc_metadata,  # Document-specific metadata first
                    "document_index": doc_index,
                    "chunk_index": chunk["chunk_index"],
                    "start_pos": chunk["start_pos"],
                    "end_pos": chunk["end_pos"],
                    "document_length": len(document),
                    "chunk_length": len(chunk["text"]),
                }
                
                all_chunks.append({
                    "text": chunk["text"],
                    "metadata": chunk_metadata,
                })
        
        return all_chunks

    def semantic_chunk_text(self, text: str) -> list[dict[str, Any]]:
        """Semantic chunking based on paragraph and sentence boundaries."""
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        chunk_index = 0
        
        for para_index, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If paragraph is small enough, use as-is
            if len(paragraph) <= self.chunk_size:
                chunks.append({
                    "text": paragraph,
                    "chunk_index": chunk_index,
                    "semantic_type": "paragraph",
                    "paragraph_index": para_index,
                })
                chunk_index += 1
            else:
                # Split large paragraphs by sentences
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk + sentence) <= self.chunk_size:
                        current_chunk += (" " if current_chunk else "") + sentence
                    else:
                        if current_chunk:
                            chunks.append({
                                "text": current_chunk.strip(),
                                "chunk_index": chunk_index,
                                "semantic_type": "sentences",
                                "paragraph_index": para_index,
                            })
                            chunk_index += 1
                        
                        # Start new chunk with current sentence
                        current_chunk = sentence
                
                # Add remaining text
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "chunk_index": chunk_index,
                        "semantic_type": "sentences",
                        "paragraph_index": para_index,
                    })
                    chunk_index += 1
        
        return chunks