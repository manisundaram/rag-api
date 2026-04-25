from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    documents: list[str] = Field(..., description="List of text documents to index")
    metadata: list[dict[str, Any]] | dict[str, Any] | None = Field(
        default=None, 
        description="Metadata: list of dicts (per-document) or single dict (global) or None"
    )


class QueryRequest(BaseModel):
    query: str = Field(..., description="Query text to search for")
    k: int = Field(default=10, description="Number of documents to retrieve")
    filters: dict[str, Any] = Field(default_factory=dict, description="Metadata filters to apply")


class RetrievedChunk(BaseModel):
    text: str = Field(..., description="The retrieved text chunk")
    score: float = Field(..., description="Relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Associated metadata")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer based on retrieved context")
    sources: list[RetrievedChunk] = Field(..., description="Source chunks used to generate the answer")


class SourcesResponse(BaseModel):
    chunks: list[RetrievedChunk] = Field(..., description="All retrieved chunks for debugging")
    query: str = Field(..., description="Original query")
    filters: dict[str, Any] = Field(default_factory=dict, description="Applied filters")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Additional error details")


class StatusResponse(BaseModel):
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    details: dict[str, Any] = Field(default_factory=dict, description="Additional status details")