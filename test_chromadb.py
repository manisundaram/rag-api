#!/usr/bin/env python3
"""
Simple ChromaDB test to isolate the embedding dimension issue.
"""

import chromadb
from chromadb.config import Settings as ChromaSettings
import openai
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_chromadb_dimensions():
    """Test ChromaDB with OpenAI embeddings to isolate dimension issues."""
    
    # Initialize OpenAI client
    client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Create ChromaDB client
    chroma_client = chromadb.PersistentClient(
        path="./test_chroma_db",
        settings=ChromaSettings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )
    
    # Delete collection if it exists
    try:
        chroma_client.delete_collection("test_collection")
        print("Deleted existing test collection")
    except:
        print("No existing test collection")
    
    # Create collection WITHOUT embedding function
    collection = chroma_client.create_collection(
        name="test_collection",
        metadata={"description": "Test collection"},
        embedding_function=None  # Explicitly disable auto-embedding
    )
    print(f"Created collection: {collection.name}")
    
    # Generate OpenAI embedding
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=["test document about machine learning"]
    )
    embedding = response.data[0].embedding
    print(f"OpenAI embedding dimension: {len(embedding)}")
    
    # Add document with embedding
    collection.add(
        documents=["test document about machine learning"],
        metadatas=[{"source": "test"}],
        ids=["doc1"],
        embeddings=[embedding]
    )
    print("Added document with embedding")
    
    # Generate query embedding
    query_response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=["machine learning query"]
    )
    query_embedding = query_response.data[0].embedding
    print(f"Query embedding dimension: {len(query_embedding)}")
    
    # Test query with embeddings
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"]
        )
        print("Query successful!")
        print(f"Results: {results}")
    except Exception as e:
        print(f"Query failed: {e}")
        
        # Try to query with texts to see auto-embedding behavior
        print("Testing auto-embedding behavior...")
        try:
            auto_results = collection.query(
                query_texts=["machine learning query"],
                n_results=1,
                include=["documents", "metadatas", "distances"]
            )
            print(f"Auto-embedding query worked: {auto_results}")
        except Exception as auto_e:
            print(f"Auto-embedding query also failed: {auto_e}")

if __name__ == "__main__":
    asyncio.run(test_chromadb_dimensions())