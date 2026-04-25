# RAG API

A FastAPI-based microservice for Retrieval-Augmented Generation (RAG) built on top of `ai-service-kit`. This service provides document indexing, retrieval, and question-answering capabilities using vector embeddings and large language models.

## Features

- **Document Indexing**: Chunk and index documents with metadata
- **Hybrid Search**: Combines vector similarity and keyword search
- **Reranking**: Multiple reranking strategies (LLM-based, embedding similarity, keyword overlap)
- **Question Answering**: Generate answers based on retrieved context
- **Vector Storage**: ChromaDB integration for vector embeddings
- **Provider Support**: OpenAI, Google Gemini, and Anthropic Claude
- **Health Monitoring**: Built-in health checks and metrics
- **CORS Support**: Configurable cross-origin resource sharing

## Architecture

The RAG pipeline consists of several modular components:

- **Chunker** (`app/chunker.py`): Text segmentation with overlapping windows
- **Embeddings** (`app/embeddings.py`): Provider-agnostic embedding generation
- **Vector Store** (`app/vectorstore.py`): ChromaDB wrapper for vector operations
- **Retriever** (`app/retriever.py`): Hybrid search with vector + keyword retrieval
- **Reranker** (`app/reranker.py`): Multiple reranking strategies
- **RAG Pipeline** (`app/rag.py`): End-to-end RAG orchestration

## Quick Start

### Prerequisites

- Python 3.11+
- ChromaDB
- API keys for your chosen provider (OpenAI, Gemini, or Claude)

### Installation

```bash
# Clone and navigate to the project
cd rag-api

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file and configure
cp .env.example .env
# Edit .env with your API keys and settings
```

### Configuration

Edit `.env` file:

```env
# Provider Configuration
PROVIDER_TYPE=openai
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=text-embedding-3-small

# RAG Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
DEFAULT_K=10
DEFAULT_RERANK_K=5

# Vector Store
VECTORSTORE_BACKEND=chroma
DEFAULT_COLLECTION_NAME=documents
```

### Running the Service

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## API Endpoints

### Health & Monitoring

- `GET /ping` - Service ping
- `GET /health` - Health checks
- `GET /diagnostics` - Detailed diagnostics
- `GET /metrics` - Service metrics
- `GET /debug/config` - Configuration debug info

### RAG Operations

#### Index Documents

```bash
POST /index
Content-Type: application/json

{
  "documents": ["Document text 1", "Document text 2"],
  "metadata": {"source": "manual", "category": "tech"}
}
```

#### Query Documents

```bash
POST /query
Content-Type: application/json

{
  "query": "What is machine learning?",
  "k": 10,
  "filters": {"source": "manual"}
}
```

#### Get Sources (Debug)

```bash
GET /sources?query=machine learning&k=5&filters={"source":"manual"}
```

### Additional Endpoints

- `GET /stats` - Collection statistics
- `DELETE /index` - Clear all documents
- `POST /similar` - Find similar documents
- `GET /search/metadata` - Search by metadata only

## Usage Examples

### Basic Document Indexing and Querying

```python
import httpx

# Index documents
response = httpx.post("http://localhost:8001/index", json={
    "documents": [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science."
    ],
    "metadata": {"source": "tutorial", "topic": "tech"}
})

# Query for information
response = httpx.post("http://localhost:8001/query", json={
    "query": "What is machine learning?",
    "k": 5,
    "filters": {"source": "tutorial"}
})

answer_data = response.json()
print(f"Answer: {answer_data['answer']}")
for source in answer_data['sources']:
    print(f"Source: {source['text']} (Score: {source['score']})")
```

### Advanced Filtering

```python
# Index with metadata
httpx.post("http://localhost:8001/index", json={
    "documents": ["HR policy document", "Engineering guidelines"],
    "metadata": {"department": "hr", "confidentiality": "internal"}
})

# Query with filters
response = httpx.post("http://localhost:8001/query", json={
    "query": "What are the policies?",
    "filters": {"department": "hr", "confidentiality": "internal"}
})
```

## Configuration

### RAG Parameters

- `CHUNK_SIZE`: Maximum tokens per text chunk (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 50)
- `DEFAULT_K`: Default number of documents to retrieve (default: 10)
- `DEFAULT_RERANK_K`: Number of documents to rerank (default: 5)

### Provider Models

**OpenAI:**

- `text-embedding-3-small` (1536 dimensions)
- `text-embedding-3-large` (3072 dimensions)
- `text-embedding-ada-002` (1536 dimensions)

**Google Gemini:**

- `models/embedding-001` (768 dimensions)
- `models/text-embedding-004` (768 dimensions)

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Project Structure

```
rag-api/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app and endpoints
│   ├── config.py        # Configuration management
│   ├── bootstrap.py     # Service initialization
│   ├── models.py        # Pydantic models
│   ├── chunker.py       # Text chunking
│   ├── embeddings.py    # Embedding generation
│   ├── vectorstore.py   # ChromaDB interface
│   ├── retriever.py     # Document retrieval
│   ├── reranker.py      # Result reranking
│   └── rag.py           # Main RAG pipeline
├── tests/
│   ├── test_app.py
│   └── test_config.py
├── requirements.txt
├── pyproject.toml
├── .env
└── README.md
```

## Dependencies

- **Core**: FastAPI, Pydantic, Uvicorn
- **AI/ML**: ai-service-kit, ChromaDB, sentence-transformers
- **Testing**: pytest, httpx

## License

This project is built on top of the ai-service-kit framework.
