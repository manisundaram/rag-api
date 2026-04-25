# RAG API

A production-ready FastAPI microservice implementing a complete Retrieval-Augmented Generation (RAG) pipeline with multiple embedding providers, vector storage, and intelligent document retrieval.

## 🚀 Features

- **Complete RAG Pipeline**: Document chunking, embedding generation, vector storage, and retrieval
- **Multiple Embedding Providers**: OpenAI, Google Gemini with easy extensibility  
- **Vector Database**: ChromaDB for efficient similarity search
- **Smart Reranking**: Multiple reranking strategies for improved relevance
- **Health Monitoring**: Comprehensive health checks and diagnostics
- **Debug Tools**: Built-in debugging endpoints for development
- **Environment-based Configuration**: Secure API key management
- **VS Code Ready**: Pre-configured debugging and development setup

## 🏗️ Architecture

```
┌─────────────────┬─────────────────┬─────────────────┐
│   FastAPI App   │  RAG Pipeline   │  Vector Store   │
├─────────────────┼─────────────────┼─────────────────┤
│ • Health        │ • Document      │ • ChromaDB      │
│ • RAG Endpoints │   Chunking      │ • Persistence   │
│ • Debug Tools   │ • Embeddings    │ • Collections   │
│ • Auto Docs     │ • Retrieval     │ • Metadata      │
└─────────────────┴─────────────────┴─────────────────┘
```

## 📋 API Endpoints

### Health & Monitoring
- `GET /ping` - Basic service health check
- `GET /health` - Detailed health status with dependencies
- `GET /diagnostics` - Comprehensive system diagnostics
- `GET /metrics` - Performance and usage metrics

### RAG Pipeline  
- `POST /index` - Index documents for retrieval
- `POST /query` - Query with RAG for generated answers
- `GET /sources` - Retrieve relevant document sources
- `GET /stats` - Collection statistics and insights
- `POST /similar` - Find similar documents
- `POST /search/metadata` - Search with metadata filtering

### Debug & Development
- `GET /debug/startup-status` - Service initialization status
- `GET /debug/config` - Current configuration settings
- `GET /debug/embeddings` - Test embedding providers
- `GET /debug/trace-error` - Error tracing for debugging
- `GET /debug/chunking` - Test document chunking
- `GET /debug/collection-data` - Inspect vector data
- `GET /debug/chromadb` - ChromaDB connection testing

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/manisundaram/rag-api.git
   cd rag-api
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Start the server**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

6. **Access the API**
   - API: http://localhost:8000
   - Interactive Docs: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

## ⚙️ Configuration

### Environment Variables

```env
# Provider Selection
PROVIDER_TYPE=openai  # openai, gemini, or claude

# API Keys
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
ANTHROPIC_API_KEY=your_anthropic_key

# Server Configuration  
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# RAG Settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
DEFAULT_K=10
DEFAULT_RERANK_K=5
```

### Supported Providers

| Provider | Model | Dimensions | Status |
|----------|-------|------------|--------|
| OpenAI | text-embedding-3-small | 1536 | ✅ Active |
| Google Gemini | gemini-embedding-001 | 768 | ✅ Active |
| Anthropic Claude | claude-3-5-haiku-latest | TBD | 🔄 Planned |

## 📖 Usage Examples

### Index Documents
```bash
curl -X POST "http://localhost:8000/index" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "content": "FastAPI is a modern web framework for Python",
        "metadata": {"source": "docs", "topic": "web"}
      }
    ]
  }'
```

### Query with RAG
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is FastAPI?",
    "k": 5
  }'
```

### Search with Metadata
```bash
curl -X POST "http://localhost:8000/search/metadata" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "web framework",
    "metadata_filter": {"topic": "web"},
    "k": 3
  }'
```

## 🛠️ Development

### VS Code Setup
The project includes VS Code configuration for debugging:

1. Open in VS Code
2. Install recommended extensions
3. Set breakpoints in code
4. Run "Debug RAG API" configuration

### Testing
```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=app

# Test specific functionality
python test_chromadb.py
```

### Debug Endpoints
Use debug endpoints during development:

```bash
# Test embedding providers
curl http://localhost:8000/debug/embeddings

# Check configuration
curl http://localhost:8000/debug/config

# Inspect vector data
curl http://localhost:8000/debug/collection-data
```

## 🏢 Technology Stack

- **API Framework**: FastAPI with automatic OpenAPI docs
- **Vector Database**: ChromaDB with persistent storage  
- **Embedding Providers**: OpenAI, Google Gemini APIs
- **Language Models**: Integration ready for various LLMs
- **Configuration**: Pydantic settings with environment variables
- **Development**: VS Code debugging, pytest testing
- **Deployment**: Docker ready, uvicorn ASGI server

## 🔒 Security

- ✅ **No hardcoded secrets** - All API keys via environment variables
- ✅ **Clean git history** - No sensitive data in commits  
- ✅ **Proper .gitignore** - Database files and logs excluded
- ✅ **Environment isolation** - .env files for configuration

## 📁 Project Structure

```
rag-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── rag.py              # RAG pipeline orchestration  
│   ├── embeddings.py       # Embedding provider factory
│   ├── embedding_providers.py # Provider implementations
│   ├── vectorstore.py      # ChromaDB wrapper
│   ├── chunker.py          # Document chunking
│   ├── retriever.py        # Document retrieval
│   ├── reranker.py         # Result reranking
│   └── models.py           # Pydantic models
├── tests/
├── .vscode/
│   └── launch.json         # VS Code debug config
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project metadata
└── README.md              # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- [ai-service-kit](https://github.com/your-org/ai-service-kit) - Shared embedding provider framework
- [ai-service-template](https://github.com/your-org/ai-service-template) - FastAPI microservice template

---

**Built with ❤️ using FastAPI and modern Python practices**

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
