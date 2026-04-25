import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_ping(client):
    response = client.get("/ping")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "service" in data


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "checks" in data


def test_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "uptime" in data


def test_debug_config(client):
    response = client.get("/debug/config")
    assert response.status_code == 200
    data = response.json()
    assert "app" in data
    assert "bootstrap" in data


def test_index_documents(client):
    request_data = {
        "documents": ["This is a test document.", "Another test document."],
        "metadata": {"source": "test"}
    }
    response = client.post("/index", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "indexed_documents" in data
    assert data["indexed_documents"] == 2


def test_query_documents(client):
    # First index some documents
    index_data = {
        "documents": ["The capital of France is Paris.", "Python is a programming language."],
        "metadata": {"source": "test"}
    }
    client.post("/index", json=index_data)
    
    # Then query
    query_data = {
        "query": "What is the capital of France?",
        "k": 5,
        "filters": {}
    }
    response = client.post("/query", json=query_data)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data


def test_get_sources(client):
    # First index some documents
    index_data = {
        "documents": ["Test document about machine learning.", "Another document about AI."],
        "metadata": {"source": "test"}
    }
    client.post("/index", json=index_data)
    
    # Then get sources
    response = client.get("/sources?query=machine learning&k=5")
    assert response.status_code == 200
    data = response.json()
    assert "chunks" in data
    assert "query" in data


def test_collection_stats(client):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "document_count" in data