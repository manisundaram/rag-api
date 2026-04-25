from __future__ import annotations

import logging
import traceback
from fastapi.encoders import jsonable_encoder
from ai_service_kit.health import SimpleHealthResponse, check_health, get_diagnostics, get_metrics, ping_service
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .bootstrap import build_service_context, debug_snapshot
from .config import get_settings
from .models import IndexRequest, QueryRequest, QueryResponse, SourcesResponse, ErrorResponse
from .rag import RAGPipeline


def create_app() -> FastAPI:
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_api.log')
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== STARTING APPLICATION CREATION ===")
        
        logger.info("Step 1: Getting settings...")
        settings = get_settings()
        logger.info(f"Settings loaded: {settings.app_name}")
        
        logger.info("Step 2: Building service context...")
        service_context = build_service_context(settings)
        logger.info("Service context built")
        
        logger.info("Step 3: Creating RAG pipeline...")
        # 🔴 SET BREAKPOINT HERE - This is likely where it crashes
        rag_pipeline = RAGPipeline(settings)
        logger.info("RAG pipeline created")
        
    except Exception as startup_error:
        logger.error(f"STARTUP FAILED: {startup_error}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Don't re-raise, create a minimal app instead
        settings = get_settings()
        service_context = None
        rag_pipeline = None

    app = FastAPI(
        title=settings.app_name,
        description="RAG-based question answering API",
        version=settings.app_version,
        debug=settings.app_debug,
        openapi_tags=[
            {
                "name": "Health",
                "description": "Health monitoring, diagnostics, and service status endpoints",
            },
            {
                "name": "RAG Pipeline", 
                "description": "Core RAG functionality: indexing, querying, and document operations",
            },
            {
                "name": "Debug",
                "description": "Development and debugging endpoints for troubleshooting",
            },
        ]
    )
    app.state.settings = settings
    app.state.service_context = service_context
    app.state.rag_pipeline = rag_pipeline

    if settings.enable_cors and settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Health and monitoring endpoints (from template)
    @app.get("/ping", tags=["Health"])
    async def ping() -> dict[str, object]:
        return jsonable_encoder(ping_service(app.state.service_context))

    @app.get("/health", tags=["Health"])
    async def health() -> dict[str, object]:
        return jsonable_encoder(await check_health(app.state.service_context))

    @app.get("/diagnostics", tags=["Health"])
    async def diagnostics() -> dict[str, object]:
        return jsonable_encoder(await get_diagnostics(app.state.service_context))

    @app.get("/metrics", tags=["Health"])
    async def metrics() -> dict[str, object]:
        return jsonable_encoder(get_metrics(app.state.service_context))

    @app.get("/debug/config", tags=["Health"])
    async def debug_config() -> dict[str, object]:
        return jsonable_encoder(
            {
                "app": dict(settings.masked_debug_config()),
                "bootstrap": debug_snapshot(app.state.service_context),
            }
        )

    # RAG endpoints
    @app.post("/index", response_model=dict[str, object], tags=["RAG Pipeline"])
    async def index_documents(request: IndexRequest):
        """Index documents for retrieval."""
        if app.state.rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG pipeline failed to initialize")
            
        try:
            # Add debug breakpoint here if needed
            # breakpoint()  # Uncomment this line to add a breakpoint
            
            result = await app.state.rag_pipeline.index_documents(
                documents=request.documents,
                metadata=request.metadata
            )
            return result
        except Exception as e:
            logging.error(f"Index documents error: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            
            raise HTTPException(
                status_code=500,
                detail={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc().split('\n')
                }
            )

    @app.post("/query", response_model=QueryResponse, tags=["RAG Pipeline"])
    async def query_documents(request: QueryRequest):
        """Query documents and generate an answer."""
        try:
            # Add debug breakpoint here if needed
            # breakpoint()  # Uncomment this line to add a breakpoint
            
            response = await app.state.rag_pipeline.query(
                query=request.query,
                k=request.k,
                filters=request.filters,
                use_hybrid=True,
                rerank_method="llm_scoring",
                include_sources=True
            )
            return response
        except Exception as e:
            logging.error(f"Query documents error: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            
            raise HTTPException(
                status_code=500,
                detail={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc().split('\n')
                }
            )

    @app.get("/debug/startup-status", tags=["Debug"])
    async def startup_status():
        """Check if all components initialized properly."""
        return {
            "settings_loaded": app.state.settings is not None,
            "service_context_loaded": app.state.service_context is not None, 
            "rag_pipeline_loaded": app.state.rag_pipeline is not None,
            "startup_error_check": "Check rag_api.log for detailed startup errors"
        }

    @app.get("/sources", response_model=SourcesResponse, tags=["RAG Pipeline"])
    async def get_sources(query: str, k: int = 10, filters: str = "{}"):
        """Get retrieved source chunks for debugging - FIXED VERSION."""
        if app.state.rag_pipeline is None:
            raise HTTPException(status_code=503, detail="RAG pipeline failed to initialize - check /debug/startup-status")
            
        try:
            # 🔴 SET BREAKPOINT HERE (click in left margin)
            import json
            import chromadb
            from chromadb.config import Settings as ChromaSettings
            
            parsed_filters = json.loads(filters) if filters != "{}" else {}
            
            # 🔴 SET BREAKPOINT HERE 
            # Use the EXACT approach from working isolated test
            chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # 🔴 SET BREAKPOINT HERE 
            # Create/get collection with explicit embedding_function=None
            collection_name = "fixed_collection"
            try:
                # Try to get existing collection
                collection = chroma_client.get_collection(collection_name)
            except Exception:
                # 🔴 SET BREAKPOINT HERE 
                # Create new collection without auto-embedding
                collection = chroma_client.create_collection(
                    name=collection_name,
                    metadata={"description": "Fixed RAG collection"},
                    embedding_function=None
                )
                # Add some test data if collection is new
                provider = app.state.rag_pipeline.embedding_provider
                test_docs = ["Machine learning is a subset of AI.", "Python is great for data science."]
                test_embeddings = await provider.embed_batch(test_docs)
                collection.add(
                    documents=test_docs,
                    metadatas=[{"source": "test"}, {"source": "test"}],
                    ids=["doc1", "doc2"],
                    embeddings=test_embeddings
                )
            
            # 🔴 SET BREAKPOINT HERE 
            # Generate query embedding and search
            provider = app.state.rag_pipeline.embedding_provider
            query_embedding = await provider.embed_text(query)
            
            # 🔴 SET BREAKPOINT HERE 
            # Query with explicit embeddings (not text)
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to our format
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
            
            return SourcesResponse(
                chunks=chunks,
                query=query,
                filters=parsed_filters
            )
            
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in filters parameter"
            )
        except Exception as e:
            # Log the full error with stack trace
            logging.error(f"Sources endpoint error: {str(e)}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            
            # Return detailed error for debugging
            raise HTTPException(
                status_code=500,
                detail={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc().split('\n'),
                    "debug_info": "Using direct ChromaDB approach"
                }
            )

    @app.post("/debug/chromadb", tags=["Debug"])
    async def debug_chromadb():
        """Test ChromaDB with the exact same pattern as the working isolated test."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        try:
            # Create client exactly like isolated test
            chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Delete and create collection exactly like isolated test
            collection_name = "test_rag_collection"
            try:
                chroma_client.delete_collection(collection_name)
                print(f"Deleted existing {collection_name}")
            except:
                print(f"No existing {collection_name}")
            
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Test RAG collection"},
                embedding_function=None
            )
            print(f"Created collection: {collection.name}")
            
            # Generate embeddings using our provider
            embedding_provider = app.state.rag_pipeline.embedding_provider
            documents = ["Machine learning test document", "Python programming guide"]
            embeddings = await embedding_provider.embed_batch(documents)
            
            print(f"Generated embeddings with dimensions: {[len(emb) for emb in embeddings]}")
            
            # Add documents with embeddings
            collection.add(
                documents=documents,
                metadatas=[{"source": "test"}, {"source": "test"}],
                ids=["doc1", "doc2"],
                embeddings=embeddings
            )
            print("Added documents successfully")
            
            # Generate query embedding and test query
            query_embedding = await embedding_provider.embed_text("machine learning query")
            print(f"Query embedding dimension: {len(query_embedding)}")
            
            # Test query
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=2,
                include=["documents", "metadatas", "distances"]
            )
            
            return {
                "success": True,
                "message": "ChromaDB test successful",
                "embedding_dimensions": [len(emb) for emb in embeddings],
                "query_embedding_dimension": len(query_embedding),
                "collection_count": collection.count(),
                "query_results": len(results.get("documents", []))
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    @app.get("/debug/trace-error", tags=["Debug"])
    async def trace_error():
        """Debug endpoint to trace the exact collection error step by step."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        
        debug_info = []
        
        try:
            # Step 1: Test ChromaDB client creation
            debug_info.append("Step 1: Creating ChromaDB client...")
            chroma_client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            debug_info.append("✓ ChromaDB client created successfully")
            
            # Step 2: List existing collections
            debug_info.append("Step 2: Listing existing collections...")
            try:
                collections = chroma_client.list_collections()
                debug_info.append(f"✓ Found {len(collections)} collections: {[c.name for c in collections]}")
                
                # Show details about each collection
                for col in collections:
                    debug_info.append(f"  Collection: {col.name} (ID: {col.id})")
                    
            except Exception as list_e:
                debug_info.append(f"✗ Error listing collections: {list_e}")
            
            # Step 3: Test our RAG pipeline vectorstore initialization
            debug_info.append("Step 3: Testing RAG pipeline vectorstore...")
            try:
                vectorstore = app.state.rag_pipeline.vector_store
                debug_info.append(f"✓ Vectorstore object: {vectorstore}")
                debug_info.append(f"✓ Collection name: {vectorstore.collection_name}")
                debug_info.append(f"✓ Current _collection: {vectorstore._collection}")
                
                # Try to get collection
                debug_info.append("Step 4: Calling _get_collection()...")
                collection = vectorstore._get_collection()
                debug_info.append(f"✓ Got collection: {collection}")
                
            except Exception as vs_e:
                debug_info.append(f"✗ Error with vectorstore: {vs_e}")
                debug_info.append(f"✗ Vectorstore traceback: {traceback.format_exc()}")
            
            # Step 4: Test embedding generation
            debug_info.append("Step 5: Testing embedding generation...")
            try:
                provider = app.state.rag_pipeline.embedding_provider
                test_embedding = await provider.embed_text("test")
                debug_info.append(f"✓ Generated embedding with dimension: {len(test_embedding)}")
            except Exception as emb_e:
                debug_info.append(f"✗ Error generating embedding: {emb_e}")
                
            return {"debug_steps": debug_info, "success": True}
            
        except Exception as e:
            debug_info.append(f"✗ Fatal error: {str(e)}")
            debug_info.append(f"✗ Traceback: {traceback.format_exc()}")
            return {"debug_steps": debug_info, "success": False, "error": str(e)}

    @app.get("/debug/embeddings", tags=["Debug"])
    async def debug_embeddings():
        """Debug endpoint to test embedding dimensions and ChromaDB behavior."""
        try:
            # Test embedding generation
            provider = app.state.rag_pipeline.embedding_provider
            test_text = "machine learning test"
            embedding = await provider.embed_text(test_text)
            
            # Test ChromaDB collection info
            vector_store = app.state.rag_pipeline.vector_store
            collection = vector_store._get_collection()
            count = collection.count()
            
            # Try to peek at stored embeddings
            stored_dim = 0
            if count > 0:
                try:
                    peek_result = collection.peek(limit=1)
                    embeddings_data = peek_result.get("embeddings")
                    if embeddings_data is not None and len(embeddings_data) > 0:
                        stored_embedding = embeddings_data[0]
                        stored_dim = len(stored_embedding) if stored_embedding else 0
                except Exception:
                    stored_dim = "error_peeking"
                    
            return {
                "provider_type": app.state.rag_pipeline.settings.provider_type,
                "selected_model": app.state.rag_pipeline.settings.selected_provider_model(),
                "generated_embedding_dim": len(embedding),
                "collection_name": vector_store.collection_name,
                "collection_count": count,
                "stored_embedding_dim": stored_dim,
                "collection_metadata": collection.metadata if hasattr(collection, 'metadata') else None
            }
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }

    @app.get("/stats", tags=["RAG Pipeline"])
    async def get_collection_stats():
        """Get statistics about the indexed documents."""
        try:
            stats = await app.state.rag_pipeline.get_collection_stats()
            return stats
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get collection stats: {str(e)}"
            )

    @app.delete("/index", tags=["RAG Pipeline"])
    async def clear_index():
        """Clear all indexed documents."""
        try:
            success = await app.state.rag_pipeline.clear_index()
            if success:
                return {"status": "success", "message": "Index cleared successfully"}
            else:
                return {"status": "error", "message": "Failed to clear index"}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to clear index: {str(e)}"
            )

    @app.post("/similar", tags=["RAG Pipeline"])
    async def find_similar(document_text: str, k: int = 10, filters: str = "{}"):
        """Find documents similar to the given text."""
        try:
            import json
            parsed_filters = json.loads(filters) if filters != "{}" else {}
            
            similar_docs = await app.state.rag_pipeline.find_similar_documents(
                document_text=document_text,
                k=k,
                filters=parsed_filters
            )
            
            return {
                "similar_documents": similar_docs,
                "query_text": document_text,
                "filters": parsed_filters
            }
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in filters parameter"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to find similar documents: {str(e)}"
            )

    @app.get("/search/metadata", tags=["RAG Pipeline"])
    async def search_by_metadata(filters: str, limit: int = 100):
        """Search documents by metadata only."""
        try:
            import json
            parsed_filters = json.loads(filters)
            
            results = await app.state.rag_pipeline.search_by_metadata(
                filters=parsed_filters,
                limit=limit
            )
            
            return {
                "results": results,
                "filters": parsed_filters,
                "count": len(results)
            }
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON in filters parameter"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to search by metadata: {str(e)}"
            )

    @app.post("/debug/chunking", tags=["Debug"])
    async def debug_chunking():
        """Debug endpoint to see what chunks are being generated."""
        try:
            # Test the chunking process
            test_documents = ["Python is a high-level programming language."]
            test_metadata = [{"source":"test_guide","category":"programming"}]
            
            chunker = app.state.rag_pipeline.chunker
            chunks = chunker.chunk_documents(test_documents, test_metadata)
            
            return {
                "input_documents": test_documents,
                "input_metadata": test_metadata,
                "output_chunks": chunks,
                "chunk_count": len(chunks),
                "first_chunk_metadata": chunks[0].get("metadata", {}) if chunks else {}
            }
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }

    @app.get("/debug/collection-data", tags=["Debug"])
    async def debug_collection_data():
        """Debug endpoint to see all data in the collection."""
        try:
            vector_store = app.state.rag_pipeline.vector_store
            collection = vector_store._get_collection()
            
            # Get all documents in collection
            all_results = collection.get(
                include=["documents", "metadatas"],
                limit=100
            )
            
            # Show first document metadata for debugging
            sample_metadata = {}
            if all_results.get("metadatas") and len(all_results["metadatas"]) > 0:
                sample_metadata = all_results["metadatas"][0]
            
            # Test specific where clause
            test_where = {"category": {"$eq": "programming"}}
            filtered_results = collection.get(
                where=test_where,
                include=["documents", "metadatas"],
                limit=100
            )
            
            return {
                "total_documents": len(all_results.get("documents", [])),
                "sample_metadata": sample_metadata,
                "test_where_clause": test_where,
                "filtered_count": len(filtered_results.get("documents", [])),
                "has_category_in_sample": "category" in sample_metadata if sample_metadata else False
            }
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__
            }

    @app.post("/search/metadata", tags=["RAG Pipeline"])
    async def search_by_metadata_post(request: dict):
        """Search documents by metadata only - POST version."""
        try:
            filters = request.get("filters", {})
            limit = request.get("limit", 100)
            
            results = await app.state.rag_pipeline.search_by_metadata(
                filters=filters,
                limit=limit
            )
            
            return {
                "results": results,
                "filters": filters,
                "count": len(results)
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to search by metadata: {str(e)}"
            )

    return app


app = create_app()