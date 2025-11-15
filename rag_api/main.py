# main.py
import os
import asyncio
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
import sys

# Добавляем путь к embedder модулю (если нужно)
sys.path.append('/app')

# Импортируем embedder из соседнего сервиса (будет через volume или копирование)
try:
    from embedder import LocalGGUFEmbedder
except ImportError:
    print("[WARNING] LocalGGUFEmbedder not found, using simple embedding")
    # Fallback - простой embedder для тестов
    class LocalGGUFEmbedder:
        def __init__(self, model_path):
            print(f"[EMBEDDER] Dummy embedder initialized")
        
        async def embed(self, texts: List[str]):
            import numpy as np
            # Фейковые эмбеддинги для тестов
            return np.random.randn(len(texts), 768).astype(float)

# Настройки
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/embeddinggemma-300M-Q8_0.gguf")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "documents")

app = FastAPI(title="RAG Search API")

# Глобальные объекты (инициализируются при старте)
embedder = None
qdrant_client = None


@app.on_event("startup")
async def startup_event():
    global embedder, qdrant_client
    
    print(f"[API] Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    print(f"[API] Loading embedder from {MODEL_PATH}")
    embedder = LocalGGUFEmbedder(MODEL_PATH)
    
    print("[API] Startup complete")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    score: float
    text: str
    file: str
    chunk: int


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """
    Поиск релевантных документов по запросу
    """
    # 1. Получаем embedding для запроса
    query_vectors = await embedder.embed([request.query])
    query_vector = query_vectors[0].tolist()
    
    # 2. Ищем в Qdrant
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=request.top_k
    )
    
    # 3. Формируем ответ
    response = []
    for hit in results:
        payload = hit.payload or {}
        response.append(SearchResult(
            score=hit.score,
            text=payload.get("text", ""),
            file=payload.get("file", "unknown"),
            chunk=payload.get("chunk", 0)
        ))
    
    return response


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/collection/info")
async def collection_info():
    """
    Получить информацию о коллекции: количество векторов, размерность и т.д.
    """
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        return {
            "collection_name": COLLECTION_NAME,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "config": {
                "vector_size": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else None
            }
        }
    except Exception as e:
        return {"error": str(e), "collection_name": COLLECTION_NAME}


@app.get("/collection/sample")
async def collection_sample(limit: int = 5):
    """
    Получить несколько случайных векторов из коллекции для проверки
    """
    try:
        results = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
        
        points = []
        for point in results[0]:  # results is (points, next_page_offset)
            points.append({
                "id": point.id,
                "file": point.payload.get("file", "unknown"),
                "chunk": point.payload.get("chunk", 0),
                "text_preview": point.payload.get("text", "")[:100] + "..." if point.payload.get("text") else ""
            })
        
        return {
            "total_points": len(points),
            "points": points
        }
    except Exception as e:
        return {"error": str(e)}
