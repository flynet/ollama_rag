# vector_db.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector
from typing import List
import numpy as np
import hashlib

class VectorDB:
    def __init__(self, host="qdrant", port=6333, collection_name="documents"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def ensure_collection(self, dim: int, distance: Distance = Distance.COSINE):
        # Создаём коллекцию, если её нет
        try:
            self.client.get_collection(self.collection_name)
            # Если существует — ничего не делаем
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=distance)
            )
    
    def _generate_id(self, file_path: str, chunk_index: int) -> int:
        """Generate unique integer ID from file path and chunk index"""
        text = f"{file_path}::chunk::{chunk_index}"
        hash_obj = hashlib.md5(text.encode())
        # Convert first 8 bytes to int (positive)
        return int.from_bytes(hash_obj.digest()[:8], 'big')

    def upsert(self, file_path: str, chunks: List[str], vectors: np.ndarray):
        """
        upsert chunks->vectors. vectors: numpy array shape (n, dim)
        В payload сохраняем file и chunk index.
        """
        if len(chunks) == 0:
            return

        dim = vectors.shape[1]
        # ensure collection
        self.ensure_collection(dim=dim)

        points = []
        # Use hash-based integer ID
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = self._generate_id(file_path, i)
            points.append(PointStruct(
                id=pid, 
                payload={"file": file_path, "chunk": i, "text": chunk}, 
                vector=vec.tolist()
            ))

        # batch upsert
        self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_by_file(self, file_path: str):
        """
        Удаляем все точки, у которых payload.file == file_path
        """
        flt = Filter(must=[FieldCondition(key="file", match=MatchValue(value=file_path))])
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=flt)
        )

    def file_exists(self, file_path: str) -> bool:
        """
        Проверяем, есть ли уже векторы для этого файла в базе
        """
        try:
            # Try to count points with this file path
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(must=[FieldCondition(key="file", match=MatchValue(value=file_path))]),
                exact=False
            )
            return result.count > 0
        except Exception as e:
            print(f"[DB] Error checking file existence: {e}")
            return False
