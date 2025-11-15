# embedder.py
import asyncio
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class LocalGGUFEmbedder:
    """
    Wrapper around sentence-transformers for Gemma embedding model.
    expose: async embed(list[str]) -> np.ndarray (n, dim)
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        model_path: HuggingFace model name (e.g., "google/gemma-2-2b-it")
        device: "cuda" or "cpu"
        """
        self.model_path = model_path
        print(f"[EMBEDDER] Loading model from: {model_path}")
        self.model = SentenceTransformer(model_path, device=device, trust_remote_code=True)
        print("[EMBEDDER] Model loaded successfully")

    def _embed_sync(self, texts: List[str]):
        """
        Synchronous embedding call.
        Returns numpy.ndarray shape=(len(texts), dim)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

    async def embed(self, texts: List[str]):
        loop = asyncio.get_running_loop()
        # Run embedding in background thread to not block event loop
        vectors = await loop.run_in_executor(None, self._embed_sync, texts)
        return vectors
