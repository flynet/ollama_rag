# main.py
import os
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncIterator
import json

app = FastAPI(title="RAG Middleware")

# Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
RAG_API_URL = os.environ.get("RAG_API_URL", "http://rag-api:8000")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
ENABLE_RAG = os.environ.get("ENABLE_RAG", "true").lower() == "true"


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False


async def get_rag_context(query: str) -> str:
    """Get relevant context from RAG API"""
    if not ENABLE_RAG:
        return ""
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{RAG_API_URL}/search",
                json={"query": query, "top_k": RAG_TOP_K}
            )
            response.raise_for_status()
            results = response.json()
            
            if not results:
                return ""
            
            # Format context from search results
            context_parts = []
            for idx, result in enumerate(results, 1):
                text = result.get("text", "")
                file = result.get("file", "unknown")
                score = result.get("score", 0)
                
                # Only include relevant results (score > 0.5)
                if score > 0.5:
                    context_parts.append(f"[Document {idx} from {file}]:\n{text}")
            
            if context_parts:
                context = "\n\n".join(context_parts)
                return f"""Relevant information from knowledge base:

{context}

---
Please use the above information to answer the question if relevant. If the information doesn't help answer the question, you can provide a general response."""
            
    except Exception as e:
        print(f"[RAG] Error fetching context: {e}")
    
    return ""


async def stream_ollama_response(model: str, messages: List[dict]) -> AsyncIterator[bytes]:
    """Stream response from Ollama"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": True}
        ) as response:
            async for line in response.aiter_lines():
                if line.strip():
                    yield (line + "\n").encode()


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Proxy chat requests to Ollama with RAG enhancement
    """
    messages = [msg.model_dump() for msg in request.messages]
    
    # Get the last user message for RAG context
    last_user_message = None
    for msg in reversed(messages):
        if msg["role"] == "user":
            last_user_message = msg["content"]
            break
    
    # Add RAG context if available
    if last_user_message and ENABLE_RAG:
        context = await get_rag_context(last_user_message)
        
        if context:
            # Insert context before the last user message
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    # Enhance the user message with context
                    messages[i]["content"] = f"{context}\n\nUser question: {messages[i]['content']}"
                    break
    
    # Stream response from Ollama
    if request.stream:
        return StreamingResponse(
            stream_ollama_response(request.model, messages),
            media_type="application/x-ndjson"
        )
    else:
        # Non-streaming response
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": request.model, "messages": messages, "stream": False}
            )
            return response.json()


@app.get("/api/tags")
async def get_models():
    """Proxy model list from Ollama"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        return response.json()


@app.post("/api/generate")
async def generate(request: Request):
    """Proxy generate requests to Ollama"""
    body = await request.json()
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        if body.get("stream", False):
            async def stream_response():
                async with client.stream(
                    "POST",
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json=body
                ) as response:
                    async for line in response.aiter_lines():
                        if line.strip():
                            yield (line + "\n").encode()
            
            return StreamingResponse(stream_response(), media_type="application/x-ndjson")
        else:
            response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=body)
            return response.json()


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "rag_enabled": ENABLE_RAG,
        "ollama_url": OLLAMA_BASE_URL,
        "rag_api_url": RAG_API_URL
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
