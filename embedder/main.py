# main.py
import os
import asyncio
import time
from pathlib import Path
from typing import Dict
import signal

from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler

from embedder import LocalGGUFEmbedder
from vector_db import VectorDB

# Параметры (можно настроить через env)
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/embeddinggemma-300M-Q8_0.gguf")
WATCH_PATH = os.environ.get("WATCH_PATH", "/data/documents")
DEBOUNCE_SEC = float(os.environ.get("DEBOUNCE_SEC", "1.0"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1024"))   # символы
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "128"))

# --- helper chunking ---
def split_text_into_chunks(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end == L:
            break
        start = max(0, end - overlap)
    return chunks

# --- Watcher with debounce ---
class DebounceHandler(FileSystemEventHandler):
    """
    Watchdog handler that debounces events and pushes stabilized files to asyncio.Queue.
    Watchdog events happen in a separate thread, so we use loop.call_soon_threadsafe to schedule
    the debounce resets in the asyncio loop.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, delay: float = DEBOUNCE_SEC):
        self.loop = loop
        self.queue = queue
        self.delay = delay
        # mapping Path -> asyncio.TimerHandle
        self.timers = {}

    def on_any_event(self, event):
        # Ignore directories
        if event.is_directory:
            return

        path = Path(event.src_path)
        # schedule / reset timer for this path in event loop
        self.loop.call_soon_threadsafe(self._reset_timer, path)

    def _reset_timer(self, path: Path):
        # cancel existing timer
        handle = self.timers.get(path)
        if handle:
            handle.cancel()

        # create new timer
        handle = self.loop.call_later(self.delay, lambda: asyncio.create_task(self._notify(path)))
        self.timers[path] = handle

    async def _notify(self, path: Path):
        # remove timer handle
        self.timers.pop(path, None)
        # push to queue (only if file exists)
        if path.exists():
            print(f"[WATCHER] Stable event for {path}")
            await self.queue.put({"file": str(path), "event": "modified"})
        else:
            # If file deleted - still notify so worker can remove from DB
            print(f"[WATCHER] File disappeared (deleted?) {path}")
            await self.queue.put({"file": str(path), "event": "deleted"})

# --- ingestion worker ---
async def ingestion_worker(queue: asyncio.Queue, embedder: LocalGGUFEmbedder, vector_db: VectorDB):
    print("[WORKER] Worker started")
    while True:
        task = await queue.get()
        file_path = task.get("file")
        ev = task.get("event")
        skip_if_exists = task.get("skip_if_exists", False)
        print(f"[WORKER] Got task: {ev} -> {file_path}")

        path = Path(file_path)
        if ev == "deleted":
            # удаляем все chunks, связанные с этим файлом
            try:
                vector_db.delete_by_file(file_path)
                print(f"[WORKER] Deleted vectors for {file_path}")
            except Exception as e:
                print(f"[WORKER] Error deleting vectors for {file_path}: {e}")
            queue.task_done()
            continue

        # ev == modified / created -> process file
        if not path.exists():
            print(f"[WORKER] File not found: {file_path}")
            queue.task_done()
            continue

        # Skip if file already indexed (for initial scan)
        if skip_if_exists and vector_db.file_exists(file_path):
            print(f"[WORKER] Skipping already indexed file: {file_path}")
            queue.task_done()
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[WORKER] Error reading file {file_path}: {e}")
            queue.task_done()
            continue

        chunks = split_text_into_chunks(text)
        if not chunks:
            print(f"[WORKER] No chunks to process for {file_path}")
            queue.task_done()
            continue

        # Optional: delete previous vectors for this file (keeps DB clean)
        try:
            vector_db.delete_by_file(file_path)
        except Exception as e:
            print(f"[WORKER] Error deleting old vectors: {e}")

        try:
            # batch embed - llama embed is blocking, wrapped as async in embedder.embed()
            vectors = await embedder.embed(chunks)
        except Exception as e:
            print(f"[WORKER] Embedding error for {file_path}: {e}")
            queue.task_done()
            continue

        try:
            vector_db.upsert(file_path, chunks, vectors)
            print(f"[WORKER] Upserted {len(chunks)} chunks for {file_path}")
        except Exception as e:
            print(f"[WORKER] Upsert error for {file_path}: {e}")

        queue.task_done()

# --- main runner ---
async def main():
    # create queue
    queue = asyncio.Queue()

    # create embedder
    embedder = LocalGGUFEmbedder(MODEL_PATH)

    # connect to vector DB
    vector_db = VectorDB()
    
    # Get embedding dimension and ensure collection exists
    print("[MAIN] Initializing collection...")
    try:
        # Get a test embedding to determine dimension
        test_vectors = await embedder.embed(["test"])
        dim = test_vectors.shape[1]
        vector_db.ensure_collection(dim=dim)
        print(f"[MAIN] Collection '{vector_db.collection_name}' ready (dim={dim})")
    except Exception as e:
        print(f"[MAIN] Error initializing collection: {e}")
        raise

    # Ensure watch path exists
    watch_path = Path(WATCH_PATH)
    watch_path.mkdir(parents=True, exist_ok=True)
    
    # Initial scan: index all existing files
    print(f"[MAIN] Scanning {WATCH_PATH} for existing files...")
    existing_files = list(watch_path.rglob("*"))
    file_count = 0
    for file_path in existing_files:
        if file_path.is_file():
            print(f"[MAIN] Queuing existing file: {file_path}")
            # Don't skip existing files - just reindex everything on startup
            await queue.put({"file": str(file_path), "event": "modified", "skip_if_exists": False})
            file_count += 1
    print(f"[MAIN] Found {file_count} existing files to index")

    # start worker task(s)
    worker_task = asyncio.create_task(ingestion_worker(queue, embedder, vector_db))

    # start watcher in separate thread via watchdog Observer
    loop = asyncio.get_running_loop()
    event_handler = DebounceHandler(loop, queue, delay=DEBOUNCE_SEC)
    observer = PollingObserver()  # Use PollingObserver for Docker volumes
    observer.schedule(event_handler, WATCH_PATH, recursive=True)
    observer.start()
    print(f"[MAIN] Started watcher on {WATCH_PATH} (debounce {DEBOUNCE_SEC}s)")


    # trap signals to stop cleanly
    stop = asyncio.Event()
    def _signal_handler():
        print("[MAIN] Received stop signal")
        stop.set()
    for s in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(s, _signal_handler)

    # wait until signal
    await stop.wait()

    # shutdown
    print("[MAIN] Stopping observer and worker...")
    observer.stop()
    observer.join(timeout=2)
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())
