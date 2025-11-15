FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip build-essential git \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install sentence-transformers and other requirements
RUN pip3 install --no-cache-dir sentence-transformers transformers accelerate qdrant-client numpy watchdog

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python3", "main.py"]
