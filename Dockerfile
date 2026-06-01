FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1

# System packages: deadsnakes PPA로 Python 3.12 설치 (Ubuntu 22.04 기본은 3.10)
# CUDA 라이브러리는 PyTorch cu124 wheel에 번들되어 있으므로 별도 설치 불필요
# GPU 접근은 호스트의 NVIDIA Container Toolkit이 처리
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libgomp1 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Dependency install layer (cached unless pyproject.toml/uv.lock changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

# Copy project (data/, scripts/, results/, weights/ all included via COPY context)
COPY . .

# Default: interactive shell. Override CMD to run a specific training script.
# Example:
#   docker run --gpus all oct-ai uv run python scripts/09_augment/run_nafnet_aug.py --batch-size 48
CMD ["bash"]
