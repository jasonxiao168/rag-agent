FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl 仅用于 healthcheck；编译工具用于 numpy 等依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY eval ./eval

# 索引与会话数据挂载到 /data，容器重建不丢
ENV DATA_DIR=/data \
    FASTEMBED_CACHE_DIR=/data/fastembed_cache
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
