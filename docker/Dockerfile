FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app && chown -R 1001:0 /app
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY datas datas
COPY src src
COPY pyproject.toml .
COPY uv.lock .
COPY docker/entrypoint.sh .

RUN uv sync

RUN chmod +x entrypoint.sh

EXPOSE 8000 8001

ENTRYPOINT ["/app/entrypoint.sh"]