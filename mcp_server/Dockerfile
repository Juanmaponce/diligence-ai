FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "mcp[cli]==1.7.1" \
    httpx==0.27.2 \
    chromadb==0.5.15

COPY server.py .

CMD ["python", "server.py"]
