FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake both embedding models into the image — avoids ~170MB of downloads on every cold start.
# sentence-transformers: used directly by our code.
# chromadb default EF (ONNX): pre-downloaded so any chromadb internal path is covered.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()"

COPY . .

RUN chmod +x start.sh

EXPOSE 8501

CMD ["./start.sh"]
