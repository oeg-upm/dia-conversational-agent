FROM python:3.11-slim
 
WORKDIR /app
 
# System deps for EasyOCR and Docling
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
 
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy the whole project so "from backend.rag import BasicRAG" works
COPY . .
 
EXPOSE 9000
 
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "9000"]