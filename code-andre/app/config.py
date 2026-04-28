from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    # Source
    SOURCE_URL: str = os.getenv("SOURCE_URL", "")
    SOURCE_PASSWORD: str = os.getenv("SOURCE_PASSWORD", "")

    # MinIO
    S3_ENDPOINT: str = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    S3_ACCESS_KEY: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    S3_SECRET_KEY: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    S3_REGION: str = os.getenv("S3_REGION", "us-east-1")

    S3_BUCKET_RAW: str = os.getenv("S3_BUCKET_RAW", "docs-raw")
    S3_BUCKET_PARSED: str = os.getenv("S3_BUCKET_PARSED", "docs-parsed")
    S3_BUCKET_LOGS: str = os.getenv("S3_BUCKET_LOGS", "logs")

    # Chroma
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "tfm_docs_v1")

    # Ollama
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b-instruct")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # RAG
    TOP_K: int = int(os.getenv("TOP_K", "10"))
    CHUNK_SIZE_CHARS: int = int(os.getenv("CHUNK_SIZE_CHARS", "1200"))
    CHUNK_OVERLAP_CHARS: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))

    # Logs
    LOG_DIR: str = os.getenv("LOG_DIR", "./data/logs")

settings = Settings()