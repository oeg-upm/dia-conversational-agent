import chromadb
from chromadb.config import Settings as ChromaSettings
from .config import settings

def get_collection():
    client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(name=settings.CHROMA_COLLECTION)