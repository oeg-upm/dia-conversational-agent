import os
import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from .config import settings
from .nextcloud_source import list_pdfs, download_pdf
from .storage import put_bytes, exists
from .parsing import parse_pdf_to_markdown
from .chunking import chunk_text
from .ollama_client import embed_text
from .vectorstore import get_collection

MANIFEST_PATH = "./data/manifest.json"

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_manifest() -> Dict[str, Any]:
    if not os.path.exists(MANIFEST_PATH):
        return {"version": 1, "docs": {}}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_manifest(m: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)

def upsert_chunks(doc_id: str, filename: str, source_url: str, chunks: List[str]) -> int:
    col = get_collection()
    ids = []
    embeddings = []
    metadatas = []
    documents = []

    for i, ch in enumerate(chunks):
        chunk_id = f"{doc_id}:{i}"
        ids.append(chunk_id)
        embeddings.append(embed_text(ch))
        metadatas.append({
            "doc_id": doc_id,
            "chunk_index": i,
            "filename": filename,
            "source_url": source_url,
        })
        documents.append(ch)

    # Upsert pattern: Chroma add() fails if ids exist; easiest MVP: try delete then add.
    # (En producción: usar update/upsert cuando aplique)
    try:
        col.delete(ids=ids)
    except Exception:
        pass
    col.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
    return len(ids)

def main():
    os.makedirs("./data/cache", exist_ok=True)

    manifest = load_manifest()
    docs = manifest["docs"]

    pdfs: List[Tuple[str, str]] = list_pdfs()
    print(f"Found {len(pdfs)} PDFs in source")

    indexed = 0
    skipped = 0

    for filename, url in pdfs:
        print(f"\n---\nProcessing: {filename}")

        pdf_bytes = download_pdf(url)
        doc_id = sha256_bytes(pdf_bytes)

        if doc_id in docs:
            print(f"⏭️  Skip (already indexed): {doc_id}")
            skipped += 1
            continue

        # 1) Store raw PDF in MinIO
        raw_key = f"{doc_id}.pdf"
        if not exists(settings.S3_BUCKET_RAW, raw_key):
            put_bytes(settings.S3_BUCKET_RAW, raw_key, pdf_bytes, content_type="application/pdf")
        print(f"✅ Uploaded raw PDF to MinIO: s3://{settings.S3_BUCKET_RAW}/{raw_key}")

        # 2) Save temp file for Docling
        tmp_path = os.path.join("./data/cache", raw_key)
        with open(tmp_path, "wb") as f:
            f.write(pdf_bytes)

        # 3) Parse with Docling → markdown
        md = parse_pdf_to_markdown(tmp_path)

        # 4) Store parsed artifact in MinIO
        parsed_key = f"{doc_id}/docling.md"
        put_bytes(settings.S3_BUCKET_PARSED, parsed_key, md.encode("utf-8"), content_type="text/markdown")
        print(f"✅ Uploaded parsed MD to MinIO: s3://{settings.S3_BUCKET_PARSED}/{parsed_key}")

        # 5) Chunking
        chunks = chunk_text(md, settings.CHUNK_SIZE_CHARS, settings.CHUNK_OVERLAP_CHARS)
        print(f"Chunks: {len(chunks)}")

        # 6) Index in Chroma
        n = upsert_chunks(doc_id, filename, url, chunks)
        print(f"✅ Indexed chunks in Chroma: {n}")

        # 7) Update manifest
        docs[doc_id] = {
            "filename": filename,
            "source_url": url,
            "raw_key": raw_key,
            "parsed_key": parsed_key,
            "chunks": len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        save_manifest(manifest)
        indexed += 1

    print("\n==== DONE ====")
    print(f"Indexed: {indexed}, Skipped: {skipped}, Total: {len(pdfs)}")

if __name__ == "__main__":
    main()