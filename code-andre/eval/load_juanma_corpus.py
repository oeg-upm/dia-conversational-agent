"""
load_juanma_corpus.py
=====================
Carga los PDFs de Juanma necesarios para evaluar el dataset humanizado.
Los sube al backend con los metadatos correctos (course, degree, source)
para que la recuperación RAG funcione con los mismos filtros que en el dataset.

Uso:
  # Con el backend corriendo (uvicorn rag_backend_lite:app --port 8001):
  python load_juanma_corpus.py

  # Para una ChromaDB diferente (otra configuración de embeddings):
  CHROMA_DIR=./chroma_juanma_bge python load_juanma_corpus.py
"""

import json
import re
import sys
import time
from pathlib import Path

import requests

# ── Configuración ─────────────────────────────────────────────────────────────
BACKEND_URL  = "http://127.0.0.1:8001"
DATASET_PATH = Path(__file__).parent.parent / "dataset" / "rag_dataset_humanized_v1.json"
GUIDES_BASE  = Path("/Users/andrep/Downloads/Guías aprendizaje")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_chunk_id(chunk_id: str, source_document: str) -> dict:
    """Extrae course, degree, source de un chunk_id del dataset de Juanma."""
    base = re.sub(r"_ch_\d+$", "", chunk_id)
    m = re.match(r"^(Curso \d{4}_\d{4})_", base)
    course = m.group(1) if m else "Unknown"
    rest = base[len(course) + 1:]
    dm = re.match(rf"^(.+)_{re.escape(source_document)}$", rest)
    degree = dm.group(1) if dm else rest
    return {"course": course, "degree": degree, "source": source_document}


def find_pdf(course: str, degree: str, source: str) -> Path | None:
    """
    Localiza el PDF en la carpeta de Guías aprendizaje.
    El directorio de año usa ':' (macOS): 'Curso 2023_2024' → 'Curso 2023:2024'.
    El nombre del fichero puede tener '_' o ':' dependiendo de cómo fue guardado.
    """
    course_dir = course.replace("_", ":", 1)   # Curso 2023_2024 → Curso 2023:2024

    # Búsqueda exacta primero
    candidates = list(GUIDES_BASE.glob(f"**/{source}"))
    for p in candidates:
        if course_dir in str(p):
            return p

    # Fallback: reemplazar '_' por ':' en el nombre del archivo (quirk de macOS)
    source_colon = source.replace("_", ":")
    candidates2 = list(GUIDES_BASE.glob(f"**/{source_colon}"))
    for p in candidates2:
        if course_dir in str(p):
            return p

    # Fallback 2: búsqueda por año aproximado (grado puede variar el nombre)
    all_by_name = list(GUIDES_BASE.glob(f"**/{source}")) + \
                  list(GUIDES_BASE.glob(f"**/{source_colon}"))
    if all_by_name:
        return all_by_name[0]   # devuelve el primero disponible

    return None


def upload_pdf(pdf_path: Path, course: str, degree: str, source: str) -> bool:
    """Sube un PDF al backend via /upload con los metadatos correctos."""
    with open(pdf_path, "rb") as f:
        resp = requests.post(
            f"{BACKEND_URL}/upload",
            files={"files": (source, f, "application/pdf")},
            data={"course": course, "degree": degree, "category": "Juanma"},
            timeout=120,
        )
    if resp.status_code == 200:
        msg = resp.json().get("status_message", "")
        return True, msg
    return False, resp.text


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Comprobar backend
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        info = r.json()
        print(f"Backend activo — LLM: {info['llm']} | Embeddings: {info['embeddings']} "
              f"| Chunks previos: {info['chroma_chunks']}")
    except Exception as e:
        print(f"ERROR: No se puede conectar al backend ({e})")
        print("Arranca primero el backend:")
        print("  uvicorn rag_backend_lite:app --port 8001 --host 127.0.0.1")
        sys.exit(1)

    with open(DATASET_PATH, encoding="utf-8") as f:
        items = json.load(f)

    # Recoger combinaciones únicas (course, degree, source)
    unique: dict[tuple, dict] = {}
    for it in items:
        if it["source_document"] == "N/A":
            continue
        meta = parse_chunk_id(it["chunk_id"], it["source_document"])
        key = (meta["course"], meta["degree"], meta["source"])
        unique[key] = meta

    print(f"\nDocumentos únicos a cargar: {len(unique)}")

    # Cargar los que ya están en Chroma
    existing_resp = requests.get(f"{BACKEND_URL}/files", timeout=10).json()
    loaded_sources: set[str] = set()
    for courses in existing_resp.get("hierarchy", {}).values():
        for degrees in courses.values():
            loaded_sources.update(degrees)

    uploaded = skipped = not_found = 0

    for (course, degree, source), meta in unique.items():
        label = f"{source}  [{course} / {degree[:40]}]"

        pdf_path = find_pdf(course, degree, source)
        if pdf_path is None:
            print(f"  ✗ NO ENCONTRADO: {label}")
            not_found += 1
            continue

        ok, msg = upload_pdf(pdf_path, course, degree, source)
        if ok and "Sin ficheros nuevos" in msg:
            print(f"  ↺ Ya cargado:    {label}")
            skipped += 1
        elif ok:
            print(f"  ✓ Subido:        {label}")
            uploaded += 1
        else:
            print(f"  ✗ Error:         {label}  →  {msg[:80]}")

        time.sleep(0.2)   # pausa leve para no saturar el backend

    # Resumen
    r2 = requests.get(f"{BACKEND_URL}/health", timeout=5).json()
    print(f"\n{'═'*60}")
    print(f"  Subidos:         {uploaded}")
    print(f"  Ya existían:     {skipped}")
    print(f"  No encontrados:  {not_found}")
    print(f"  Chunks totales en ChromaDB: {r2['chroma_chunks']}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
