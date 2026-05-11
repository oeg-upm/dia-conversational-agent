"""
load_guides.py — Carga masiva de guías de aprendizaje al backend lite
======================================================================
Recorre la carpeta de guías y sube cada PDF al endpoint /upload del backend.

Estructura esperada:
  <BASE_DIR>/
    <Nombre de titulación>/
      Asignatura1.pdf
      Asignatura2.pdf
      ...

Uso:
  python load_guides.py
  python load_guides.py --dir "/ruta/alternativa" --url http://localhost:8001
"""

import os
import time
import argparse
import requests

# ── Configuración ─────────────────────────────────────────────────────────────
DEFAULT_DIR = "/Users/andrep/Downloads/Guías aprendizaje/Curso 2025:2026"
DEFAULT_URL = "http://localhost:8001"
COURSE_NAME = "Curso 2025/2026"   # nombre limpio (sin ":" que da problemas en IDs)

MASTER_KEYWORDS = ["máster", "master", "msc", "m.sc"]


def detect_category(degree_name: str) -> str:
    """Devuelve 'Máster' o 'Grado' según el nombre de la carpeta."""
    lower = degree_name.lower()
    if any(kw in lower for kw in MASTER_KEYWORDS):
        return "Máster"
    return "Grado"


def upload_pdf(backend_url: str, pdf_path: str, course: str, category: str, degree: str) -> dict:
    """Sube un único PDF al backend y devuelve la respuesta JSON."""
    filename = os.path.basename(pdf_path)
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{backend_url}/upload",
            data={"course": course, "category": category, "degree": degree},
            files=[("files", (filename, f, "application/pdf"))],
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def load_all(base_dir: str, backend_url: str):
    if not os.path.isdir(base_dir):
        print(f"ERROR: No existe la carpeta '{base_dir}'")
        return

    # Verificar que el backend está vivo
    try:
        health = requests.get(f"{backend_url}/health", timeout=5).json()
        print(f"Backend OK — LLM: {health['llm']} | "
              f"Embeddings: {health['embeddings']} | "
              f"Chunks en DB: {health['chroma_chunks']}")
    except Exception as e:
        print(f"ERROR: No se puede conectar al backend ({backend_url}): {e}")
        print("Asegúrate de que el backend está arrancado:")
        print("  uvicorn rag_backend_lite:app --port 8001")
        return

    print(f"\nCarpeta base: {base_dir}")
    print(f"Curso:        {COURSE_NAME}")
    print("=" * 60)

    # Recopilar todo antes de empezar para mostrar totales
    plan = []
    for degree_name in sorted(os.listdir(base_dir)):
        degree_path = os.path.join(base_dir, degree_name)
        if not os.path.isdir(degree_path):
            continue
        pdfs = sorted(
            f for f in os.listdir(degree_path)
            if f.lower().endswith(".pdf")
        )
        if pdfs:
            category = detect_category(degree_name)
            plan.append((degree_name, degree_path, category, pdfs))

    total_pdfs = sum(len(p[3]) for p in plan)
    print(f"Titulaciones encontradas: {len(plan)}")
    print(f"PDFs a procesar:          {total_pdfs}\n")

    # Carga
    uploaded = 0
    skipped  = 0
    errors   = 0
    t_start  = time.time()

    for degree_name, degree_path, category, pdfs in plan:
        print(f"\n▶ {category}: {degree_name}  ({len(pdfs)} PDFs)")
        print(f"  {'─' * 54}")

        for idx, pdf_name in enumerate(pdfs, start=1):
            pdf_path = os.path.join(degree_path, pdf_name)
            label    = pdf_name[:48] + ("…" if len(pdf_name) > 48 else "")
            print(f"  [{idx:2}/{len(pdfs)}] {label}", end=" ", flush=True)

            t0 = time.time()
            try:
                result  = upload_pdf(backend_url, pdf_path, COURSE_NAME, category, degree_name)
                elapsed = time.time() - t0
                msg     = result.get("status_message", "")

                if "ya estaban" in msg or "already exist" in msg.lower():
                    print(f"— omitido (ya existe)")
                    skipped += 1
                else:
                    print(f"— OK  ({elapsed:.1f}s)")
                    uploaded += 1

            except Exception as e:
                print(f"— ERROR: {e}")
                errors += 1

    # Resumen final
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  Proceso completado en {elapsed_total / 60:.1f} min")
    print(f"  ✓ Subidos:  {uploaded}")
    print(f"  ↷ Omitidos: {skipped}  (ya estaban en la DB)")
    print(f"  ✗ Errores:  {errors}")
    print("=" * 60)

    if errors == 0:
        # Verificar estado final de la DB
        try:
            health = requests.get(f"{backend_url}/health", timeout=5).json()
            print(f"\nChunks totales en ChromaDB: {health['chroma_chunks']}")
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carga guías de aprendizaje al RAG backend.")
    parser.add_argument(
        "--dir", default=DEFAULT_DIR,
        help="Carpeta raíz con subcarpetas por titulación"
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help="URL base del backend (default: http://localhost:8001)"
    )
    args = parser.parse_args()
    load_all(args.dir, args.url)
