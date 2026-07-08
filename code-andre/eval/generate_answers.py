"""
generate_answers.py
===================
Envía cada pregunta de un dataset al backend en ejecución y almacena
la respuesta y los chunks recuperados.
Produce un JSON listo para pasar a evaluate_humanized.py.

Uso:
  # Dataset humanizado (por defecto)
  python generate_answers.py --config bge

  # Dataset original de Juanma
  python generate_answers.py --config bge --dataset original

  # Ruta de salida personalizada
  python generate_answers.py --config bge --output mi_fichero.json

Datasets disponibles:
  humanized  →  rag_dataset_humanized_v1.json         (por defecto)
  original   →  rag_dataset_v3_octen_qwen2.5_V2.json  (Juanma)

Requisito: backend corriendo en http://127.0.0.1:8001
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# ── Configuración ─────────────────────────────────────────────────────────────
BACKEND_URL   = "http://127.0.0.1:8001"
OUTPUT_DIR    = Path(__file__).parent / "experiment_results"

DATASETS = {
    "humanized": Path(__file__).parent.parent / "dataset" / "rag_dataset_humanized_v1.json",
    "original":  Path("/Users/andrep/Documents/GitHub/dia-conversational-agent/code-juanma/dataset/rag_dataset_v3_octen_qwen2.5_V2.json"),
}

# Categorías que usan RAG (necesitan selected_context)
RAG_CATEGORIES = {"factual", "procedural", "comparative"}
# Categorías que NO usan RAG (ambiguous y out_of_scope: LLM-only)
LLM_ONLY_CATEGORIES = {"out_of_scope", "ambiguous"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_chunk_id(chunk_id: str, source_document: str) -> dict:
    base = re.sub(r"_ch_\d+$", "", chunk_id)
    m = re.match(r"^(Curso \d{4}_\d{4})_", base)
    course = m.group(1) if m else "Unknown"
    rest = base[len(course) + 1:]
    dm = re.match(rf"^(.+)_{re.escape(source_document)}$", rest)
    degree = dm.group(1) if dm else rest
    return {"course": course, "degree": degree, "source": source_document}


def reset_session():
    try:
        requests.get(f"{BACKEND_URL}/reset", timeout=5)
    except Exception:
        pass


def query_backend(question: str, selected_context: list, retries: int = 2) -> dict:
    """Llama a POST /chat y devuelve {response, contexts}."""
    payload = {
        "message": question,
        "selected_context": selected_context,
        "chat_history": [],
    }
    for attempt in range(retries + 1):
        try:
            r = requests.post(
                f"{BACKEND_URL}/chat?context=true",
                json=payload,
                timeout=120,
            )
            r.raise_for_status()
            data = r.json()
            return {
                "answer":   data.get("response", ""),
                "contexts": data.get("context", []),
            }
        except Exception as e:
            if attempt < retries:
                time.sleep(3)
            else:
                return {"answer": f"[ERROR: {e}]", "contexts": []}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default="bge",
                        help="Embedding config: bge | qwen4b | octen (se usa en el nombre del fichero)")
    parser.add_argument("--dataset", default="humanized",
                        choices=["humanized", "original"],
                        help="Dataset a usar: humanized (default) | original")
    parser.add_argument("--output",  default=None,
                        help="Ruta de salida JSON (opcional)")
    args = parser.parse_args()

    # Comprobar backend
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        info = r.json()
        print(f"Backend activo — LLM: {info['llm']} | "
              f"Embeddings: {info['embeddings']} | "
              f"Chunks: {info['chroma_chunks']}")
    except Exception as e:
        print(f"ERROR: No se puede conectar al backend: {e}")
        sys.exit(1)

    dataset_path = DATASETS[args.dataset]
    with open(dataset_path, encoding="utf-8") as f:
        items = json.load(f)

    output_path = Path(args.output) if args.output else \
        OUTPUT_DIR / f"{args.dataset}_answers_{args.config}.json"
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"\nDataset:   {args.dataset}  ({dataset_path.name})")
    print(f"Config:    {args.config}")
    print(f"Preguntas: {len(items)}  →  {output_path.name}")

    results = []
    errors = 0

    for item in tqdm(items, desc="Generando respuestas"):
        reset_session()

        q_type = item["question_type"]

        if q_type in RAG_CATEGORIES and item["source_document"] != "N/A":
            meta = parse_chunk_id(item["chunk_id"], item["source_document"])
            selected_context = [{
                "course": meta["course"],
                "degree": meta["degree"],
                "source": meta["source"],
            }]
        else:
            # LLM-only: sin contexto documental
            selected_context = []

        result = query_backend(item["question"], selected_context)

        new_item = dict(item)   # copia todos los campos originales
        new_item["answer"]   = result["answer"]
        new_item["contexts"] = result["contexts"]
        new_item["generation_method"] = f"{args.dataset}_rag_{args.config}"

        if result["answer"].startswith("[ERROR"):
            errors += 1
            tqdm.write(f"  ✗ Error en sample {item['sample_id']}: {result['answer']}")

        results.append(new_item)
        time.sleep(0.1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'═'*60}")
    print(f"  Guardado en:  {output_path}")
    print(f"  Total:        {len(results)}")
    print(f"  Errores:      {errors}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
