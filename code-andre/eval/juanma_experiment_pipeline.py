"""
juanma_experiment_pipeline.py
==============================
Pipeline completo: 3 configs de embeddings × 2 datasets = 6 experimentos.

Configs:
  bge    → bge-m3:latest                             + qwen2.5:32b
  qwen4b → qwen3-embedding:4b                        + qwen2.5:32b
  octen  → nicolasfer45/Octen-Embedding-4B-GGUF:latest + qwen2.5:32b

Datasets:
  original   → rag_dataset_v3_octen_qwen2.5_V2.json (Juanma)
  humanized  → rag_dataset_humanized_v1.json

Flujo por cada config:
  1. Arranca backend con EMBED_MODEL y CHROMA_DIR correctos
  2. Carga el corpus de Juanma (si no está ya cargado)
  3. Genera respuestas para dataset original  → original_answers_{config}.json
  4. Genera respuestas para dataset humanized → humanized_answers_{config}.json
  5. Para el backend
  6. Evalúa con RAGAS ambos ficheros         → ragas_original_{config}.csv
                                             → ragas_humanized_{config}.csv

Al final imprime una tabla comparativa de todas las métricas.

Uso:
  python juanma_experiment_pipeline.py                   # todo
  python juanma_experiment_pipeline.py --start qwen4b   # reanudar desde qwen4b
  python juanma_experiment_pipeline.py --only octen     # solo una config
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Rutas ─────────────────────────────────────────────────────────────────────
EVAL_DIR     = Path(__file__).parent
RESULTS_DIR  = EVAL_DIR / "experiment_results"
BACKEND_URL  = "http://127.0.0.1:8001"
BACKEND_PORT = 8001

# ── Definición de configuraciones ─────────────────────────────────────────────
CONFIGS = [
    {
        "id":         "bge",
        "embed_model": "bge-m3:latest",
        "chroma_dir":  "./chroma_juanma_bge",
        "multiquery":  False,
    },
    {
        "id":         "qwen4b",
        "embed_model": "qwen3-embedding:4b",
        "chroma_dir":  "./chroma_juanma_qwen4b",
        "multiquery":  False,
    },
    {
        "id":         "octen",
        "embed_model": "nicolasfer45/Octen-Embedding-4B-GGUF:latest",
        "chroma_dir":  "./chroma_juanma_octen",
        "multiquery":  True,
    },
]

DATASETS = ["original", "humanized"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def start_backend(cfg: dict) -> tuple:
    """Arranca el backend con la config indicada. Devuelve (proc, log_path)."""
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    env = os.environ.copy()

    ollama_url = os.getenv("OLLAMA_URL", "http://100.78.104.3:11434")

    env.update({
        "OLLAMA_URL":                          ollama_url,
        "EMBED_URL":                           ollama_url,
        "LLM_MODEL":                           "qwen2.5:32b",
        "EMBED_MODEL":                         cfg["embed_model"],
        "CHROMA_DIR":                          cfg["chroma_dir"],
        "N_QUERIES":                           "3" if cfg["multiquery"] else "1",
        "PROMPT_VERSION":                      "P0",
        "LANGCHAIN_TRACING_V2":                "false",
        "LANGCHAIN_TRACING":                   "false",
        "LANGSMITH_TRACING":                   "false",
        "ANONYMIZED_TELEMETRY":                "false",
        "PYTHONUNBUFFERED":                    "1",
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
    })

    log_path = RESULTS_DIR / f"backend_{cfg['id']}.log"
    log_path.parent.mkdir(exist_ok=True)
    log_file = open(log_path, "w")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "rag_backend_lite:app",
         "--port", str(BACKEND_PORT), "--host", "127.0.0.1"],
        cwd=EVAL_DIR,
        env=env,
        stdout=log_file,
        stderr=log_file,
    )
    log_file.close()
    log(f"  Backend arrancado (PID {proc.pid}) — {cfg['embed_model']}")
    return proc, log_path


def wait_for_backend(proc, log_path: Path, timeout: int = 300) -> bool:
    log(f"  Esperando backend en {BACKEND_URL}...")
    start     = time.time()
    last_err  = ""
    log_offset = 0
    next_tick  = start + 20

    while time.time() - start < timeout:
        # Streamer de log del backend
        try:
            with open(log_path, "r", errors="replace") as lf:
                lf.seek(log_offset)
                new_text = lf.read()
                log_offset += len(new_text.encode("utf-8", errors="replace"))
            for line in new_text.splitlines():
                if line.strip():
                    print(f"  [backend] {line}", flush=True)
        except FileNotFoundError:
            pass

        if proc.poll() is not None:
            log(f"  ERROR: backend terminó inesperadamente (código {proc.returncode})")
            return False

        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                log(f"  Backend listo — embeddings: {data['embeddings']} "
                    f"| chunks: {data['chroma_chunks']}")
                return True
        except Exception as exc:
            msg = str(exc)[:80]
            if msg != last_err:
                log(f"  [health] {msg}")
                last_err = msg

        if time.time() >= next_tick:
            log(f"  … esperando ({int(time.time()-start)}s / {timeout}s)")
            next_tick = time.time() + 20

        time.sleep(3)

    log("  ERROR: backend no respondió en tiempo.")
    return False


def stop_backend(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log(f"  Backend parado (PID {proc.pid})")


def corpus_loaded(min_chunks: int = 100) -> bool:
    """Comprueba si el corpus ya está cargado comprobando el chunk count."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return r.json().get("chroma_chunks", 0) >= min_chunks
    except Exception:
        return False


def run_load_corpus() -> bool:
    log("  Cargando corpus de Juanma...")
    result = subprocess.run(
        [sys.executable, str(EVAL_DIR / "load_juanma_corpus.py")],
        cwd=EVAL_DIR,
    )
    return result.returncode == 0


def run_generate(config_id: str, dataset: str) -> Path | None:
    output = RESULTS_DIR / f"{dataset}_answers_{config_id}.json"
    if output.exists():
        log(f"  Ya existe {output.name} — omitiendo generación.")
        return output

    log(f"  Generando respuestas: dataset={dataset} config={config_id}...")
    result = subprocess.run(
        [sys.executable, str(EVAL_DIR / "generate_answers.py"),
         "--config",  config_id,
         "--dataset", dataset,
         "--output",  str(output)],
        cwd=EVAL_DIR,
    )
    if result.returncode != 0 or not output.exists():
        log(f"  ERROR: generate_answers falló para {dataset}/{config_id}")
        return None
    log(f"  ✓ {output.name} generado")
    return output


def run_evaluate(input_json: Path, config_id: str, dataset: str) -> Path | None:
    output = RESULTS_DIR / f"ragas_{dataset}_{config_id}.csv"
    if output.exists():
        log(f"  Ya existe {output.name} — omitiendo evaluación.")
        return output

    log(f"  Evaluando con RAGAS: {input_json.name}...")
    result = subprocess.run(
        [sys.executable, str(EVAL_DIR / "evaluate_humanized.py"),
         "--input",  str(input_json),
         "--output", str(output)],
        cwd=EVAL_DIR,
    )
    if result.returncode != 0 or not output.exists():
        log(f"  ERROR: evaluate_humanized falló para {dataset}/{config_id}")
        return None
    log(f"  ✓ {output.name} guardado")
    return output


# ── Resumen final ─────────────────────────────────────────────────────────────

def print_summary():
    try:
        import pandas as pd
    except ImportError:
        print("\n(instala pandas para ver el resumen: pip install pandas)")
        return

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision",
                   "context_recall", "answer_similarity", "answer_correctness"]

    rows = []
    for cfg in CONFIGS:
        for ds in DATASETS:
            csv_path = RESULTS_DIR / f"ragas_{ds}_{cfg['id']}.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            means = {col: df[col].mean() for col in metric_cols if col in df.columns}
            means["config"]  = cfg["id"]
            means["dataset"] = ds
            rows.append(means)

    if not rows:
        print("\n  Sin resultados todavía.")
        return

    summary = pd.DataFrame(rows).set_index(["config", "dataset"])
    print("\n" + "=" * 80)
    print("  RESULTADOS FINALES — ORIGINAL vs HUMANIZADO")
    print("=" * 80)
    print(summary.round(3).to_string())
    print("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(start_from: str = None, only: str = None):
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("  PIPELINE JUANMA — ORIGINAL vs HUMANIZADO  (3 configs × 2 datasets)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    configs = CONFIGS
    if only:
        configs = [c for c in CONFIGS if c["id"] == only]
        if not configs:
            print(f"ERROR: config '{only}' no existe.")
            sys.exit(1)
    elif start_from:
        ids = [c["id"] for c in CONFIGS]
        if start_from not in ids:
            print(f"ERROR: config '{start_from}' no existe.")
            sys.exit(1)
        configs = CONFIGS[ids.index(start_from):]

    answer_files: dict[tuple, Path] = {}   # (config_id, dataset) → path

    # ── Fase 1: generar respuestas ─────────────────────────────────────────────
    for cfg in configs:
        print(f"\n{'─'*70}")
        print(f"  CONFIG: {cfg['id']}  |  embed: {cfg['embed_model']}")
        print(f"{'─'*70}")

        # Arrancar backend
        backend_proc, backend_log = start_backend(cfg)
        if not wait_for_backend(backend_proc, backend_log):
            stop_backend(backend_proc)
            log(f"ERROR: Backend no arrancó para {cfg['id']}. Saltando config.")
            continue

        # Cargar corpus si hace falta
        if not corpus_loaded():
            if not run_load_corpus():
                stop_backend(backend_proc)
                log(f"ERROR: No se pudo cargar el corpus para {cfg['id']}.")
                continue

        # Generar respuestas para ambos datasets
        for ds in DATASETS:
            out = run_generate(cfg["id"], ds)
            if out:
                answer_files[(cfg["id"], ds)] = out

        stop_backend(backend_proc)
        time.sleep(2)   # pausa breve entre configs para liberar recursos

    # ── Fase 2: evaluar con RAGAS ─────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  FASE 2 — Evaluación RAGAS (no requiere backend)")
    print(f"{'─'*70}")

    for cfg in configs:
        for ds in DATASETS:
            key = (cfg["id"], ds)
            # Buscar el fichero (puede existir de una ejecución anterior)
            input_json = answer_files.get(key) or \
                         RESULTS_DIR / f"{ds}_answers_{cfg['id']}.json"
            if not input_json.exists():
                log(f"  Sin respuestas para {cfg['id']}/{ds} — omitiendo evaluación.")
                continue
            run_evaluate(input_json, cfg["id"], ds)

    # ── Resumen ────────────────────────────────────────────────────────────────
    print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline completo: 3 configs × 2 datasets")
    parser.add_argument("--start", default=None, metavar="CONFIG_ID",
                        help="Reanudar desde esta config: bge | qwen4b | octen")
    parser.add_argument("--only",  default=None, metavar="CONFIG_ID",
                        help="Ejecutar solo una config")
    args = parser.parse_args()

    run_pipeline(start_from=args.start, only=args.only)
