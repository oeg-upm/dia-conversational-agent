"""
experiment_pipeline.py — Pipeline de 9 experimentos RAG DIA
=============================================================
Ejecuta las 9 condiciones experimentales (modelo × prompt) de forma
automática, gestionando el ciclo arranque/evaluación/parada del backend.

Condiciones:
  C1  qwen2.5:32b  P0  ← YA HECHA, se copia de results.json
  C2  qwen2.5:32b  P1
  C3  qwen2.5:32b  P2
  C4  gemma3:27b   P0
  C5  gemma3:27b   P1
  C6  gemma3:27b   P2
  C7  llama3.1:8b  P0
  C8  llama3.1:8b  P1
  C9  llama3.1:8b  P2

Uso:
  python experiment_pipeline.py
  python experiment_pipeline.py --start C4   # reanudar desde C4
  python experiment_pipeline.py --only C7    # solo una condición

Requisitos previos en el clúster:
  ollama pull gemma3:27b
  ollama pull llama3.1:8b
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_PORT   = 8001
BACKEND_URL    = f"http://127.0.0.1:{BACKEND_PORT}"   # avoid localhost→IPv6 on macOS
CLUSTER_URL    = os.getenv("CLUSTER_URL", "http://100.78.104.3:11434")
OLLAMA_URL     = os.getenv("OLLAMA_URL",  "http://100.78.104.3:11434")
EMBED_MODEL    = "qwen3-embedding:8b"
BACKEND_SCRIPT = Path(__file__).parent / "rag_backend_lite.py"
EVALUATOR      = Path(__file__).parent / "safety_evaluator.py"
RESULTS_DIR    = Path(__file__).parent / "experiment_results"
C1_SOURCE      = Path(__file__).parent / "results.json"

# ── Definición de condiciones ─────────────────────────────────────────────────
CONDITIONS = [
    {"id": "C1", "model": "qwen2.5:32b", "prompt": "P0", "skip": True},
    {"id": "C2", "model": "qwen2.5:32b", "prompt": "P1"},
    {"id": "C3", "model": "qwen2.5:32b", "prompt": "P2"},
    {"id": "C4", "model": "gemma3:27b",  "prompt": "P0"},
    {"id": "C5", "model": "gemma3:27b",  "prompt": "P1"},
    {"id": "C6", "model": "gemma3:27b",  "prompt": "P2"},
    {"id": "C7", "model": "llama3.1:8b", "prompt": "P0"},
    {"id": "C8", "model": "llama3.1:8b", "prompt": "P1"},
    {"id": "C9", "model": "llama3.1:8b", "prompt": "P2"},
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def wait_for_backend(proc: subprocess.Popen, log_path: Path, timeout: int = 300) -> bool:
    """Espera a que el backend responda en /health, mostrando el log del backend."""
    log(f"  Esperando backend en {BACKEND_URL}  (log → {log_path.name})")
    start      = time.time()
    last_err   = ""
    next_tick  = start + 15          # imprime heartbeat cada 15 s
    log_offset = 0                   # bytes ya leídos del log

    while time.time() - start < timeout:
        # ── Mostrar líneas nuevas del log del backend ─────────────────────────
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

        # ── Detectar si el proceso murió prematuramente ───────────────────────
        if proc.poll() is not None:
            # leer el resto del log antes de reportar
            try:
                with open(log_path, "r", errors="replace") as lf:
                    lf.seek(log_offset)
                    for line in lf.read().splitlines():
                        if line.strip():
                            print(f"  [backend] {line}", flush=True)
            except FileNotFoundError:
                pass
            log(f"  ERROR: El proceso backend terminó inesperadamente "
                f"(código {proc.returncode}).")
            return False

        # ── Health check ──────────────────────────────────────────────────────
        try:
            r = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                log(f"  Backend listo — LLM: {data['llm']} | "
                    f"Prompt: {data.get('prompt_version','?')}")
                return True
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            if msg != last_err:
                log(f"  [health] {msg}")
                last_err = msg

        # ── Heartbeat ─────────────────────────────────────────────────────────
        now = time.time()
        if now >= next_tick:
            elapsed = int(now - start)
            log(f"  … esperando backend ({elapsed}s / {timeout}s)")
            next_tick = now + 15

        time.sleep(3)

    log("  ERROR: Backend no respondió en tiempo.")
    return False


def start_backend(model: str, prompt_version: str, cid: str) -> tuple:
    """Arranca el backend con el modelo y prompt indicados.

    Returns (proc, log_path).
    """
    # También poner en el entorno del proceso padre para que el fork lo herede
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    env = os.environ.copy()
    env.update({
        "OLLAMA_URL":                          OLLAMA_URL,
        "LLM_MODEL":                           model,
        "EMBED_MODEL":                         EMBED_MODEL,
        "PROMPT_VERSION":                      prompt_version,
        "LANGCHAIN_TRACING_V2":                "false",
        "LANGCHAIN_TRACING":                   "false",
        "LANGSMITH_TRACING":                   "false",
        "ANONYMIZED_TELEMETRY":                "false",
        "PYTHONUNBUFFERED":                    "1",
        "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES",
    })

    log_path = RESULTS_DIR / f"backend_{cid}.log"
    log_path.parent.mkdir(exist_ok=True)
    log_file = open(log_path, "w")   # fd heredado por el hijo; padre lo cierra tras fork

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "rag_backend_lite:app",
         "--port", str(BACKEND_PORT), "--host", "127.0.0.1"],
        cwd=Path(__file__).parent,
        env=env,
        stdout=log_file,
        stderr=log_file,
    )
    log_file.close()   # el padre cierra su copia; el hijo sigue con la suya

    log(f"  Backend arrancado (PID {proc.pid}) — {model} / {prompt_version}")
    return proc, log_path


def stop_backend(proc: subprocess.Popen):
    """Para el backend limpiamente."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log(f"  Backend parado (PID {proc.pid})")


def run_evaluator(output_path: Path) -> bool:
    """Ejecuta el safety_evaluator y guarda resultados en output_path."""
    env = os.environ.copy()
    env.update({
        "BACKEND_URL":          BACKEND_URL,
        "CLUSTER_URL":          CLUSTER_URL,
        "LANGCHAIN_TRACING_V2": "false",
    })
    result = subprocess.run(
        [sys.executable, str(EVALUATOR),
         "--backend", BACKEND_URL,
         "--cluster", CLUSTER_URL,
         "--output",  str(output_path)],
        cwd=Path(__file__).parent,
        env=env,
    )
    return result.returncode == 0


def add_condition_metadata(results_path: Path, condition: dict):
    """Añade metadatos de la condición al JSON de resultados."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    data["condition"] = {
        "id":             condition["id"],
        "model":          condition["model"],
        "prompt_version": condition["prompt"],
    }
    data["metadata"]["condition_id"]     = condition["id"]
    data["metadata"]["llm_model"]        = condition["model"]
    data["metadata"]["prompt_version"]   = condition["prompt"]
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_summary(results_dir: Path):
    """Imprime tabla resumen de todas las condiciones completadas."""
    print("\n" + "=" * 70)
    print("  RESUMEN FINAL DE EXPERIMENTOS")
    print("=" * 70)
    print(f"  {'Cond':<5} {'Modelo':<18} {'Prompt':<6} {'PASS':<6} {'TOTAL':<6} {'%'}")
    print("  " + "─" * 60)

    for cond in CONDITIONS:
        path = results_dir / f"safety_results_{cond['id']}.json"
        if not path.exists():
            print(f"  {cond['id']:<5} {cond['model']:<18} {cond['prompt']:<6} {'—':<6} {'—':<6} {'pendiente'}")
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        meta = d.get("metadata", {})
        total = meta.get("total", "?")
        passed = meta.get("pass", "?")
        rate  = meta.get("pass_rate", "?")
        print(f"  {cond['id']:<5} {cond['model']:<18} {cond['prompt']:<6} {passed:<6} {total:<6} {rate}%")

    print("=" * 70)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline(start_from: str = "C1", only: str = None):
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("  PIPELINE DE EXPERIMENTOS — RAG DIA SAFETY EVAL")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Filtrar condiciones según argumentos
    conditions = CONDITIONS
    if only:
        conditions = [c for c in CONDITIONS if c["id"] == only]
        if not conditions:
            print(f"ERROR: Condición '{only}' no existe.")
            sys.exit(1)
    elif start_from != "C1":
        ids = [c["id"] for c in CONDITIONS]
        if start_from not in ids:
            print(f"ERROR: Condición '{start_from}' no existe.")
            sys.exit(1)
        idx = ids.index(start_from)
        conditions = CONDITIONS[idx:]

    for condition in conditions:
        cid    = condition["id"]
        model  = condition["model"]
        prompt = condition["prompt"]
        output = RESULTS_DIR / f"safety_results_{cid}.json"

        print(f"\n{'─' * 70}")
        print(f"  ▶ {cid} | modelo: {model} | prompt: {prompt}")
        print(f"{'─' * 70}")

        # ── C1: ya hecha, solo copiar ─────────────────────────────────────────
        if condition.get("skip"):
            if not C1_SOURCE.exists():
                log(f"ERROR: No se encuentra {C1_SOURCE}. Corre primero la evaluación C1.")
                sys.exit(1)
            shutil.copy2(C1_SOURCE, output)
            add_condition_metadata(output, condition)
            log(f"C1 copiada → {output.name}")
            continue

        # ── Comprobar si ya está hecha ────────────────────────────────────────
        if output.exists():
            log(f"{cid} ya tiene resultados ({output.name}), omitiendo.")
            continue

        # ── Arrancar backend ──────────────────────────────────────────────────
        backend_proc, backend_log = start_backend(model, prompt, cid)

        if not wait_for_backend(backend_proc, backend_log, timeout=300):
            stop_backend(backend_proc)
            log(f"ERROR: Backend no arrancó para {cid}. Abortando.")
            sys.exit(1)

        # ── Evaluar ───────────────────────────────────────────────────────────
        log(f"Lanzando evaluador para {cid}...")
        success = run_evaluator(output)

        # ── Parar backend ─────────────────────────────────────────────────────
        stop_backend(backend_proc)

        if not success or not output.exists():
            log(f"ERROR: Evaluación {cid} falló o no generó resultados.")
            continue

        # ── Añadir metadatos de condición ─────────────────────────────────────
        add_condition_metadata(output, condition)
        log(f"✓ {cid} completada → {output.name}")

    print_summary(RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de 9 experimentos RAG DIA.")
    parser.add_argument("--start", default="C1", metavar="Cn",
                        help="Condición desde la que empezar (default: C1)")
    parser.add_argument("--only",  default=None,  metavar="Cn",
                        help="Ejecutar solo una condición (ej: --only C4)")
    args = parser.parse_args()

    run_pipeline(start_from=args.start, only=args.only)
