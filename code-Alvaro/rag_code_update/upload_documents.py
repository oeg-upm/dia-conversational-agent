"""
bulk_upload.py
--------------
Walks the document directory tree and uploads all PDF files to the RAG backend,
attaching hierarchical metadata (course / category / degree) to each file.

Directory structure supported:

  BASE_DIR/
  ├── 2020/                        ← course (year)
  │   ├── Grado/                   ← category (intermediate folder, Case A)
  │   │   └── Computer Science/    ← degree
  │   │       └── *.pdf
  │   └── Máster/
  │       └── Data Science/
  │           └── *.pdf
  └── 2021/                        ← course (year)
      ├── Computer Science/        ← degree directly (Case B, category inferred)
      │   └── *.pdf
      └── Máster en IA/
          └── *.pdf
"""

import os
import time
import requests

# ---------------------------------------------------------------------------
# Configuration  ← edit these two lines
# ---------------------------------------------------------------------------
BACKEND_URL = "http://localhost:9000/upload"
BASE_DIR = r"/home/alvaro/Escritorio/Guías aprendizaje"  # Path to the root folder containing all course folders
# ---------------------------------------------------------------------------


def upload_all_guides() -> None:
    if not os.path.exists(BASE_DIR):
        print(f"Error: the path '{BASE_DIR}' does not exist.")
        return

    for course_name in sorted(os.listdir(BASE_DIR)):
        course_path = os.path.join(BASE_DIR, course_name)
        if not os.path.isdir(course_path):
            continue

        print(f"\n--- PROCESSING COURSE: {course_name} ---")

        for item in sorted(os.listdir(course_path)):
            item_path = os.path.join(course_path, item)
            if not os.path.isdir(item_path):
                continue

            # Case A: intermediate category folder (Grado / Máster / Master)
            if item.lower() in {"grado", "máster", "master"}:
                category_name = item
                for degree_name in sorted(os.listdir(item_path)):
                    degree_path = os.path.join(item_path, degree_name)
                    if os.path.isdir(degree_path):
                        _process_folder(course_name, category_name, degree_name, degree_path)

            # Case B: degree folder directly inside the course folder
            else:
                degree_name = item
                if "master" in degree_name.lower() or "máster" in degree_name.lower():
                    category_name = "Máster"
                else:
                    category_name = "Grado"
                _process_folder(course_name, category_name, degree_name, item_path)


def _process_folder(course: str, category: str, degree: str, path: str) -> None:
    """Upload every PDF found in *path* to the backend."""
    pdf_files = sorted(f for f in os.listdir(path) if f.lower().endswith(".pdf"))
    total = len(pdf_files)

    if total == 0:
        return

    print(f"\n  Sector: {course} > {category} > {degree}")
    print(f"  Documents: {total}")

    for index, pdf in enumerate(pdf_files, start=1):
        file_path = os.path.join(path, pdf)
        payload = {"course": course, "category": category, "degree": degree}

        # Truncate long names in the progress line
        short_name = pdf[:50] + ("…" if len(pdf) > 50 else "")
        print(f"  [{index:>{len(str(total))}}/{total}] Uploading: {short_name}", end="\r")

        t0 = time.time()
        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    BACKEND_URL,
                    data=payload,
                    files=[("files", (pdf, f, "application/pdf"))],
                    timeout=600,
                )
            elapsed = time.time() - t0

            if response.status_code == 200:
                msg = response.json().get("status_message", "")
                if "already" in msg.lower():
                    print(f"  [{index}/{total}] -- Skipped (already indexed):  {short_name}")
                else:
                    print(f"  [{index}/{total}] -- OK ({elapsed:.1f}s):              {short_name}")
            else:
                print(f"  [{index}/{total}] -- HTTP {response.status_code}: {short_name}")

        except Exception as exc:
            print(f"  [{index}/{total}] -- Error: {exc}")
            
        finally: 
            time.sleep(3)  # Small delay to avoid overwhelming the backend


if __name__ == "__main__":
    print("=" * 60)
    print("   Starting bulk upload of all academic guides…")
    print("=" * 60)

    t_start = time.time()
    upload_all_guides()
    elapsed_min = (time.time() - t_start) / 60
    print(f"\n\nProcess completed in {elapsed_min:.2f} minutes.")