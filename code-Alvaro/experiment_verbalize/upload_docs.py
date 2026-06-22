import requests
import os

API_URL = "http://localhost:9000/upload"

files = [
    "/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Sistemas de Planificación.pdf",
    "/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Probabilidades y Estadística II.pdf",
    "/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Investigación Operativa.pdf",
    "/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Lógica.pdf",
    "/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Minería de Datos.pdf",
    "/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Programación Declarativa, Lógica y Restricciones.pdf",
    "/home/alvaro/Escritorio/Guías aprendizaje/Curso 2020:2021/Grado/Grado en Ingeneiría Informática/Reconocimiento de Formas.pdf",
]

# descomentar para utilizar los documentos verbalizados
#files = [
  #  "/home/alvaro/Escritorio/dia-conversational-agent/code-Alvaro/dataset_verb/docs_verb/Sistemas de Planificación.pdf",
  #  "/home/alvaro/Escritorio/dia-conversational-agent/code-Alvaro/dataset_verb/docs_verb/Probabilidades y Estadística II.pdf",
  #  "/home/alvaro/Escritorio/dia-conversational-agent/code-Alvaro/dataset_verb/docs_verb/Investigación Operativa.pdf",
  #  "/home/alvaro/Escritorio/dia-conversational-agent/code-Alvaro/dataset_verb/docs_verb/Lógica.pdf",
  #  "/home/alvaro/Escritorio/dia-conversational-agent/code-Alvaro/dataset_verb/docs_verb/Minería de Datos.pdf",
  #  "/home/alvaro/Escritorio/dia-conversational-agent/code-Alvaro/dataset_verb/docs_verb/Programación Declarativa, Lógica y Restricciones.pdf",
 #   "/home/alvaro/Escritorio/dia-conversational-agent/code-Alvaro/dataset_verb/docs_verb/Reconocimiento de Formas.pdf",
#]


def upload_pdfs(file_paths):
    multipart_files = []

    for path in file_paths:
        if not os.path.exists(path):
            print(f"❌ No existe: {path}")
            continue

        filename = os.path.basename(path)
        multipart_files.append(
            ("files", (filename, open(path, "rb"), "application/pdf"))
        )

    if not multipart_files:
        print("❌ No hay archivos válidos")
        return

    try:
        response = requests.post(API_URL, files=multipart_files)

        print(f"\nStatus: {response.status_code}")
        print("Respuesta:", response.text)

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        # cerrar archivos abiertos
        for _, file_tuple in multipart_files:
            file_tuple[1].close()