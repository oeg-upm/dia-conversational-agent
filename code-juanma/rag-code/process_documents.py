import os
import requests
import time

# Configuration
BACKEND_URL = "http://localhost:8001/upload"
BASE_DIR = r"C:\Users\juanm\Desktop\Juanma\aa UPM\aaa TFM\DocumentosRAG\Guías aprendizaje"

def upload_all_guides():
    if not os.path.exists(BASE_DIR):
        print(f"Error: The path {BASE_DIR} does not exist.")
        return

    for course_name in os.listdir(BASE_DIR):
        course_path = os.path.join(BASE_DIR, course_name)
        if not os.path.isdir(course_path):
            continue

        print(f"\n--- PROCESSING COURSE: {course_name} ---")

        # List all items in the course directory (could be category folders or degree folders)
        content_in_course = os.listdir(course_path)

        for item in content_in_course:
            item_path = os.path.join(course_path, item)
            if not os.path.isdir(item_path):
                continue

            # CASE A: it's a intermediate-level folder (Bachelor's / MMaster's as in 2020)
            if item.lower() in ["grado", "máster", "master"]:
                category_name = item
                for degree_name in os.listdir(item_path):
                    degree_path = os.path.join(item_path, degree_name)
                    if os.path.isdir(degree_path):
                        process_degree_folder(course_name, category_name, degree_name, degree_path)

            # CASE B: it's directly the Bachelor's / Master's folder (as in 2021+)
            else:
                degree_name = item 
                # Detect the category based on folder name
                if "master" in degree_name.lower() or "máster" in degree_name.lower():
                    category_name = "Máster"
                else:
                    category_name = "Grado"
                
                process_degree_folder(course_name, category_name, degree_name, item_path)

def process_degree_folder(course, category, degree, path):
    """Auxiliary function to upload files from a specific folder."""
    pdf_files = [f for f in os.listdir(path) if f.lower().endswith('.pdf')]
    total_files = len(pdf_files)
    
    if total_files == 0:
        return

    print(f"\n- Sector: {course} > {category} > {degree}")
    print(f"* Documents: {total_files}")

    for index, pdf in enumerate(pdf_files, start=1):
        file_path = os.path.join(path, pdf)
        payload = {
            "course": course,
            "category": category,
            "degree": degree
        }
        
        print(f"  [{index}/{total_files}] Uploading: {pdf[:40]}...", end="\r")
        
        start_time = time.time()
        try:
            with open(file_path, "rb") as f:
                files = [("files", (pdf, f, "application/pdf"))]
                response = requests.post(BACKEND_URL, data=payload, files=files, timeout=600)
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                res_data = response.json()
                if "already exist" in res_data.get("status_message", "").lower():
                    print(f"  [{index}/{total_files}] -- Skipped (already exists): {pdf[:40]}...")
                else:
                    print(f"  [{index}/{total_files}] -- OK ({elapsed:.1f}s): {pdf[:40]}...")
            else:
                print(f"  [{index}/{total_files}] -- Error {response.status_code}")
                                
        except Exception as e:
            print(f"  [{index}/{total_files}] -- Error: {str(e)}")

if __name__ == "__main__":
    print("====================================================")
    print("     Starting the upload process for all guides...  ")
    print("====================================================")
    t_start = time.time()
    upload_all_guides()
    print(f"\n\nProcess completed in {(time.time() - t_start)/60:.2f} minutes!")