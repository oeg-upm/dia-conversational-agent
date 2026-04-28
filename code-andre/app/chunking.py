from typing import List

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """
    Simple char-based chunking (robusto y rápido para MVP).
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks