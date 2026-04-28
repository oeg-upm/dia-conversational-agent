from docling.document_converter import DocumentConverter

def parse_pdf_to_markdown(pdf_path: str) -> str:
    """
    Parse PDF with Docling and export to Markdown.
    """
    converter = DocumentConverter()
    doc = converter.convert(pdf_path).document
    return doc.export_to_markdown()