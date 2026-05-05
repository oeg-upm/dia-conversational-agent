# backend/rag.py

import os
from typing import List
import chromadb

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_community.vectorstores.utils import filter_complex_metadata


class BasicRAG:

    def __init__(self):

        self.llm = ChatOpenAI(
            model="qwen2.5:32b",
            base_url="http://100.97.20.71:5000/v1",
            api_key="not_required",
            temperature=0.1
        )

        self.embeddings = OllamaEmbeddings(
        model="qwen3-embedding:8b",
        base_url="http://100.97.20.71:5000"
        )
        

        chroma_client = chromadb.HttpClient(
            host="localhost",
            port=8000
        )

        self.vectorstore = Chroma(
            client=chroma_client,
            collection_name="basic_rag",
            embedding_function=self.embeddings,
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=200
        )

    def add_documents_from_files(self, file_paths: List[str]):
        new_docs = []

        for file_path in file_paths:
            filename = os.path.basename(file_path)

            if not filename.endswith(".pdf"):
                continue

            try:
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.ocr_options = EasyOcrOptions()

                doc_converter = DocumentConverter(
                    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
                )

                custom_chunker = HybridChunker(
                    tokenizer="Qwen/Qwen2-0.5B",
                    max_tokens=400,
                    overlap_tokens=50,
                    merge_peers=True
                )

                loader = DoclingLoader(
                    file_path=file_path,
                    export_type=ExportType.DOC_CHUNKS,
                    chunker=custom_chunker,
                    converter=doc_converter
                )

                splits = loader.load()

                for i, doc in enumerate(splits):
                    doc.metadata = {
                        "source": filename,
                        "chunk_index": i
                    }

                new_docs.extend(splits)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        if new_docs:
            cleaned_docs = filter_complex_metadata(new_docs)
            doc_ids = [
                f"{doc.metadata['source']}_ch_{doc.metadata['chunk_index']}"
                for doc in cleaned_docs
            ]
            self.vectorstore.add_documents(documents=cleaned_docs, ids=doc_ids)
            return f"Added {len(cleaned_docs)} chunks."

        return "No valid documents added."
    
    
    
    def query(self, question: str, selected_files: List[str]):

        #lo blindeamos por seguridad
        selected_files = [os.path.basename(f) for f in selected_files]
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"source": {"$in": selected_files}}
            }
        )

        docs = retriever.invoke(question)

        if not docs:
            return "No relevant context found."

        template = """
        Eres un asistente académico universitario. Responde usando el contexto proporcionado.
        Si la información está presente aunque sea parcialmente, extráela y respóndela.
        Solo di "No lo sé" si la información es completamente inexistente en el contexto.

        REGLAS:
        - Responde en español
        - Responde de forma directa y concisa
        - Incluye siempre los datos específicos: nombres, porcentajes, fechas, créditos
        - No uses listas ni bullets

        Contexto: {context}
        Pregunta: {question}
        Respuesta:"""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()

        context_text = "\n\n".join(d.page_content for d in docs)

        return chain.invoke({
            "context": context_text,
            "question": question
        })
        
    
    def list_documents(self):

        # Obtener colección directamente de chromaDB para acceder a los metadatos sin necesidad de recuperar embeddings
        collection = self.vectorstore._collection

        # Traer solo metadatos (sin embeddings)
        results = collection.get(include=["metadatas"])

        if not results or "metadatas" not in results:
            return []

        sources = [
            metadata.get("source")
            for metadata in results["metadatas"]
            if metadata and "source" in metadata
        ]

        # Quitar duplicados y ordenar
        unique_sources = sorted(list(set(sources)))

        return unique_sources