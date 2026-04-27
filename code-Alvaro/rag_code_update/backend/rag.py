import os
import asyncio
from typing import List, Dict, Any
import chromadb

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.utils import filter_complex_metadata
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker


class BasicRAG:

    def __init__(self):

        self.llm = ChatOpenAI(
            model="qwen2.5:32b",
            base_url="http://100.100.52.14:5000/v1",
            api_key="not_required",
            temperature=0.1
        )

        self.embeddings = OllamaEmbeddings(
            model="qwen3-embedding:8b",
            base_url="http://100.100.52.14:5000"
        )

        #Cambiar para ejecutar en local
        # chroma_client = chromadb.HttpClient(
        #     host="localhost",  # en vez de "chromadb"
        #     port=8000
        # )
        
        chroma_client = chromadb.HttpClient(
            host="chromadb",  # Use the service name defined in docker-compose.yml
            port=8000
        )

        self.vectorstore = Chroma(
            client=chroma_client,
            collection_name="rag_dia",
            embedding_function=self.embeddings,
        )

        # In-memory session history (last 10 turns)
        self.session_history: List[Dict[str, str]] = []

        # Last retrieved docs for /inspector
        self.last_retrieved_docs = []

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_documents_from_files(
        self,
        file_paths: List[str],
        course: str = "Unknown",
        category: str = "Unknown",
        degree: str = "Unknown",
        processed_files: set = None,
    ) -> str:
        """
        Ingest PDFs with hierarchical metadata (course / category / degree).
        Skips files whose unique ID is already present in *processed_files*.
        Returns a status message.
        """
        if processed_files is None:
            processed_files = set()

        new_docs = []
        new_filenames = []

        for file_path in file_paths:
            filename = os.path.basename(file_path)

            if not filename.lower().endswith(".pdf"):
                continue

            unique_file_id = f"{course}_{degree}_{filename}"
            if unique_file_id in processed_files:
                continue

            try:
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.ocr_options = EasyOcrOptions()

                doc_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )

                custom_chunker = HybridChunker(
                    tokenizer="Qwen/Qwen2-0.5B",
                    max_tokens=600,
                    overlap_tokens=200,
                    merge_peers=True,
                )

                loader = DoclingLoader(
                    file_path=file_path,
                    export_type=ExportType.DOC_CHUNKS,
                    chunker=custom_chunker,
                    converter=doc_converter,
                )

                splits = loader.load()

                for i, doc in enumerate(splits):
                    doc.metadata["source"] = filename
                    doc.metadata["course"] = course
                    doc.metadata["category"] = category
                    doc.metadata["degree"] = degree
                    doc.metadata["chunk_index"] = i
                    # Context injection for better retrieval interpretability
                    doc.page_content = (
                        f"[{course} - {degree} - {filename}]\n{doc.page_content}"
                    )

                new_docs.extend(splits)
                new_filenames.append(filename)
                processed_files.add(unique_file_id)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

        if new_docs:
            cleaned_docs = filter_complex_metadata(new_docs)
            doc_ids = [
                f"{doc.metadata['course']}_{doc.metadata['degree']}"
                f"_{doc.metadata['source']}_ch_{doc.metadata['chunk_index']}"
                for doc in cleaned_docs
            ]
            self.vectorstore.add_documents(documents=cleaned_docs, ids=doc_ids)
            return f"Added {len(cleaned_docs)} chunks from {len(new_filenames)} file(s)."

        return "No new valid documents added (already indexed or unsupported format)."

    # ------------------------------------------------------------------
    # RAG-Fusion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(results: List[List], k: int = 60):
        """Fuse multiple ranked lists of documents using RRF."""
        fused_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Any] = {}

        for docs in results:
            for rank, doc in enumerate(docs):
                source = doc.metadata.get("source", "unknown")
                chunk = doc.metadata.get("chunk_index", -1)
                doc_id = f"{source}_chunk_{chunk}"

                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                    doc_lookup[doc_id] = doc

                fused_scores[doc_id] += 1.0 / (rank + k)

        reranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_lookup[doc_id], score) for doc_id, score in reranked]

    def _build_filter(self, selected_files: List[str]) -> dict:
        """
        Build a ChromaDB metadata filter from a list of filenames.
        Supports simple filename strings OR dicts with course/degree/source keys.
        """
        # Normalise: accept plain strings (legacy) or dicts (advanced)
        if not selected_files:
            return {}

        # Plain filenames (legacy interface)
        if isinstance(selected_files[0], str):
            sources = [os.path.basename(f) for f in selected_files]
            if len(sources) == 1:
                return {"source": sources[0]}
            return {"source": {"$in": sources}}

        # Rich context objects (advanced interface)
        if len(selected_files) == 1:
            ctx = selected_files[0]
            return {
                "$and": [
                    {"course": ctx["course"]},
                    {"degree": ctx["degree"]},
                    {"source": ctx["source"]},
                ]
            }

        filter_list = []
        for ctx in selected_files:
            filter_list.append(
                {
                    "$and": [
                        {"course": ctx["course"]},
                        {"degree": ctx["degree"]},
                        {"source": ctx["source"]},
                    ]
                }
            )
        return {"$or": filter_list}

    # ------------------------------------------------------------------
    # Query  (RAG-Fusion + Multi-Query + session history)
    # ------------------------------------------------------------------



    async def query(self, question: str, selected_files) -> str:

        # --- Build conversation history string ---
        formatted_history = ""
        for turn in self.session_history:
            formatted_history += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"
        if not formatted_history:
            formatted_history = "Beginning of the conversation."

        # --- 1. Multi-Query generation ---
        mq_template = """
You are an expert Academic Search Assistant. Your goal is to rewrite and expand the
user's current question into 5 distinct, standalone search queries for a vector database.

CRITICAL RULES:
1. CONTEXTUAL RESOLUTION: If the user's question contains pronouns or implicit references,
   use the Chat History to resolve them into full subject names or topics.
2. STANDALONE QUERIES: Each query must be complete and understandable without the chat history.
3. PERSPECTIVES: Cover different aspects: formal name, specific requirements, evaluation
   criteria, and related terminology.
4. LANGUAGE: Output queries in the same language as the user's question.

Chat history:
{chat_history}

User question:
{question}

Output only the 5 standalone alternative queries, one per line, no numbering.
"""
        prompt_mq = PromptTemplate.from_template(mq_template)
        mq_chain = prompt_mq | self.llm | StrOutputParser()

        generated_queries_str = mq_chain.invoke(
            {"question": question, "chat_history": formatted_history}
        )

        queries = [question] + [
            q.strip() for q in generated_queries_str.split("\n") if q.strip()
        ]
        print(f"Generated queries:\n{queries}")

        # --- 2. Parallel retrieval ---
        search_filter = self._build_filter(selected_files)
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 6, "filter": search_filter}
        )

        tasks = [retriever.ainvoke(q) for q in queries]
        all_results = await asyncio.gather(*tasks)

        # --- 3. Reciprocal Rank Fusion ---
        fused_docs = self._reciprocal_rank_fusion(all_results)
        final_docs = [doc for doc, _ in fused_docs[:6]]
        self.last_retrieved_docs = final_docs

        if not final_docs:
            return "No relevant context found for the selected documents."

        # --- 4. Context assembly ---
        context_text = ""
        for d in final_docs:
            context_text += (
                f"\n---\n"
                f"FILE: {d.metadata.get('source')} | "
                f"YEAR: {d.metadata.get('course')} | "
                f"DEGREE: {d.metadata.get('degree')}\n"
                f"CONTENT: {d.page_content}\n"
            )

        print("Final retrieved chunks (after RRF):")
        for i, doc in enumerate(final_docs):
            print(
                f"  {i+1}. {doc.metadata.get('course')} - "
                f"{doc.metadata.get('degree')} - "
                f"{doc.metadata.get('source', 'Unknown')} - "
                f"Chunk {doc.metadata.get('chunk_index', 0)}"
            )

        # --- 5. Answer generation ---
        template_qa = (
            "You are an expert Academic Advisor for university students.\n"
            "Your task is to answer the user's question using EXCLUSIVELY the provided context.\n\n"
            "STRICT RULES:\n"
            "1. NO EXTERNAL KNOWLEDGE: use only the provided fragments. If the context doesn't "
            "contain the answer, simply state that you don't know.\n"
            "2. CLARITY: be concise but clear. If the question is ambiguous, ask for clarification "
            "instead of guessing.\n"
            "3. STRUCTURE: use bullet points or numbered lists for complex information if needed.\n"
            "4. NO HALLUCINATIONS: do not invent dates, names of professors, or percentages if "
            "they are not explicitly in the context.\n"
            "5. LANGUAGE: respond in the same language as the user's question.\n\n"
            "CHAT HISTORY (for conversation flow):\n"
            "{chat_history}\n\n"
            "CONTEXT (relevant fragments from academic guides):\n"
            "{context}\n\n"
            "USER QUESTION:\n"
            "{question}\n\n"
            "ANSWER (precise, structured):"
        )

        prompt_qa = ChatPromptTemplate.from_template(template_qa)
        qa_chain = prompt_qa | self.llm | StrOutputParser()

        response = qa_chain.invoke(
            {
                "context": context_text,
                "question": question,
                "chat_history": formatted_history,
            }
        )

        # --- 6. Update session history (keep last 10 turns) ---
        self.session_history.append({"user": question, "bot": response})
        if len(self.session_history) > 10:
            self.session_history.pop(0)

        return response

    # ------------------------------------------------------------------
    # Document listing with full hierarchy
    # ------------------------------------------------------------------

    def list_documents(self) -> dict:
        """
        Returns a hierarchy dict: { course: { degree: [filenames] } }
        plus a flat list of unique sources for backwards compatibility.
        """
        collection = self.vectorstore._collection
        results = collection.get(include=["metadatas"])

        if not results or "metadatas" not in results:
            return {"hierarchy": {}, "sources": []}

        hierarchy: Dict[str, Dict[str, set]] = {}
        sources: set = set()

        for meta in results["metadatas"]:
            if not meta:
                continue
            course = meta.get("course", "Unknown")
            degree = meta.get("degree", "Unknown")
            source = meta.get("source", "Unknown")

            sources.add(source)

            if course not in hierarchy:
                hierarchy[course] = {}
            if degree not in hierarchy[course]:
                hierarchy[course][degree] = set()
            hierarchy[course][degree].add(f"[{course}] {source}")

        # Convert sets to sorted lists for JSON serialisation
        clean_hierarchy = {
            c: {d: sorted(list(files)) for d, files in degrees.items()}
            for c, degrees in hierarchy.items()
        }

        return {
            "hierarchy": clean_hierarchy,
            "sources": sorted(list(sources)),
        }

    # ------------------------------------------------------------------
    # Inspector  (context viewer for last query)
    # ------------------------------------------------------------------

    def get_inspector_html(self) -> str:
        """
        Returns an HTML string visualising the chunks retrieved in the last
        query, including their neighbouring chunks for context.
        """
        if not self.last_retrieved_docs:
            return (
                "<div style='padding:20px; text-align:center;'>"
                "No context available. Please ask a question first."
                "</div>"
            )

        html = "<h3 style='margin-bottom:20px; color:#333;'>Context used for the last response</h3>"

        for i, doc in enumerate(self.last_retrieved_docs):
            source = doc.metadata.get("source", "Unknown")
            course = doc.metadata.get("course", "Unknown")
            degree = doc.metadata.get("degree", "Unknown")
            idx = doc.metadata.get("chunk_index", 0)

            prev_id = f"{course}_{degree}_{source}_ch_{idx - 1}"
            next_id = f"{course}_{degree}_{source}_ch_{idx + 1}"

            text_prev = "<i>(Start of document)</i>"
            text_next = "<i>(End of document)</i>"

            try:
                neighbours = self.vectorstore._collection.get(ids=[prev_id, next_id])
                if neighbours and "ids" in neighbours:
                    for j, doc_id in enumerate(neighbours["ids"]):
                        if doc_id == prev_id:
                            text_prev = neighbours["documents"][j]
                        elif doc_id == next_id:
                            text_next = neighbours["documents"][j]
            except Exception:
                pass

            html += f"""
<div style="border:1px solid #ccc;border-radius:8px;margin-bottom:30px;overflow:hidden;
            font-family:sans-serif;box-shadow:0 2px 4px rgba(0,0,0,.1);background:#fff;">
  <div style="background:#e0e0e0;padding:8px 15px;border-bottom:1px solid #999;
              font-weight:bold;font-size:.9em;color:#000;">
    Rank #{i+1} | {course} | {degree} | {source} | Chunk ID: {idx}
  </div>
  <div style="background:#fff3e0;padding:10px;font-size:.85em;color:#444;border-bottom:1px dotted #ccc;">
    <strong style="color:#d84315;">Previous context:</strong><br>{text_prev}
  </div>
  <div style="background:#f1f8e9;padding:15px;font-size:1em;border-left:5px solid #4caf50;color:#000;">
    <strong style="color:#2e7d32;">Retrieved chunk:</strong><br>{doc.page_content}
  </div>
  <div style="background:#e3f2fd;padding:10px;font-size:.85em;color:#444;border-top:1px dotted #ccc;">
    <strong style="color:#1565c0;">Next context:</strong><br>{text_next}
  </div>
</div>
"""
        return html