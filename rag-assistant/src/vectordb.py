import os
from typing import List, Dict, Any
import torch
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from logger import LOGGER


class VectorDB:
    """Minimal ChromaDB wrapper."""

    def __init__(self, collection_name: str = None, embedding_model: str = None):
        try:
            LOGGER.log("Initializing VectorDB...", "INFO")
            self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "rag_docs")
            self.embedding_model_name = embedding_model or os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            LOGGER.log(f"Loading embedding model {self.embedding_model_name} on {device}", "DEBUG")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": device},
            )
            LOGGER.log(f"Loaded embedding model {self.embedding_model_name} on {device}", "DEBUG")
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            LOGGER.log("VectorDB ready.", "INFO")
        except Exception as e:
            LOGGER.log_exception(e)
            raise

    def chunk_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)

    def add_documents(self, documents: List[dict], source_prefix: str = "doc"):
        try:
            if not documents:
                LOGGER.log("No documents to add.", "WARN")
                return
            total_chunks = 0
            for d in documents:
                chunks = self.chunk_text(d["text"])
                ids = [f"{d['filename']}_chunk_{i}" for i in range(len(chunks))]
                metas = [{"title": d["filename"]} for _ in chunks]
                embeds = self.embedding_model.embed_documents(chunks)
                self.collection.add(embeddings=embeds, documents=chunks, metadatas=metas, ids=ids)
                total_chunks += len(chunks)
                LOGGER.log(f"Inserted {len(chunks)} chunks from {d['filename']}", "DEBUG")
            LOGGER.log(f"Total chunks added: {total_chunks}", "INFO")
        except Exception as e:
            LOGGER.log_exception(e)
            raise

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        try:
            if not query.strip():
                return {"documents": [], "metadatas": [], "distances": []}
            query_emb = self.embedding_model.embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_emb],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            if not results or not results.get("documents"):
                LOGGER.log("No relevant documents found.", "WARN")
                return {"documents": [], "metadatas": [], "distances": []}
            docs, metas, dists = results["documents"][0], results["metadatas"][0], results["distances"][0]
            LOGGER.log(f"Found {len(docs)} relevant chunks.", "INFO")
            return {"documents": docs, "metadatas": metas, "distances": dists}
        except Exception as e:
            LOGGER.log_exception(e)
            raise