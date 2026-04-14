# src/rag_engine.py
"""
RAG engine: embeds resume chunks into FAISS vector store,
retrieves relevant sections given a JD query.
"""

import os
import numpy as np
import faiss
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class ResumeRAG:
    """
    Embeds resume chunks using OpenAI embeddings,
    stores in FAISS, and retrieves relevant chunks for any query.
    """

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.index = None
        self.chunks = []
        self.dimension = 1536  # ada-002 embedding size

    def build_index(self, chunks: list[str]):
        """Embed all resume chunks and build a FAISS index."""
        self.chunks = chunks
        print(f"[RAG] Embedding {len(chunks)} resume chunks...")

        # Get embeddings for all chunks
        embedded = self.embeddings.embed_documents(chunks)
        vectors = np.array(embedded, dtype="float32")

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        # Build flat L2 index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(vectors)
        print(f"[RAG] Index built with {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """Retrieve the top_k most relevant resume chunks for a query."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Embed the query
        query_vec = np.array(
            [self.embeddings.embed_query(query)], dtype="float32"
        )
        faiss.normalize_L2(query_vec)

        # Search
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.chunks[idx])
        return results

    def get_full_context(self, queries: list[str], top_k: int = 3) -> str:
        """
        Run multiple queries (one per JD requirement) and 
        return deduplicated relevant resume context.
        """
        seen = set()
        all_chunks = []
        for query in queries:
            chunks = self.retrieve(query, top_k=top_k)
            for chunk in chunks:
                if chunk not in seen:
                    seen.add(chunk)
                    all_chunks.append(chunk)
        return "\n\n---\n\n".join(all_chunks)
