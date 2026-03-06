"""
rag.py
------
Retrieval-Augmented Generation (RAG) pipeline.

How it works:
  1. BUILD (one-time):  Load FAQs → encode with sentence-transformers → store in FAISS index
  2. RETRIEVE (runtime): Encode user query → cosine similarity search in FAISS → return top-k FAQs

Why FAISS?
  - Lightweight, runs fully in-memory, no external server needed
  - Great for small-to-medium knowledge bases (< 100k documents)
  - IndexFlatIP with normalised vectors = exact cosine similarity search

Why all-MiniLM-L6-v2?
  - Free, runs locally (no API key needed for embeddings)
  - 384-dimension vectors, good balance of speed and quality
  - Optimised for semantic similarity tasks
"""

import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

from agent.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    KNOWLEDGE_BASE_PATH,
    TOP_K_RESULTS,
    CONFIDENCE_THRESHOLD,
    FALLBACK_MESSAGE,
)


class RAGPipeline:
    """
    Manages the RAG pipeline: index loading and semantic retrieval.

    Usage:
        rag = RAGPipeline()
        result = rag.retrieve("Do you offer SEO services?")
    """

    def __init__(self):
        # Load the sentence-transformer model once at startup
        print(f"[RAG] Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None        # FAISS index (loaded from disk)
        self.metadata = []       # List of FAQ dicts matching index positions
        self._load_index()

    def _load_index(self) -> None:
        """Load the pre-built FAISS index and FAQ metadata from disk."""
        index_path = Path(FAISS_INDEX_PATH)
        metadata_path = Path(FAISS_METADATA_PATH)

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "FAISS index not found. Please run: python scripts/build_index.py"
            )

        print(f"[RAG] Loading FAISS index from {index_path}")
        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"[RAG] Index loaded with {self.index.ntotal} vectors.")

    def retrieve(self, query: str) -> dict:
        """
        Search the knowledge base for the top-k most relevant FAQ entries.

        Args:
            query: The user's question as a plain string.

        Returns:
            A dict with:
              - "contexts": list of {question, answer, score} for top matches
              - "max_score": float, highest cosine similarity score found
              - "is_confident": bool, whether max_score is above threshold
        """
        # Step 1: Encode the query using the same model that built the index
        # normalize_embeddings=True → vectors on unit sphere → inner product = cosine similarity
        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # Step 2: Search FAISS for top-k nearest neighbours
        # scores: cosine similarities (float), indices: positions in the index
        scores, indices = self.index.search(query_vector, TOP_K_RESULTS)

        scores = scores[0].tolist()    # Flatten from shape (1, k) to (k,)
        indices = indices[0].tolist()

        # Step 3: Assemble result contexts from metadata
        contexts = []
        for score, idx in zip(scores, indices):
            if idx == -1:              # FAISS returns -1 for empty slots
                continue
            faq = self.metadata[idx]
            contexts.append({
                "question": faq["question"],
                "answer": faq["answer"],
                "category": faq.get("category", "general"),
                "score": round(float(score), 4),
            })

        max_score = max(scores) if scores else 0.0
        is_confident = max_score >= CONFIDENCE_THRESHOLD

        return {
            "contexts": contexts,
            "max_score": round(max_score, 4),
            "is_confident": is_confident,
        }
