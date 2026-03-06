"""
scripts/build_index.py
-----------------------
One-time script: Build the FAISS vector index from the FAQ knowledge base.

Run this ONCE before starting the server:
  python scripts/build_index.py

What it does:
  1. Loads all FAQs from data/faqs.json
  2. Combines question + answer into a single text chunk per FAQ
     (so we embed the full semantic meaning, not just the question)
  3. Encodes each chunk with sentence-transformers (all-MiniLM-L6-v2)
  4. Normalises the vectors (required for cosine similarity via inner product)
  5. Stores them in a FAISS IndexFlatIP (exact search, inner product)
  6. Saves the index and FAQ metadata to disk

Run this again whenever you update data/faqs.json.
"""

import json
import sys
import os
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Allow running from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    KNOWLEDGE_BASE_PATH,
)


def build_index():
    print("=" * 60)
    print("Building FAISS index from FAQ knowledge base")
    print("=" * 60)

    # ── Step 1: Load FAQs ────────────────────────────────────────
    kb_path = Path(KNOWLEDGE_BASE_PATH)
    if not kb_path.exists():
        print(f"ERROR: Knowledge base not found at '{KNOWLEDGE_BASE_PATH}'")
        sys.exit(1)

    with open(kb_path, "r", encoding="utf-8") as f:
        faqs = json.load(f)

    print(f"✓ Loaded {len(faqs)} FAQs from {KNOWLEDGE_BASE_PATH}")

    # ── Step 2: Create text chunks ───────────────────────────────
    # We combine Q + A so the embedding captures both the question phrasing
    # AND the answer content. This improves retrieval for paraphrased queries.
    texts = []
    for faq in faqs:
        chunk = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
        texts.append(chunk)

    # ── Step 3: Encode with sentence-transformers ────────────────
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Encoding {len(texts)} FAQ chunks...")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,   # Normalise → inner product = cosine similarity
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    print(f"✓ Embeddings shape: {embeddings.shape}  (n_chunks × embedding_dim)")

    # ── Step 4: Build FAISS index ────────────────────────────────
    dimension = embeddings.shape[1]
    # IndexFlatIP = Flat (exact) index using Inner Product distance
    # With normalised vectors, inner product = cosine similarity
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"✓ FAISS IndexFlatIP built with {index.ntotal} vectors (dim={dimension})")

    # ── Step 5: Save index and metadata to disk ──────────────────
    Path(FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✓ FAISS index saved to: {FAISS_INDEX_PATH}")

    with open(FAISS_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(faqs, f, indent=2, ensure_ascii=False)
    print(f"✓ FAQ metadata saved to: {FAISS_METADATA_PATH}")

    print("\n" + "=" * 60)
    print("✅ Index built successfully! You can now start the server.")
    print("   Run: uvicorn main:app --reload")
    print("=" * 60)


if __name__ == "__main__":
    build_index()
