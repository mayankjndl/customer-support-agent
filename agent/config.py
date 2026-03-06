"""
config.py
---------
Central configuration for the Customer Support Agent.
All constants, thresholds, and environment variables are loaded here
so they are easy to find, understand, and change.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()


# ─────────────────────────────────────────────
# LLM Configuration (Groq)
# ─────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = "llama-3.1-8b-instant"   # Fast, free-tier model on Groq
LLM_TEMPERATURE: float = 0.3                 # Low temperature → more factual, less creative
LLM_MAX_TOKENS: int = 600                    # Keeps responses concise


# ─────────────────────────────────────────────
# RAG Configuration
# ─────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # Lightweight, 384-dim, runs locally
FAISS_INDEX_PATH: str = "vector_store/faqs.index"
FAISS_METADATA_PATH: str = "vector_store/faqs_metadata.json"
KNOWLEDGE_BASE_PATH: str = "data/faqs.json"

TOP_K_RESULTS: int = 3                       # Number of FAQ chunks to retrieve
CONFIDENCE_THRESHOLD: float = 0.40           # Cosine similarity below this → fallback
                                             # Range: 0.0 (no match) to 1.0 (perfect match)


# ─────────────────────────────────────────────
# External API Configuration
# ─────────────────────────────────────────────
OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL: str = "http://api.openweathermap.org/data/2.5/weather"

# ExchangeRate API (free tier, no key required)
EXCHANGE_RATE_BASE_URL: str = "https://api.exchangerate-api.com/v4/latest"


# ─────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────
LOG_DIR: str = "data/logs"
LOG_FILE: str = "data/logs/queries.jsonl"    # JSONL = one JSON object per line


# ─────────────────────────────────────────────
# Fallback Message
# ─────────────────────────────────────────────
FALLBACK_MESSAGE: str = (
    "I'm not confident about this based on our knowledge base. "
    "Would you like to connect with a human support agent? "
    "You can reach us at support@pixelflow.in or call +91-98765-43210."
)
