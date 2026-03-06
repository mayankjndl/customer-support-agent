"""
logger.py
---------
Query logging module.

Every interaction is logged to a JSONL file (one JSON object per line).
This format is:
  - Append-friendly (no need to load the whole file to add a new entry)
  - Easy to parse with pandas or any JSON library
  - Readable in any text editor

What we log (per query):
  - Timestamp
  - Session ID (for tracking multi-turn conversations)
  - The original user query
  - Source of the response: "rag", "weather_api", "currency_api", "fallback"
  - Confidence score from the RAG pipeline (0.0 if not applicable)
  - Retrieved documents (FAQ question + answer + relevance score)
  - Final response sent to the user
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from agent.config import LOG_DIR, LOG_FILE


class QueryLogger:
    """
    Appends structured log entries to a JSONL file.

    Usage:
        logger = QueryLogger()
        logger.log(
            session_id="abc123",
            query="Do you offer SEO?",
            source="rag",
            confidence=0.87,
            retrieved_docs=[...],
            response="Yes, we offer SEO services...",
        )
    """

    def __init__(self):
        # Create the logs directory if it doesn't exist
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        self.log_file = LOG_FILE
        print(f"[Logger] Logging queries to: {self.log_file}")

    def log(
        self,
        session_id: str,
        query: str,
        source: str,
        confidence: float,
        retrieved_docs: list,
        response: str,
    ) -> None:
        """
        Write a single log entry to the JSONL file.

        Args:
            session_id:    Client-provided identifier (e.g., UUID from frontend).
            query:         The raw user question.
            source:        Where the answer came from ("rag", "weather_api", etc.).
            confidence:    Max cosine similarity score from FAISS (0.0 if N/A).
            retrieved_docs: List of context dicts from the RAG pipeline.
            response:      The final answer string returned to the user.
        """
        log_entry = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "session_id":    session_id,
            "query":         query,
            "source":        source,
            "confidence":    confidence,
            "retrieved_docs": retrieved_docs,
            "response":      response,
        }

        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except IOError as e:
            # Logging failure should never crash the application
            print(f"[Logger] WARNING: Failed to write log entry: {e}")

    def read_logs(self, limit: int = 50) -> list[dict]:
        """
        Read the most recent log entries (useful for debugging / admin view).

        Args:
            limit: Maximum number of most-recent entries to return.

        Returns:
            A list of log entry dicts.
        """
        if not Path(self.log_file).exists():
            return []

        entries = []
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Return the most recent `limit` entries
        return entries[-limit:]
