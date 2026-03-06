"""
router.py
---------
The central orchestrator of the agent.

Flow:
  1. Receive user query
  2. Detect intent (weather / currency / business-FAQ)
  3. Route to the correct handler
  4. Build prompt with retrieved data
  5. Call LLM for a grounded response
  6. Log the interaction
  7. Return a structured response

Why keyword-based routing?
  - Simple, fast, and fully explainable (no black-box classification)
  - Easy to extend: add new keywords or a new intent without redesigning the system
  - For a production system, this could be upgraded to an intent-classification LLM call
"""

from agent.rag import RAGPipeline
from agent.llm import GroqLLMClient
from agent.logger import QueryLogger
from agent.prompt import build_rag_prompt, build_api_prompt
from agent.config import FALLBACK_MESSAGE
from agent.api_tools import (
    get_weather,
    extract_city_from_query,
    get_exchange_rate,
    extract_currency_params,
)


# ─────────────────────────────────────────────
# Intent Detection Keyword Sets
# ─────────────────────────────────────────────
WEATHER_KEYWORDS = {
    "weather", "temperature", "forecast", "rain", "sunny",
    "humidity", "wind", "climate", "drizzle", "cloudy", "hot", "cold",
    "snowfall", "storm", "heatwave",
}

CURRENCY_KEYWORDS = {
    "currency", "exchange rate", "convert", "conversion",
    "usd", "eur", "inr", "gbp", "jpy", "dollar", "euro",
    "rupee", "pound", "yen", "forex", "rate",
}


def detect_intent(query: str) -> str:
    """
    Detect the intent of a user query using keyword matching.

    Returns:
        "weather"  — if the query is about current weather
        "currency" — if the query is about exchange rates / conversion
        "rag"      — default: answer from the business knowledge base
    """
    query_lower = query.lower()

    if any(kw in query_lower for kw in WEATHER_KEYWORDS):
        return "weather"

    if any(kw in query_lower for kw in CURRENCY_KEYWORDS):
        return "currency"

    return "rag"


# ─────────────────────────────────────────────
# Singleton instances (initialised once at startup)
# ─────────────────────────────────────────────
# These are module-level so FastAPI doesn't reload them per request
_rag: RAGPipeline | None = None
_llm: GroqLLMClient | None = None
_logger: QueryLogger | None = None


def initialise_agent():
    """
    Load all heavy components once when the FastAPI app starts.
    Called from main.py's @app.on_event("startup") hook.
    """
    global _rag, _llm, _logger
    print("[Router] Initialising agent components...")
    _rag = RAGPipeline()
    _llm = GroqLLMClient()
    _logger = QueryLogger()
    print("[Router] Agent ready.")


# ─────────────────────────────────────────────
# Main Routing Function
# ─────────────────────────────────────────────
def route_query(query: str, session_id: str = "default") -> dict:
    """
    Main entry point: route a user query to the correct handler and return
    a structured response dict.

    Args:
        query:      The user's question as a plain string.
        session_id: An identifier for the conversation session.

    Returns:
        A dict with keys:
          - answer:            str  — the response to show the user
          - source:            str  — "rag" | "weather_api" | "currency_api" | "fallback" | "error"
          - confidence:        float — RAG score (0.0 for API/fallback responses)
          - retrieved_context: list — RAG contexts (empty for API responses)
    """
    intent = detect_intent(query)
    print(f"[Router] Query: '{query}' → Intent: {intent}")

    # ── WEATHER PATH ──────────────────────────────────────────────
    if intent == "weather":
        try:
            city = extract_city_from_query(query)
            weather_data = get_weather(city)
            prompt = build_api_prompt(query, weather_data, "weather")
            answer = _llm.generate(prompt)
            source = "weather_api"
            confidence = 1.0
            retrieved_context = [weather_data]
        except (ValueError, RuntimeError) as e:
            answer = f"I couldn't fetch the weather right now: {e}"
            source = "error"
            confidence = 0.0
            retrieved_context = []

    # ── CURRENCY PATH ─────────────────────────────────────────────
    elif intent == "currency":
        try:
            params = extract_currency_params(query)
            rate_data = get_exchange_rate(
                params["from"], params["to"], params["amount"]
            )
            prompt = build_api_prompt(query, rate_data, "currency")
            answer = _llm.generate(prompt)
            source = "currency_api"
            confidence = 1.0
            retrieved_context = [rate_data]
        except (ValueError, RuntimeError) as e:
            answer = f"I couldn't fetch exchange rates right now: {e}"
            source = "error"
            confidence = 0.0
            retrieved_context = []

    # ── RAG PATH ──────────────────────────────────────────────────
    else:
        retrieval = _rag.retrieve(query)
        retrieved_context = retrieval["contexts"]
        confidence = retrieval["max_score"]

        if not retrieval["is_confident"]:
            # Confidence too low → don't hallucinate, return fallback
            answer = FALLBACK_MESSAGE
            source = "fallback"
        else:
            prompt = build_rag_prompt(query, retrieved_context)
            answer = _llm.generate(prompt)
            source = "rag"

    # ── LOG EVERY INTERACTION ─────────────────────────────────────
    _logger.log(
        session_id=session_id,
        query=query,
        source=source,
        confidence=confidence,
        retrieved_docs=retrieved_context,
        response=answer,
    )

    return {
        "answer":            answer,
        "source":            source,
        "confidence":        confidence,
        "retrieved_context": retrieved_context,
    }
