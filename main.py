"""
main.py
-------
FastAPI application entry point.

Endpoints:
  POST /chat    — Main chat endpoint (query → answer)
  GET  /health  — Health check (confirms the agent is loaded and ready)
  GET  /logs    — View recent query logs (for debugging / demo)

Run with:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Or for production:
  uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional
import time

from agent.router import route_query, initialise_agent
from agent.logger import QueryLogger


# ─────────────────────────────────────────────
# Lifespan: runs on startup and shutdown
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the FAISS index, embedding model, and LLM client once at startup.
    This avoids re-loading heavy models on every request.
    """
    initialise_agent()
    yield
    # Shutdown cleanup (nothing needed here, but good practice to have the hook)
    print("[App] Shutting down.")


# ─────────────────────────────────────────────
# FastAPI App Initialisation
# ─────────────────────────────────────────────
app = FastAPI(
    title="PixelFlow AI Customer Support Agent",
    description=(
        "An AI-powered customer support agent for PixelFlow Digital Agency. "
        "Uses RAG over a business FAQ knowledge base and integrates live weather "
        "and currency exchange APIs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: Allow any origin for demo purposes (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request & Response Schemas (Pydantic)
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        example="Do you offer SEO services?",
        description="The user's question or message.",
    )
    session_id: str = Field(
        default="default",
        example="user-session-abc123",
        description="Optional identifier for grouping a conversation session.",
    )


class ChatResponse(BaseModel):
    answer: str = Field(description="The agent's response to the query.")
    source: str = Field(
        description="Where the answer came from: 'rag', 'weather_api', 'currency_api', 'fallback', or 'error'."
    )
    confidence: float = Field(
        description="Cosine similarity score from FAISS (0.0–1.0). 1.0 for API-sourced answers."
    )
    retrieved_context: list = Field(
        description="The FAQ entries or API data used to generate the answer."
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Response generation time in milliseconds."
    )


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, summary="Ask the AI support agent")
def chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint.

    - Detects intent (business FAQ / weather / currency)
    - Routes to RAG pipeline or external API
    - Returns a grounded, structured answer

    If RAG confidence is below the threshold, returns a graceful fallback
    offering human support instead of hallucinating an answer.
    """
    start_time = time.time()

    try:
        result = route_query(query=request.query, session_id=request.session_id)
    except Exception as e:
        # Unexpected errors should not expose stack traces to the client
        print(f"[API] Unhandled error: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")

    latency_ms = round((time.time() - start_time) * 1000, 2)

    return ChatResponse(
        answer=result["answer"],
        source=result["source"],
        confidence=result["confidence"],
        retrieved_context=result["retrieved_context"],
        latency_ms=latency_ms,
    )


@app.get("/health", summary="Health check")
def health_check() -> dict:
    """
    Returns a simple status message confirming the API is running.
    Useful for deployment monitoring and load balancer health checks.
    """
    return {"status": "ok", "agent": "PixelFlow Customer Support Agent v1.0.0"}


@app.get("/logs", summary="View recent query logs")
def get_logs(limit: int = 20) -> dict:
    """
    Returns the most recent query logs (for demo and debugging purposes).
    In production, this endpoint should be protected with authentication.
    """
    logger = QueryLogger()
    logs = logger.read_logs(limit=limit)
    return {"count": len(logs), "logs": logs}
