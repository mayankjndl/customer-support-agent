# PixelFlow AI Customer Support Agent

An AI-powered customer support agent for a small digital marketing agency, built for the **Imperion Data Systems — AI & LLM Automation Intern** assignment.

---

## Architecture Overview

```
User Query (HTTP POST /chat)
        │
        ▼
┌───────────────────┐
│   FastAPI Server  │  ← main.py
│  (POST /chat)     │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Query Router    │  ← agent/router.py
│  (Intent Detect)  │
└──┬─────┬─────┬────┘
   │     │     │
   ▼     ▼     ▼
Weather Currency  RAG
 API     API   Pipeline
   │     │     │
   │     │     ├─ FAISS Vector Search  ← agent/rag.py
   │     │     │  (sentence-transformers)
   │     │     └─ Confidence Check
   │     │           │
   │     │      Low → Fallback
   │     │      High → Continue
   │     │
   └──┬──┘
      │
      ▼
┌─────────────────┐
│  Prompt Builder  │  ← agent/prompt.py
│  (Context Inject)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Groq LLM      │  ← agent/llm.py
│  (LLaMA 3.1 8B) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Logger   │  ← agent/logger.py
│  (JSONL file)   │
└────────┬────────┘
         │
         ▼
  Structured JSON Response
```

---

## Tech Stack

| Component        | Technology                     | Why                                               |
|------------------|--------------------------------|---------------------------------------------------|
| Backend          | FastAPI + Uvicorn              | Fast, async, auto-generates Swagger docs          |
| LLM              | Groq (LLaMA 3.1 8B Instant)   | Free tier, very low latency, familiar from LearnOS|
| Embeddings       | sentence-transformers (MiniLM) | Free, local, no API key needed for embeddings     |
| Vector Store     | FAISS (IndexFlatIP)            | Lightweight, no server needed, exact cosine search|
| External API 1   | OpenWeatherMap                 | Free tier, real-time weather data                 |
| External API 2   | ExchangeRate API               | Free, no API key, live currency rates             |
| Query Routing    | Keyword-based intent detection | Simple, transparent, and fully explainable        |
| Logging          | JSONL file                     | Append-friendly, zero dependencies, easy to parse |
| Config           | python-dotenv                  | Keep secrets out of source code                   |

---

## Project Structure

```
customer-support-agent/
├── main.py                    # FastAPI app: endpoints & lifespan hooks
├── requirements.txt           # Pinned dependencies
├── .env.example               # Environment variable template
├── .gitignore
│
├── agent/                     # Core agent logic (importable package)
│   ├── __init__.py
│   ├── config.py              # All constants, thresholds, env var loading
│   ├── rag.py                 # FAISS index loading + semantic retrieval
│   ├── llm.py                 # Groq LLM client wrapper
│   ├── api_tools.py           # OpenWeatherMap + ExchangeRate integrations
│   ├── prompt.py              # System prompt + prompt builder functions
│   ├── logger.py              # JSONL query logger
│   └── router.py              # Intent detection + orchestration
│
├── data/
│   ├── faqs.json              # Knowledge base: 16 FAQ entries
│   └── logs/                  # Auto-created: queries.jsonl
│
├── vector_store/              # Auto-created by build_index.py
│   ├── faqs.index             # FAISS binary index file
│   └── faqs_metadata.json     # FAQ dicts aligned to index positions
│
└── scripts/
    └── build_index.py         # One-time script to build FAISS index
```

---

## Setup Instructions

### Prerequisites
- Python 3.11+
- A free [Groq API key](https://console.groq.com/keys)
- A free [OpenWeatherMap API key](https://openweathermap.org/api)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/customer-support-agent.git
cd customer-support-agent
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # Linux/macOS
venv\Scripts\activate          # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Open .env and fill in your API keys:
#   GROQ_API_KEY=your_groq_key
#   OPENWEATHER_API_KEY=your_weather_key
```

### 5. Build the FAISS index (one-time)
```bash
python scripts/build_index.py
```
This downloads the `all-MiniLM-L6-v2` embedding model (~90MB) and builds the vector index from `data/faqs.json`.

### 6. Start the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. Test it
Open **http://localhost:8000/docs** for the interactive Swagger UI.

Or use `curl`:
```bash
# Business FAQ (RAG)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Do you offer SEO services?", "session_id": "demo"}'

# Weather API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Mumbai?", "session_id": "demo"}'

# Currency API
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Convert 100 USD to INR", "session_id": "demo"}'

# Fallback (unknown query)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the prime minister of France?", "session_id": "demo"}'
```

---

## API Reference

### `POST /chat`

**Request:**
```json
{
  "query": "What services do you offer?",
  "session_id": "user-abc123"
}
```

**Response:**
```json
{
  "answer": "We offer SEO, PPC, Social Media Management, Content Marketing...",
  "source": "rag",
  "confidence": 0.87,
  "retrieved_context": [
    {
      "question": "What digital marketing services do you offer?",
      "answer": "We offer a comprehensive range...",
      "category": "services",
      "score": 0.87
    }
  ],
  "latency_ms": 1243.5
}
```

**Source values:**
| Value          | Meaning                                          |
|----------------|--------------------------------------------------|
| `rag`          | Answered from the FAQ knowledge base             |
| `weather_api`  | Answered using live OpenWeatherMap data          |
| `currency_api` | Answered using live ExchangeRate API data        |
| `fallback`     | Confidence too low → human support offered       |
| `error`        | External API call failed                         |

### `GET /health`
Returns `{"status": "ok", "agent": "..."}`.

### `GET /logs?limit=20`
Returns the most recent N query log entries.

---

## RAG Pipeline Explained

1. **Knowledge Base**: 16 FAQs in `data/faqs.json` for "PixelFlow Digital Agency"
2. **Indexing** (`build_index.py`):
   - Each FAQ is encoded as `"Question: {q}\nAnswer: {a}"` for richer embeddings
   - Encoded with `all-MiniLM-L6-v2` (384-dimensional dense vectors)
   - Vectors are L2-normalised and stored in `faiss.IndexFlatIP`
   - With normalised vectors, **inner product = cosine similarity**
3. **Retrieval** (`rag.py`):
   - User query is encoded with the same model
   - FAISS returns top-3 most similar FAQs with similarity scores
   - If best score < 0.40 → return fallback (no hallucination)
4. **Generation** (`prompt.py` + `llm.py`):
   - Top-3 FAQs are injected into a structured prompt
   - LLM is instructed to answer **only** from the provided context

---

## Prompt Engineering

### System Prompt Design
The system prompt (`agent/prompt.py`) enforces 5 rules:
1. Only use information from the provided FAQ context
2. Never invent pricing, timelines, or contact details
3. Acknowledge uncertainty and offer human handoff
4. Keep responses concise (2–5 sentences)
5. Stay within the business domain

### Hallucination Prevention
- **Context injection**: The LLM sees the actual FAQ text in every request
- **Confidence gating**: If FAISS similarity < 0.40, the LLM is never called at all
- **Explicit instruction**: System prompt says "NEVER invent facts" in uppercase
- **Low temperature (0.3)**: Reduces creative/speculative generation

### Response Structure
The user prompt uses XML-style tags (`<context>`, `<question>`) to clearly delineate reference material from the question, making the prompt unambiguous for the model.

---

## External API Integration

### 1. OpenWeatherMap
- **Trigger**: Keywords like "weather", "temperature", "forecast", "rain", etc.
- **Extraction**: Regex patterns extract city name from query
- **Endpoint**: `api.openweathermap.org/data/2.5/weather`
- **Data returned**: Temperature, feels-like, condition, humidity, wind speed

### 2. ExchangeRate API
- **Trigger**: Keywords like "currency", "exchange rate", "convert", currency names/codes
- **Extraction**: Regex for 3-letter ISO codes + numeric amounts
- **Endpoint**: `api.exchangerate-api.com/v4/latest/{base}`
- **Data returned**: Live exchange rate + converted amount

---

## Fallback Logic

```python
if rag_confidence < 0.40:
    return "I'm not confident about this. Would you like to connect with human support?"
```

The 0.40 threshold was chosen empirically:
- Scores > 0.7: High confidence, clearly relevant FAQ
- Scores 0.4–0.7: Moderate confidence, related context
- Scores < 0.4: Low confidence, the knowledge base likely doesn't cover this

---

## Logging Format

Each query is logged to `data/logs/queries.jsonl`:

```json
{
  "timestamp": "2025-10-15T12:34:56.789Z",
  "session_id": "user-abc123",
  "query": "What is your pricing?",
  "source": "rag",
  "confidence": 0.83,
  "retrieved_docs": [{"question": "...", "answer": "...", "score": 0.83}],
  "response": "Our pricing starts at ₹15,000/month..."
}
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| FAISS over Chroma/Pinecone | No server, zero cloud cost, perfect for a small KB |
| Keyword routing over LLM routing | Transparent, debuggable, no extra API call |
| JSONL logging over database | No infrastructure needed, trivially portable |
| Combined Q+A embedding | Richer semantic signal for retrieval than question-only |
| Groq (LLaMA 3.1 8B) | Free, fast inference, no rate limits for demo |
| Low temperature (0.3) | Prioritises factual, grounded answers over creativity |
