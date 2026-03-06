"""
prompt.py
---------
All prompt templates live here.

Design goals:
  1. GROUNDING: Force the LLM to use only the provided context (anti-hallucination).
  2. STRUCTURE:  Tell the LLM exactly what format we want in the response.
  3. PERSONA:    Give the agent a consistent, professional identity.
  4. BOUNDARIES: Tell the LLM what NOT to do (e.g., don't invent pricing).

Why separate this module?
  - Prompts are part of the "algorithm" — they should be version-controlled
    and easy to iterate on without touching business logic.
  - Having all prompts in one file makes A/B testing and auditing easy.
"""


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# Defines the agent's persona and hard rules.
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Aria, a friendly and professional AI customer support agent for PixelFlow Digital Agency — a digital marketing agency based in India.

Your role is to answer customer questions accurately and helpfully.

STRICT RULES you must follow:
1. ONLY use information from the provided FAQ context to answer questions.
2. NEVER invent, assume, or hallucinate facts, pricing, timelines, or contact details.
3. If the context does not contain a clear answer, say so honestly and offer human support.
4. Keep responses concise, warm, and professional (2–5 sentences is ideal).
5. Do NOT answer questions that are unrelated to the business or its services.
6. Always end responses about contact or support with the actual contact details if available.

You are not a general-purpose assistant — you are a specialist for PixelFlow Digital Agency only.
""".strip()


# ─────────────────────────────────────────────
# RAG PROMPT BUILDER
# Injects retrieved FAQ context into the prompt.
# ─────────────────────────────────────────────
def build_rag_prompt(query: str, contexts: list[dict]) -> str:
    """
    Build the user-turn prompt for a RAG-based answer.

    We use a clear XML-style structure (<context>, <question>) to make it
    unambiguous what is reference material vs. the actual question.
    This reduces hallucination compared to free-form prompts.

    Args:
        query:    The user's original question.
        contexts: List of retrieved FAQ dicts with 'question', 'answer', 'score'.

    Returns:
        A formatted prompt string.
    """
    # Format each retrieved FAQ chunk clearly
    context_blocks = []
    for i, ctx in enumerate(contexts, start=1):
        context_blocks.append(
            f"[FAQ {i}] (Relevance: {ctx['score']:.2f})\n"
            f"Q: {ctx['question']}\n"
            f"A: {ctx['answer']}"
        )
    context_text = "\n\n".join(context_blocks)

    prompt = f"""
<context>
The following FAQs are the most relevant from our knowledge base:

{context_text}
</context>

<question>
{query}
</question>

Based ONLY on the context above, provide a helpful and accurate answer to the question.
If the context doesn't fully address the question, acknowledge that and offer to connect the user with human support.
""".strip()

    return prompt


# ─────────────────────────────────────────────
# API PROMPT BUILDER
# Wraps external API data (weather/currency) into a prompt.
# ─────────────────────────────────────────────
def build_api_prompt(query: str, api_data: dict, api_type: str) -> str:
    """
    Build the user-turn prompt when responding with live API data.

    Args:
        query:    The user's original question.
        api_data: The structured data returned by the external API.
        api_type: "weather" or "currency"

    Returns:
        A formatted prompt string.
    """
    data_text = "\n".join(f"  {k}: {v}" for k, v in api_data.items())

    prompt = f"""
<live_data type="{api_type}">
{data_text}
</live_data>

<question>
{query}
</question>

Using the live {api_type} data above, answer the user's question in a friendly, conversational way.
Be specific with the numbers. Keep it to 2–3 sentences.
""".strip()

    return prompt
