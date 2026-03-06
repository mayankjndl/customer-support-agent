"""
api_tools.py
------------
External API integrations.

Currently supports:
  1. OpenWeatherMap API — current weather for any city
  2. ExchangeRate API   — live currency conversion (no API key needed)

Design principle:
  Each tool is a standalone function that:
    - Takes clean, extracted parameters (not a raw query string)
    - Returns a typed dict of data (or raises a clear exception)
    - Is completely independent of the LLM / RAG pipeline

This makes each API tool easy to test in isolation.
"""

import re
import requests
from agent.config import OPENWEATHER_API_KEY, OPENWEATHER_BASE_URL, EXCHANGE_RATE_BASE_URL


# ─────────────────────────────────────────────────────────────────
# WEATHER API
# ─────────────────────────────────────────────────────────────────

def get_weather(city: str) -> dict:
    """
    Fetch current weather for a city using the OpenWeatherMap API.

    Args:
        city: City name as a string (e.g., "Mumbai", "London").

    Returns:
        A dict with weather fields ready to be injected into a prompt.

    Raises:
        ValueError: If the API key is missing or the city is not found.
        RuntimeError: If the API call fails for any other reason.
    """
    if not OPENWEATHER_API_KEY:
        raise ValueError(
            "OPENWEATHER_API_KEY is not set in your .env file. "
            "Get a free key at https://openweathermap.org/api"
        )

    params = {
        "q": city,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",     # Celsius
    }

    try:
        response = requests.get(OPENWEATHER_BASE_URL, params=params, timeout=5)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            raise ValueError(f"City '{city}' not found. Please check the spelling.")
        raise RuntimeError(f"Weather API error: {e}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Network error fetching weather: {e}")

    data = response.json()

    # Extract and return only the fields we need (clean data contract)
    return {
        "city":        data["name"],
        "country":     data["sys"]["country"],
        "temperature": f"{data['main']['temp']}°C",
        "feels_like":  f"{data['main']['feels_like']}°C",
        "condition":   data["weather"][0]["description"].capitalize(),
        "humidity":    f"{data['main']['humidity']}%",
        "wind_speed":  f"{data['wind']['speed']} m/s",
    }


def extract_city_from_query(query: str) -> str:
    """
    Extract the city name from a weather-related user query using regex.

    Patterns handled:
      "weather in Mumbai"       → "Mumbai"
      "temperature at Delhi"    → "Delhi"
      "What's the weather in New York?" → "New York"

    Falls back to "Delhi" (our default city) if no city is found.
    """
    patterns = [
        r"(?:weather|temperature|forecast|climate|rain|sunny|cold|hot)\s+(?:in|at|for|of)\s+([A-Za-z\s]+?)(?:\?|$|\.|\s*$)",
        r"(?:in|at)\s+([A-Za-z\s]+?)\s+(?:weather|temperature|forecast)",
        r"(?:weather|temperature|forecast)\s+([A-Za-z\s]+?)(?:\?|$|\.)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            city = match.group(1).strip()
            if len(city) > 1:   # Avoid single-character false positives
                return city
    return "Delhi"   # Default fallback


# ─────────────────────────────────────────────────────────────────
# CURRENCY EXCHANGE API
# ─────────────────────────────────────────────────────────────────

# Map common currency names/words to ISO 4217 codes
CURRENCY_NAME_MAP = {
    "dollar": "USD",  "dollars": "USD",  "usd": "USD",
    "euro":   "EUR",  "euros":   "EUR",  "eur": "EUR",
    "rupee":  "INR",  "rupees":  "INR",  "inr": "INR",
    "pound":  "GBP",  "pounds":  "GBP",  "gbp": "GBP",
    "yen":    "JPY",  "jpy":     "JPY",
    "yuan":   "CNY",  "cny":     "CNY",
    "dirham": "AED",  "aed":     "AED",
    "franc":  "CHF",  "chf":     "CHF",
    "cad":    "CAD",  "aud":     "AUD",  "sgd": "SGD",
}


def get_exchange_rate(from_currency: str, to_currency: str, amount: float = 1.0) -> dict:
    """
    Fetch live currency exchange rates using the ExchangeRate API (free, no key).

    Args:
        from_currency: ISO code of the source currency (e.g., "USD").
        to_currency:   ISO code of the target currency (e.g., "INR").
        amount:        Amount to convert (default: 1.0).

    Returns:
        A dict with conversion details ready to inject into a prompt.
    """
    url = f"{EXCHANGE_RATE_BASE_URL}/{from_currency.upper()}"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Currency API error: {e}")

    data = response.json()

    if "rates" not in data:
        raise RuntimeError("Unexpected response format from currency API.")

    to_upper = to_currency.upper()
    if to_upper not in data["rates"]:
        raise ValueError(f"Currency code '{to_upper}' not recognised.")

    rate = data["rates"][to_upper]
    converted = round(amount * rate, 4)

    return {
        "from_currency": from_currency.upper(),
        "to_currency":   to_upper,
        "exchange_rate": rate,
        "amount":        amount,
        "converted":     converted,
        "result_string": f"{amount} {from_currency.upper()} = {converted} {to_upper}",
    }


def extract_currency_params(query: str) -> dict:
    """
    Extract currency conversion parameters from a user query.

    Examples:
      "convert 100 USD to INR"       → {from: USD, to: INR, amount: 100}
      "exchange rate dollar to euro" → {from: USD, to: EUR, amount: 1}
      "how many rupees is 50 pounds" → {from: GBP, to: INR, amount: 50}
    """
    query_lower = query.lower()

    # Find numeric amount (e.g., 100, 50.5)
    amount_match = re.search(r"\b(\d+(?:\.\d+)?)\b", query)
    amount = float(amount_match.group(1)) if amount_match else 1.0

    # Try to find 3-letter uppercase currency codes directly (e.g., "USD", "INR")
    codes = re.findall(r"\b([A-Z]{3})\b", query.upper())
    # Filter to known codes only
    valid_codes = [c for c in codes if c in {v for v in CURRENCY_NAME_MAP.values()}]

    if len(valid_codes) >= 2:
        return {"from": valid_codes[0], "to": valid_codes[1], "amount": amount}

    # Fall back to name-based matching
    found_codes = []
    for name, code in CURRENCY_NAME_MAP.items():
        if name in query_lower and code not in found_codes:
            found_codes.append(code)

    if len(found_codes) >= 2:
        return {"from": found_codes[0], "to": found_codes[1], "amount": amount}

    # Ultimate fallback
    return {"from": "USD", "to": "INR", "amount": amount}
