"""
Groq chat-completion client.

Fixes vs the original `generate_with_groq()`:
- Model name, endpoint, and API key are read from `app.config.settings`
  instead of being hardcoded constants.
- Transient failures (timeouts, 429 rate limits, 5xx) are retried with
  exponential backoff via `tenacity`, instead of a single `requests.post`
  that raises on the first hiccup.
- Errors are logged and re-raised as a small typed exception hierarchy so
  the API layer can map them to sensible HTTP status codes instead of a
  bare 500.
"""

from __future__ import annotations

import logging

import requests
from tenacity import (  # noqa: F401 - stop_after_attempt/wait_exponential re-exported for tests
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base class for LLM-client errors."""


class LLMConfigError(LLMError):
    """Raised when the client is misconfigured (e.g. missing API key)."""


class LLMRequestError(LLMError):
    """Raised when the Groq API call ultimately fails after retries."""


class _RetryableHTTPError(Exception):
    """Internal signal used to trigger a tenacity retry on 429/5xx responses."""

    def __init__(self, response: requests.Response):
        self.response = response
        super().__init__(f"Retryable HTTP status {response.status_code}")


def _raise_for_retry(response: requests.Response) -> None:
    if response.status_code == 429 or response.status_code >= 500:
        raise _RetryableHTTPError(response)
    response.raise_for_status()


@retry(
    reraise=True,
    stop=stop_after_attempt(max(settings.groq_max_retries, 1)),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, _RetryableHTTPError)),
)
def _post_with_retry(payload: dict) -> dict:
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    logger.debug("Calling Groq API (model=%s)", payload.get("model"))
    response = requests.post(
        settings.groq_endpoint,
        headers=headers,
        json=payload,
        timeout=settings.groq_timeout_seconds,
    )
    _raise_for_retry(response)
    return response.json()


def generate_answer(prompt: str, *, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Send a single-turn completion request and return the assistant text.

    The caller (generation/prompts.py + the RAG pipeline) is responsible for
    building the full prompt, including conversation history and retrieved
    context — this function only knows how to talk to Groq reliably.
    """
    if not settings.groq_api_key:
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key "
            "(get one at https://console.groq.com/keys)."
        )

    payload = {
        "model": settings.groq_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        data = _post_with_retry(payload)
    except _RetryableHTTPError as exc:
        logger.error("Groq API returned status %s after retries", exc.response.status_code)
        raise LLMRequestError(f"Groq API error {exc.response.status_code}: {exc.response.text[:300]}") from exc
    except requests.RequestException as exc:
        logger.error("Groq API request failed: %s", exc)
        raise LLMRequestError(f"Failed to reach Groq API: {exc}") from exc

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        logger.error("Unexpected Groq API response shape: %s", data)
        raise LLMRequestError("Groq API returned an unexpected response shape.") from exc
