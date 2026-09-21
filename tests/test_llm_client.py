import requests

from app.generation import llm_client
from app.generation.llm_client import LLMConfigError, LLMRequestError, generate_answer


class _FakeResponse:
    def __init__(self, status_code: int, json_data: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}", response=self)


def test_raises_config_error_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "groq_api_key", None)
    try:
        generate_answer("hello")
        assert False, "expected LLMConfigError"
    except LLMConfigError:
        pass


def test_succeeds_on_first_try(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "fake-key")

    def fake_post(*args, **kwargs):
        return _FakeResponse(200, {"choices": [{"message": {"content": " The answer. "}}]})

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    assert generate_answer("hello") == "The answer."


def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "fake-key")
    calls = {"n": 0}

    def fake_post(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            return _FakeResponse(429, text="rate limited")
        return _FakeResponse(200, {"choices": [{"message": {"content": "ok after retry"}}]})

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    monkeypatch.setattr(llm_client, "_post_with_retry", llm_client._post_with_retry.retry_with(
        stop=llm_client.stop_after_attempt(5), wait=llm_client.wait_exponential(multiplier=0.01, min=0.01, max=0.05)
    ))
    result = generate_answer("hello")
    assert result == "ok after retry"
    assert calls["n"] == 2


def test_raises_request_error_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "fake-key")

    def fake_post(*args, **kwargs):
        return _FakeResponse(500, text="server error")

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    monkeypatch.setattr(llm_client, "_post_with_retry", llm_client._post_with_retry.retry_with(
        stop=llm_client.stop_after_attempt(2), wait=llm_client.wait_exponential(multiplier=0.01, min=0.01, max=0.05)
    ))
    try:
        generate_answer("hello")
        assert False, "expected LLMRequestError"
    except LLMRequestError:
        pass


def test_includes_non_retryable_http_response(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "fake-key")

    def fake_post(*args, **kwargs):
        return _FakeResponse(404, text="model not found")

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    try:
        generate_answer("hello")
        assert False, "expected LLMRequestError"
    except LLMRequestError as exc:
        assert "Groq API error 404: model not found" in str(exc)


def test_malformed_response_shape_raises_request_error(monkeypatch):
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "fake-key")

    def fake_post(*args, **kwargs):
        return _FakeResponse(200, {"unexpected": "shape"})

    monkeypatch.setattr(llm_client.requests, "post", fake_post)
    try:
        generate_answer("hello")
        assert False, "expected LLMRequestError"
    except LLMRequestError:
        pass
