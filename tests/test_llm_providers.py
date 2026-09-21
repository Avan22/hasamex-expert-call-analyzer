"""Offline tests for the provider switch in core/llm.py.

The Gemini client is replaced with a fake that records the request, so these
check request shape and response handling without any network access.
"""

from types import SimpleNamespace

import pytest

from core import llm


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "_clients", {})
    monkeypatch.setattr(llm, "_gemini_limiter", llm._RateLimiter(0))
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)
    return sleeps
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("CLAUDE_MODEL", raising=False)


def test_provider_switch_and_defaults(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    assert llm.model_name() == "gemini-3.5-flash-lite" and llm.max_workers() == 3
    monkeypatch.setenv("MODEL_PROVIDER", "anthropic")
    assert llm.model_name() == "claude-sonnet-5" and llm.max_workers() == 6
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    with pytest.raises(llm.LLMError):
        llm.provider()


def test_cache_key_depends_on_provider():
    a = llm._cache_key("gemini", "m", "s", [], {})
    b = llm._cache_key("anthropic", "m", "s", [], {})
    assert a != b


class FakeGemini:
    def __init__(self, text='{"ok": true}', finish="STOP", block=None):
        self.requests = []
        self._resp = SimpleNamespace(
            text=text,
            prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name=finish))],
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5,
                                           thoughts_token_count=7),
        )
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.requests.append((model, contents, config))
        return self._resp


def test_gemini_request_shape(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    fake = FakeGemini()
    monkeypatch.setattr(llm, "_gemini_client", lambda: fake)
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
              "additionalProperties": False}
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "{}"},
                {"role": "user", "content": "fix it"}]
    out = llm.call_json("SYSTEM", messages, schema, use_cache=False)
    assert out == {"ok": True}
    model, contents, config = fake.requests[0]
    assert model == "gemini-3.5-flash-lite"
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert config.system_instruction == "SYSTEM"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema


def test_gemini_cache_hit_skips_network(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    fake = FakeGemini()
    monkeypatch.setattr(llm, "_gemini_client", lambda: fake)
    llm.call_json("S", [{"role": "user", "content": "q"}], {"type": "object"})
    llm.call_json("S", [{"role": "user", "content": "q"}], {"type": "object"})
    assert len(fake.requests) == 1


@pytest.mark.parametrize("finish", ["MAX_TOKENS", "SAFETY", "RECITATION"])
def test_gemini_bad_finish_reason_raises(monkeypatch, finish):
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "_gemini_client", lambda: FakeGemini(finish=finish))
    with pytest.raises(llm.LLMError, match="(?i)" + finish.split("_")[0]):
        llm.call_json("S", [{"role": "user", "content": "q"}], {"type": "object"}, use_cache=False)


def test_gemini_blocked_prompt_and_bad_json(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.setattr(llm, "_gemini_client", lambda: FakeGemini(block="SAFETY"))
    with pytest.raises(llm.LLMError, match="blocked"):
        llm.call_json("S", [{"role": "user", "content": "q"}], {"type": "object"}, use_cache=False)
    monkeypatch.setattr(llm, "_gemini_client", lambda: FakeGemini(text="not json"))
    with pytest.raises(llm.LLMError, match="invalid JSON"):
        llm.call_json("S", [{"role": "user", "content": "q"}], {"type": "object"}, use_cache=False)


def test_missing_key_is_clear_error(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert not llm.api_key_present()
    with pytest.raises(llm.LLMError, match="GEMINI_API_KEY"):
        llm.call_json("S", [{"role": "user", "content": "q"}], {"type": "object"}, use_cache=False)


def _quota_error(quota_id: str, delay: str = "18s"):
    from google.genai import errors

    return errors.ClientError(429, {"error": {
        "code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaId": quota_id}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay},
        ]}})


class FlakyGemini(FakeGemini):
    def __init__(self, errors_first):
        super().__init__()
        self.errors_first = list(errors_first)

    def _generate(self, model, contents, config):
        self.requests.append((model, contents, config))
        if self.errors_first:
            raise self.errors_first.pop(0)
        return self._resp


def test_gemini_429_waits_server_retry_delay_then_succeeds(monkeypatch, isolated):
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    fake = FlakyGemini([_quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")])
    monkeypatch.setattr(llm, "_gemini_client", lambda: fake)
    assert llm.call_json("S", [{"role": "user", "content": "q"}], {"type": "object"},
                         use_cache=False) == {"ok": True}
    assert len(fake.requests) == 2
    assert len(isolated) == 1 and isolated[0] == pytest.approx(19.0, abs=0.5)  # server delay + 1s


def test_gemini_daily_quota_fails_fast(monkeypatch):
    monkeypatch.setenv("MODEL_PROVIDER", "gemini")
    fake = FlakyGemini([_quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")])
    monkeypatch.setattr(llm, "_gemini_client", lambda: fake)
    with pytest.raises(llm.LLMError, match="daily quota"):
        llm.call_json("S", [{"role": "user", "content": "q"}], {"type": "object"}, use_cache=False)
    assert len(fake.requests) == 1


def test_rate_limiter_spaces_requests(monkeypatch):
    clock = {"t": 100.0}
    waits = []
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    rl = llm._RateLimiter(5)  # 12 s apart
    rl.acquire(); rl.acquire(); rl.acquire()
    assert waits == [12.0, 24.0]
