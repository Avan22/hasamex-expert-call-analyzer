"""Thin, swappable wrapper around the model API.

Every model call in the project goes through ``call_json``. Two real providers
sit behind it, chosen by one environment variable:

    MODEL_PROVIDER=anthropic   Claude via the Anthropic SDK (the documented
                               default choice; see README "Model choice")
    MODEL_PROVIDER=gemini      Gemini via Google's google-genai SDK (wired in to
                               get a verified live run on Gemini's free tier)

Both use schema-constrained JSON output, so answers and citations come back
as separable fields rather than prose that has to be re-parsed. The rest of
the pipeline (prompts, schemas, verification) is identical for both.

Responses are cached on disk, keyed by a hash of the provider, model and full
request, so the Streamlit app does not re-bill the same question on every
rerun and the test suite is reproducible. Delete ``.cache/`` or pass
``use_cache=False`` to force fresh calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("core.llm")

DEFAULT_PROVIDER = "gemini"
DEFAULT_MODELS = {"anthropic": "claude-sonnet-5", "gemini": "gemini-3.5-flash-lite"}
MODEL_ENV = {"anthropic": "CLAUDE_MODEL", "gemini": "GEMINI_MODEL"}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}
DEFAULT_WORKERS = {"anthropic": 6, "gemini": 3}

CACHE_DIR = Path(os.getenv("HASAMEX_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache" / "llm"))

_clients: dict[str, object] = {}
_stats_lock = threading.Lock()
# Counters so callers (e.g. test_citations.py) can prove calls were live.
stats = {"api_calls": 0, "cache_hits": 0, "retries": 0, "input_tokens": 0, "output_tokens": 0}


def _count(**deltas: int) -> None:
    with _stats_lock:
        for k, v in deltas.items():
            stats[k] += v or 0


class LLMError(RuntimeError):
    pass


def provider() -> str:
    p = os.getenv("MODEL_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if p not in DEFAULT_MODELS:
        raise LLMError(f"MODEL_PROVIDER must be 'anthropic' or 'gemini', got {p!r}")
    return p


def model_name() -> str:
    p = provider()
    return os.getenv(MODEL_ENV[p], DEFAULT_MODELS[p])


def describe() -> str:
    """Human-readable provider/model label for the UI and test output."""
    return f"{provider()} / {model_name()}"


def api_key_present() -> bool:
    return bool(os.getenv(KEY_ENV[provider()]))


def max_workers() -> int:
    return int(os.getenv("LLM_MAX_WORKERS", DEFAULT_WORKERS[provider()]))


def _require_key(p: str) -> str:
    key = os.getenv(KEY_ENV[p])
    if not key:
        raise LLMError(f"{KEY_ENV[p]} is not set. Copy .env.example to .env and add your key.")
    return key


def _cache_key(p: str, model: str, system: str, messages: list[dict], schema: dict) -> str:
    blob = json.dumps(
        {"provider": p, "model": model, "system": system, "messages": messages, "schema": schema},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def call_json(
    system: str,
    messages: list[dict],
    schema: dict,
    *,
    max_tokens: int = 16000,
    use_cache: bool = True,
) -> dict:
    """Send one request and return the parsed JSON object the model produced.

    ``messages`` is a list of ``{"role": "user"|"assistant", "content": str}``.
    """
    p, model = provider(), model_name()
    key = _cache_key(p, model, system, messages, schema)
    path = CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        _count(cache_hits=1)
        return json.loads(path.read_text(encoding="utf-8"))

    call = _call_anthropic if p == "anthropic" else _call_gemini
    text = call(model, system, messages, schema, max_tokens)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"{p} returned invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise LLMError(f"{p} returned JSON that is not an object")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


# --- Anthropic (Claude) --------------------------------------------------------

def _call_anthropic(model: str, system: str, messages: list[dict], schema: dict, max_tokens: int) -> str:
    import anthropic

    client = _clients.get("anthropic")
    if client is None:
        client = _clients["anthropic"] = anthropic.Anthropic(
            api_key=_require_key("anthropic"), max_retries=4
        )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
    except anthropic.AuthenticationError as e:
        raise LLMError("Anthropic API rejected the key (401). Check ANTHROPIC_API_KEY.") from e
    except anthropic.NotFoundError as e:
        raise LLMError(f"Model {model!r} not found. Check CLAUDE_MODEL.") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Rate limited by the Anthropic API after retries.") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the Anthropic API.") from e

    if response.stop_reason == "refusal":
        raise LLMError("The model declined this request.")
    if response.stop_reason == "max_tokens":
        raise LLMError("Model output hit max_tokens before the JSON was complete.")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise LLMError("Model returned no text block.")
    _count(api_calls=1, input_tokens=response.usage.input_tokens,
           output_tokens=response.usage.output_tokens)
    log.info("anthropic model=%s in=%s out=%s", model,
             response.usage.input_tokens, response.usage.output_tokens)
    return text


# --- Google Gemini -------------------------------------------------------------

# Gemini free tier is rate limited per model (the live API reported 5 requests
# per minute and, for gemini-3.6-flash, 20 per day) and Flash models return 503
# during demand spikes. Retries are therefore done
# here rather than inside the SDK, so that every attempt goes through one shared
# rate limiter, honours the server's RetryInfo delay, and is logged.
_GEMINI_TRANSIENT = {429, 500, 502, 503, 504}


class _RateLimiter:
    """Spaces request starts at least 60/rpm seconds apart, across threads."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            self._next = max(now, self._next) + self.interval
        if wait > 0:
            time.sleep(wait)

    def push_back(self, seconds: float) -> None:
        """After a 429, delay every waiting request by the server's retry delay."""
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


_gemini_limiter = _RateLimiter(float(os.getenv("GEMINI_RPM", 5)))


def _gemini_client():
    from google import genai
    from google.genai import types

    client = _clients.get("gemini")
    if client is None:
        client = _clients["gemini"] = genai.Client(
            api_key=_require_key("gemini"),
            http_options=types.HttpOptions(
                timeout=300_000,  # milliseconds, per attempt
                retry_options=types.HttpRetryOptions(attempts=1),  # retries handled below
            ),
        )
    return client


def _quota_info(err) -> tuple[float | None, str]:
    """Pull the server's retry delay and the violated quota id out of a 429."""
    delay, quota_id = None, ""
    details = (getattr(err, "details", None) or {}).get("error", {}).get("details", [])
    for d in details:
        kind = d.get("@type", "")
        if kind.endswith("RetryInfo"):
            m = re.match(r"([\d.]+)s", str(d.get("retryDelay", "")))
            if m:
                delay = float(m.group(1))
        elif kind.endswith("QuotaFailure"):
            quota_id = ",".join(v.get("quotaId", "") for v in d.get("violations", []))
    return delay, quota_id


def _call_gemini(model: str, system: str, messages: list[dict], schema: dict, max_tokens: int) -> str:
    from google.genai import errors, types

    client = _gemini_client()
    # Gemini's roles are "user" and "model"; the conversation is sent in full
    # on every call (stateless), exactly like the Anthropic path.
    contents = [
        types.Content(
            role="model" if m["role"] == "assistant" else "user",
            parts=[types.Part(text=m["content"])],
        )
        for m in messages
    ]
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_json_schema=schema,
        max_output_tokens=max_tokens,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    attempts = int(os.getenv("GEMINI_RETRY_ATTEMPTS", 8))
    for attempt in range(1, attempts + 1):
        _gemini_limiter.acquire()
        try:
            response = client.models.generate_content(model=model, contents=contents, config=config)
            break
        except errors.APIError as e:
            code = e.code or 0
            if code in (401, 403):
                raise LLMError(f"Gemini API rejected the key ({code}). Check GEMINI_API_KEY.") from e
            if code == 404:
                raise LLMError(f"Gemini model {model!r} not found or not available. Check GEMINI_MODEL.") from e
            if code not in _GEMINI_TRANSIENT:
                raise LLMError(f"Gemini API error {code}: {e.message}") from e
            if code == 429:
                delay, quota_id = _quota_info(e)
                if "PerDay" in quota_id:
                    raise LLMError(f"Gemini free-tier daily quota exhausted ({quota_id}).") from e
                delay = (delay or 20.0) + 1.0
                _gemini_limiter.push_back(delay)
            else:
                delay = min(60.0, 5.0 * 2 ** (attempt - 1))
            if attempt == attempts:
                raise LLMError(f"Gemini API still failing after {attempts} attempts ({code}): {e.message}") from e
            log.warning("gemini %s on attempt %d/%d; retrying in %.0fs", code, attempt, attempts, delay)
            _count(retries=1)
            if code != 429:  # a 429 already pushed the shared limiter back; acquire() waits
                time.sleep(delay)

    feedback = response.prompt_feedback
    if feedback is not None and feedback.block_reason:
        raise LLMError(f"Gemini blocked the prompt ({feedback.block_reason}).")
    if not response.candidates:
        raise LLMError("Gemini returned no candidates.")
    finish = response.candidates[0].finish_reason
    finish_name = getattr(finish, "name", str(finish))
    if finish_name == "MAX_TOKENS":
        raise LLMError("Gemini output hit max_output_tokens before the JSON was complete.")
    if finish_name not in ("STOP", "FINISH_REASON_UNSPECIFIED", "None"):
        # SAFETY, RECITATION, BLOCKLIST, PROHIBITED_CONTENT, MALFORMED_... etc.
        raise LLMError(f"Gemini stopped early (finish_reason={finish_name}).")
    text = response.text
    if not text:
        raise LLMError("Gemini returned an empty response.")
    usage = response.usage_metadata
    _count(api_calls=1,
           input_tokens=getattr(usage, "prompt_token_count", 0),
           output_tokens=(getattr(usage, "candidates_token_count", 0) or 0)
           + (getattr(usage, "thoughts_token_count", 0) or 0))
    if usage is not None:
        log.info("gemini model=%s in=%s out=%s thinking=%s", model, usage.prompt_token_count,
                 usage.candidates_token_count, usage.thoughts_token_count)
    return text
