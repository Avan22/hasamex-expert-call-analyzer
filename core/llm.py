"""Thin, swappable wrapper around the Claude API.

Every model call in the project goes through ``call_json``. The model id comes
from one environment variable (``CLAUDE_MODEL``), and output is constrained to
a JSON schema through structured outputs, so answers and citations come back
as separable fields rather than prose that has to be re-parsed.

Responses are cached on disk, keyed by a hash of the full request, so the
Streamlit app does not re-bill the same question on every rerun and the test
suite is reproducible. Delete ``.cache/`` or pass ``use_cache=False`` to force
fresh calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("core.llm")

DEFAULT_MODEL = "claude-sonnet-5"
CACHE_DIR = Path(os.getenv("HASAMEX_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache" / "llm"))

_client = None


class LLMError(RuntimeError):
    pass


def model_name() -> str:
    return os.getenv("CLAUDE_MODEL", DEFAULT_MODEL)


def _get_client():
    global _client
    if _client is None:
        import anthropic

        if not os.getenv("ANTHROPIC_API_KEY"):
            raise LLMError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = anthropic.Anthropic(max_retries=4)
    return _client


def _cache_key(model: str, system: str, messages: list[dict], schema: dict) -> str:
    blob = json.dumps(
        {"model": model, "system": system, "messages": messages, "schema": schema},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def call_json(
    system: str,
    messages: list[dict],
    schema: dict,
    *,
    max_tokens: int = 8000,
    use_cache: bool = True,
) -> dict:
    """Send one request and return the parsed JSON object the model produced."""
    import anthropic

    model = model_name()
    key = _cache_key(model, system, messages, schema)
    path = CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    client = _get_client()
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
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Model returned invalid JSON: {e}") from e

    log.info(
        "llm call model=%s in=%s out=%s",
        model, response.usage.input_tokens, response.usage.output_tokens,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data
