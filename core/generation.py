"""Grounded answer generation.

The model gets a question plus a numbered set of retrieved transcript turns
and must return JSON: a list of short claims, each carrying the id of the
turn it relies on and a verbatim quote from it. Segment ids are constrained
by the schema to the ids actually supplied, so the model cannot cite a turn
it was not shown.

Nothing the model returns is shown until ``verify.py`` has checked it. The
loop is:

    retrieve -> generate -> verify every quote -> if any fail, send the
    failures and the real turn text back once and ask for a corrected answer
    -> verify again -> drop whatever still fails (and log it) -> drop any
    claim left with no verified quote.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from . import llm
from .retrieval import RetrievalHit
from .verify import CitationVerifier, FailedCitation, verify_payload

log = logging.getLogger("core.generation")

SYSTEM_PROMPT = """You are a research analyst at an expert network. You turn expert-call \
transcripts into answers that an institutional investor can rely on and check.

Rules, in order of importance:
1. Use only the transcript excerpts supplied in the user message. Do not use outside \
knowledge about robotic surgery, hospitals, markets or companies, even if you know it.
2. Every claim must be backed by at least one citation. A citation is the excerpt's \
segment_id plus a quote copied character-for-character from that excerpt's text: a \
contiguous span of at least 3 words, no ellipses, no joining of separate sentences, no \
changes to wording, spelling, tense or numbers. Quote only the expert's words, never the \
interviewer's question.
3. Claims must say no more than their quotes support. Keep the expert's own hedging \
("probably", "maybe", "in some areas") and use the same figures, ranges and units that \
appear in the quotes. Do not compute averages or convert numbers.
4. If the excerpts do not address the question, set answerable to false and return an \
empty points list. If they address it only partly, answer the supported part and state \
what is not covered in not_covered. Never fill a gap with a plausible guess.
5. Attribute views to the named expert (e.g. "Dr. Martin", "Anna Keller") when the \
answer draws on more than one expert.
6. Be concise: one sentence per claim, usually 2 to 4 claims. Do not put timestamps \
or segment ids in claim text; citations carry them."""


def format_excerpts(hits: list[RetrievalHit]) -> str:
    blocks = []
    for h in hits:
        s = h.segment
        q = f'\nInterviewer question before this turn: "{h.question_context}"' if h.question_context else ""
        blocks.append(
            f"<excerpt segment_id=\"{s.segment_id}\">\n"
            f"Expert: {s.expert_name} ({s.role}, {s.market}); timestamp {s.timestamp}; "
            f"speaker label \"{s.speaker}\"{q}\n"
            f"Expert's words: {s.text}\n"
            f"</excerpt>"
        )
    return "\n\n".join(blocks)


def citation_schema(segment_ids: list[str]) -> dict:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "segment_id": {"type": "string", "enum": sorted(set(segment_ids))},
                "quote": {"type": "string"},
            },
            "required": ["segment_id", "quote"],
            "additionalProperties": False,
        },
    }


def answer_schema(segment_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "answerable": {"type": "boolean"},
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "citations": citation_schema(segment_ids),
                    },
                    "required": ["claim", "citations"],
                    "additionalProperties": False,
                },
            },
            "not_covered": {"type": "string"},
        },
        "required": ["answerable", "points", "not_covered"],
        "additionalProperties": False,
    }


# --- deterministic claim check: numbers in a claim must appear in its quotes ---

# "one" is left out on purpose: it is far more often a pronoun ("one of the
# factors") than a figure, and the digit "1" is still checked.
_NUMBER_WORDS = {
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
}


def numbers_in(text: str) -> set[str]:
    found = set(re.findall(r"\d+(?:\.\d+)?", text))
    for word in re.findall(r"[a-z]+", text.lower()):
        if word in _NUMBER_WORDS:
            found.add(_NUMBER_WORDS[word])
    return found


def check_claim_numbers(
    payload, allowed: frozenset[str] = frozenset(), _path: str = "$"
) -> list[FailedCitation]:
    """For every dict holding verified citations plus a text field, require
    that every number in the text appears in the quotes it cites, or in
    ``allowed`` (numbers from the question itself, such as "3-5 years", and
    the number of experts, as in "all three experts"). Offending items have
    their citations cleared, which makes them unsupported."""
    issues: list[FailedCitation] = []
    if isinstance(payload, dict):
        cites = payload.get("citations")
        if isinstance(cites, list):
            text = " ".join(
                str(payload.get(k, "")) for k in ("claim", "summary", "position") if k in payload
            )
            quoted = set().union(*(numbers_in(c["quote"]) for c in cites)) if cites else set()
            missing = numbers_in(text) - quoted - allowed
            if cites and missing:
                issues.append(
                    FailedCitation(
                        "(claim)", text,
                        f"claim states number(s) {sorted(missing)} that do not appear in its quotes",
                        _path,
                    )
                )
                log.warning("claim rejected at %s: numbers %s not in quotes", _path, sorted(missing))
                payload["citations"] = []
        for k, v in payload.items():
            if k != "citations":
                issues.extend(check_claim_numbers(v, allowed, f"{_path}.{k}"))
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            issues.extend(check_claim_numbers(item, allowed, f"{_path}[{i}]"))
    return issues


# --- generic generate -> verify -> retry loop ---------------------------------

@dataclass
class VerifiedGeneration:
    payload: dict
    failures_first_pass: list[FailedCitation] = field(default_factory=list)
    failures_final: list[FailedCitation] = field(default_factory=list)
    attempts: int = 1


def _feedback_message(failures: list[FailedCitation], verifier: CitationVerifier) -> str:
    lines = [
        "Some citations in your answer failed automatic verification against the transcripts:",
    ]
    for f in failures:
        lines.append(f"- segment_id={f.claimed_segment_id} quote={f.quote!r}: {f.reason}")
        seg = verifier.segments.get(f.claimed_segment_id)
        if seg is not None:
            lines.append(f"  The exact text of {seg.segment_id} is: {seg.text!r}")
    lines.append(
        "Return the complete corrected JSON. Copy quotes character-for-character from the "
        "exact text above, keep the figures in each claim identical to its quotes, and remove "
        "any claim you cannot support with a verbatim quote."
    )
    return "\n".join(lines)


def generate_verified(
    system: str,
    user_content: str,
    schema: dict,
    verifier: CitationVerifier,
    allowed_expert_ids: set[str] | None = None,
    use_cache: bool = True,
    allowed_numbers: frozenset[str] = frozenset(),
) -> VerifiedGeneration:
    messages = [{"role": "user", "content": user_content}]
    raw = llm.call_json(system, messages, schema, use_cache=use_cache)
    payload, failures = verify_payload(raw, verifier, allowed_expert_ids)
    failures += check_claim_numbers(payload, allowed_numbers)
    result = VerifiedGeneration(payload, failures_first_pass=list(failures))
    if not failures:
        return result

    log.warning("%d citation(s) failed; retrying once with the source text", len(failures))
    retry_messages = messages + [
        {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
        {"role": "user", "content": _feedback_message(failures, verifier)},
    ]
    raw2 = llm.call_json(system, retry_messages, schema, use_cache=use_cache)
    payload2, failures2 = verify_payload(raw2, verifier, allowed_expert_ids)
    failures2 += check_claim_numbers(payload2, allowed_numbers)
    for f in failures2:
        log.warning("dropped after retry: %s | %r", f.reason, f.quote)
    result.payload, result.failures_final, result.attempts = payload2, failures2, 2
    return result


# --- answers ------------------------------------------------------------------

@dataclass
class Answer:
    question: str
    scope: list[str]                  # expert ids the answer is drawn from
    status: str                       # answered | not_discussed | no_verified_support | error
    points: list[dict] = field(default_factory=list)
    not_covered: str = ""
    retrieved: list[str] = field(default_factory=list)
    failures_first_pass: list[dict] = field(default_factory=list)
    failures_final: list[dict] = field(default_factory=list)
    attempts: int = 0
    error: str = ""

    @property
    def citations(self) -> list[dict]:
        return [c for p in self.points for c in p["citations"]]

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


NOT_DISCUSSED = "Not discussed in the transcripts."


def generate_answer(
    question: str,
    hits: list[RetrievalHit],
    verifier: CitationVerifier,
    scope: list[str],
    instruction: str = "",
    use_cache: bool = True,
) -> Answer:
    ans = Answer(question, scope, "not_discussed", retrieved=[h.segment.segment_id for h in hits])
    if not hits:
        ans.not_covered = NOT_DISCUSSED + " Retrieval found no transcript turn related to the question."
        return ans

    user = (
        f"{instruction}\n\nQuestion: {question}\n\n"
        f"Transcript excerpts (the only permitted source):\n\n{format_excerpts(hits)}"
    ).strip()
    try:
        gen = generate_verified(
            SYSTEM_PROMPT, user, answer_schema(ans.retrieved), verifier,
            allowed_expert_ids=set(scope), use_cache=use_cache,
            allowed_numbers=frozenset(numbers_in(question) | {str(len(verifier.transcripts))}),
        )
    except llm.LLMError as e:
        ans.status, ans.error = "error", str(e)
        return ans

    ans.attempts = gen.attempts
    ans.failures_first_pass = [f.to_dict() for f in gen.failures_first_pass]
    ans.failures_final = [f.to_dict() for f in gen.failures_final]
    ans.not_covered = gen.payload.get("not_covered", "").strip()

    if not gen.payload.get("answerable"):
        ans.status = "not_discussed"
        ans.not_covered = ans.not_covered or NOT_DISCUSSED
        return ans

    supported = [p for p in gen.payload.get("points", []) if p["citations"]]
    for p in gen.payload.get("points", []):
        if not p["citations"]:
            log.warning("claim dropped, no verified citation: %r", p["claim"])
    ans.points = supported
    ans.status = "answered" if supported else "no_verified_support"
    return ans
