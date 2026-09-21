"""Cross-expert themes and disagreements, as a small map-reduce.

Map:    the per-expert answers to the interview-guide questions (already
        retrieved, generated and verified by ``generation.py``).
Reduce: one call that sees those per-expert findings plus the full text of
        every transcript turn they cite, and returns common themes and
        disagreements, each with citations.

The reduce output goes through the same verifier as everything else. A theme
survives only if verified quotes from at least two experts back it (its
coverage, e.g. 3/3, is shown in the UI), and a disagreement survives only if at
least two experts' positions each have their own verified quote.

At 3 transcripts the map stage is cheap. At 30+ it is the part that makes the
reduce possible, because the reduce sees compact per-expert findings rather
than every transcript (see README, "Scaling").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from . import llm
from .generation import SYSTEM_PROMPT, Answer, citation_schema, format_excerpts, generate_verified
from .parser import Transcript, preceding_question
from .retrieval import RetrievalHit
from .verify import CitationVerifier

log = logging.getLogger("core.themes")

THEMES_INSTRUCTION = """Compare the experts below and identify:

(a) Common themes: points on which the experts substantively agree. Prefer themes \
that all experts share. Each theme needs citations from every expert that holds it \
(one quote per expert is enough).

(b) Disagreements or divergences: points where the experts' views, emphasis or \
numbers differ in a way an investor would care about (for example growth \
expectations, purchase timelines, or how much economics versus clinical factors \
drive purchasing). For each, give every relevant expert's position with that \
expert's own quote. A difference only counts if the quotes show it; do not \
manufacture conflict between views that are merely phrased differently.

Summaries and positions follow the same rules as claims: no numbers that are not \
in their quotes, no timestamps, no outside knowledge. Return 3 to 5 themes and 2 to \
4 disagreements if the material supports them, fewer if it does not."""


def themes_schema(segment_ids: list[str], expert_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "themes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "citations": citation_schema(segment_ids),
                    },
                    "required": ["title", "summary", "citations"],
                    "additionalProperties": False,
                },
            },
            "disagreements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "summary": {"type": "string"},
                        "positions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "expert_id": {"type": "string", "enum": sorted(expert_ids)},
                                    "position": {"type": "string"},
                                    "citations": citation_schema(segment_ids),
                                },
                                "required": ["expert_id", "position", "citations"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["topic", "summary", "positions"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["themes", "disagreements"],
        "additionalProperties": False,
    }


@dataclass
class CrossExpertAnalysis:
    themes: list[dict] = field(default_factory=list)
    disagreements: list[dict] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    failures_first_pass: list[dict] = field(default_factory=list)
    failures_final: list[dict] = field(default_factory=list)
    attempts: int = 0
    error: str = ""

    @property
    def citations(self) -> list[dict]:
        out = [c for t in self.themes for c in t["citations"]]
        out += [c for d in self.disagreements for p in d["positions"] for c in p["citations"]]
        return out


def _map_summary(transcripts: list[Transcript], answers: list[Answer]) -> str:
    by_expert: dict[str, list[Answer]] = {}
    for a in answers:
        by_expert.setdefault(a.scope[0], []).append(a)
    parts = []
    for t in transcripts:
        lines = [f"## {t.expert_name} (expert_id {t.expert_id}; {t.role}, {t.market})"]
        for a in by_expert.get(t.expert_id, []):
            lines.append(f"Q: {a.question}")
            if a.status != "answered":
                lines.append(f"  - (no verified answer: {a.not_covered or a.status})")
            for p in a.points:
                cites = ", ".join(c["segment_id"] for c in p["citations"])
                lines.append(f"  - {p['claim']} [{cites}]")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def analyze(
    transcripts: list[Transcript],
    per_expert_answers: list[Answer],
    verifier: CitationVerifier,
    use_cache: bool = True,
) -> CrossExpertAnalysis:
    result = CrossExpertAnalysis()

    # Evidence for the reduce step: every turn cited by a verified map answer.
    cited_ids = sorted({c["segment_id"] for a in per_expert_answers for c in a.citations})
    if not cited_ids:
        result.error = "No verified per-expert answers to compare."
        return result
    hits = [
        RetrievalHit(verifier.segments[sid], 0.0, (), _question_before(transcripts, sid))
        for sid in cited_ids
    ]
    expert_ids = [t.expert_id for t in transcripts]
    user = (
        f"{THEMES_INSTRUCTION}\n\n"
        f"Per-expert findings (map stage, already verified):\n\n"
        f"{_map_summary(transcripts, per_expert_answers)}\n\n"
        f"Transcript excerpts (the only permitted source for quotes):\n\n{format_excerpts(hits)}"
    )
    try:
        gen = generate_verified(
            SYSTEM_PROMPT, user, themes_schema(cited_ids, expert_ids), verifier,
            allowed_expert_ids=set(expert_ids), use_cache=use_cache,
        )
    except llm.LLMError as e:
        result.error = str(e)
        return result

    result.attempts = gen.attempts
    result.failures_first_pass = [f.to_dict() for f in gen.failures_first_pass]
    result.failures_final = [f.to_dict() for f in gen.failures_final]

    for theme in gen.payload.get("themes", []):
        experts = sorted({c["expert_id"] for c in theme["citations"]})
        if len(experts) < 2:
            msg = f"theme dropped (verified support from {len(experts)} expert(s)): {theme['title']}"
            log.warning(msg)
            result.dropped.append(msg)
            continue
        theme["experts"] = experts
        theme["coverage"] = f"{len(experts)}/{len(expert_ids)}"
        result.themes.append(theme)

    for dis in gen.payload.get("disagreements", []):
        positions = [p for p in dis["positions"] if p["citations"]]
        if len({p["expert_id"] for p in positions}) < 2:
            msg = f"disagreement dropped (fewer than 2 verified positions): {dis['topic']}"
            log.warning(msg)
            result.dropped.append(msg)
            continue
        dis["positions"] = positions
        result.disagreements.append(dis)
    return result


def _question_before(transcripts: list[Transcript], segment_id: str) -> str | None:
    for t in transcripts:
        for s in t.segments:
            if s.segment_id == segment_id:
                q = preceding_question(t, s)
                return q.text if q else None
    return None
