"""Retrieval: find the transcript segments most relevant to a question.

Okapi BM25 over expert turns, implemented in pure Python (no extra
dependencies). Each expert turn is indexed together with the interviewer
question that prompted it, because a short answer like "Very important."
only makes sense next to its question. The unit returned is always the
expert turn itself: interviewer text is context, never quotable evidence.

At 3 transcripts the whole corpus would fit in one prompt. Retrieval is still
a real step here because it is the seam that has to carry the load at 30+
transcripts (see README, "Scaling"). The index is built once per corpus and
reused for every query; swapping BM25 for embeddings or a vector store only
means replacing ``Retriever.search``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .parser import Segment, Transcript, preceding_question

STOPWORDS = set(
    """a about above after again all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers him his how i if
    in into is it its itself just me more most my no nor not now of off on once only
    or other our out over own same she should so some such than that the their them
    then there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours
    expert experts think thinks say says said tell told describe view views opinion
    opinions transcript transcripts interview call calls mention mentioned discuss
    discussed talk talked anyone everyone someone across between agree disagree
    """.split()
)

# Small, explicit domain synonym map so that e.g. "ROI" finds turns about
# "economic case" and "pay for itself". Kept deliberately short and visible.
SYNONYMS = {
    "roi": ["economic", "economics", "pay", "financial", "finance", "cost"],
    "budget": ["capital", "funding", "finance", "cost"],
    "price": ["cost"],
    "pricing": ["cost"],
    "expensive": ["cost"],
    "money": ["cost", "funding", "finance"],
    "timeline": ["month", "long", "cycle"],
    "time": ["month", "long"],
    "duration": ["month", "long"],
    "forecast": ["expect", "growth", "outlook"],
    "outlook": ["expect", "growth"],
    "future": ["expect", "growth", "outlook"],
    "trend": ["growth", "expect", "increasing"],
    "growth": ["growing", "increasing", "grow"],
    "barrier": ["issue", "holding", "stall", "barrier"],
    "obstacle": ["barrier", "issue"],
    "challenge": ["barrier", "issue"],
    "training": ["train", "trained", "surgeon"],
    "outcome": ["clinical", "outcome", "patient"],
    "adoption": ["adoption", "growing", "increasing"],
    "procurement": ["purchase", "purchasing", "procurement", "committee"],
    "buy": ["purchase", "buying"],
    "decision": ["decide", "decision", "approval", "approved"],
    "utilization": ["utilisation"],
    "utilisation": ["utilization"],
}


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _raw_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _raw_tokens(text)]


def query_tokens(query: str) -> list[str]:
    """Tokenize a query and add domain synonyms (looked up before stemming)."""
    raw = _raw_tokens(query)
    expanded = [_stem(t) for t in raw]
    for t in raw:
        for syn in SYNONYMS.get(t, SYNONYMS.get(_stem(t), [])):
            expanded.append(_stem(syn))
    return expanded


@dataclass(frozen=True)
class RetrievalHit:
    segment: Segment
    score: float
    matched_terms: tuple[str, ...]
    question_context: str | None  # the interviewer question before this turn


class Retriever:
    def __init__(self, transcripts: list[Transcript], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs: list[tuple[Segment, str | None, Counter, int]] = []
        for t in transcripts:
            for seg in t.expert_segments:
                q = preceding_question(t, seg)
                q_text = q.text if q else None
                tokens = tokenize(seg.text) + tokenize(q_text or "")
                self.docs.append((seg, q_text, Counter(tokens), len(tokens)))
        n = len(self.docs)
        self.avgdl = sum(d[3] for d in self.docs) / max(n, 1)
        df: Counter = Counter()
        for _, _, tf, _ in self.docs:
            df.update(tf.keys())
        self.idf = {
            term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def search(
        self,
        query: str,
        expert_ids: list[str] | None = None,
        top_k: int = 5,
    ) -> list[RetrievalHit]:
        q_tokens = query_tokens(query)
        q_counts = Counter(q_tokens)
        hits: list[RetrievalHit] = []
        for seg, q_text, tf, dl in self.docs:
            if expert_ids and seg.expert_id not in expert_ids:
                continue
            score = 0.0
            matched = []
            for term, qf in q_counts.items():
                f = tf.get(term, 0)
                if not f:
                    continue
                matched.append(term)
                idf = self.idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                # Expanded synonyms that repeat a term add weight via qf, capped.
                score += idf * (f * (self.k1 + 1)) / denom * min(qf, 2)
            if score > 0:
                hits.append(RetrievalHit(seg, score, tuple(sorted(matched)), q_text))
        hits.sort(key=lambda h: (-h.score, h.segment.expert_id, h.segment.timestamp_seconds))
        return hits[:top_k]

    def search_per_expert(
        self, query: str, expert_ids: list[str], top_k_each: int = 4
    ) -> list[RetrievalHit]:
        """Balanced retrieval: top hits from each expert, so one talkative
        expert cannot crowd the others out of a cross-expert answer."""
        out: list[RetrievalHit] = []
        for eid in expert_ids:
            out.extend(self.search(query, [eid], top_k_each))
        return out
