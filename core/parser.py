"""Transcript parsing.

Turns an expert-call transcript (.txt) into a list of structured, immutable
segments. A segment is one speaker turn: the timestamp line that marks the
start of the turn plus the ``Speaker: text`` line that follows it.

The ``text`` field is stored exactly as it appears in the file (only the line
ending is removed). Nothing downstream is allowed to rewrite it; every quote
shown in the UI is re-sliced out of this field by ``verify.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TIMESTAMP_RE = re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})$")
HEADER_EXPERT_RE = re.compile(r"^Expert\s+\d+\s*[–—-]\s*(?P<name>.+)$")
SPEAKER_RE = re.compile(r"^(?P<speaker>[^:]{1,60}):\s?(?P<text>.*)$")
INTERVIEWER_NAMES = {"interviewer", "moderator", "analyst"}


class TranscriptParseError(ValueError):
    """Raised when a transcript does not follow the expected format."""


@dataclass(frozen=True)
class Segment:
    segment_id: str          # stable, human-readable id, e.g. "FR@05:07"
    expert_id: str           # short id of the transcript, e.g. "FR"
    expert_name: str
    role: str
    market: str
    timestamp: str           # raw "MM:SS" string exactly as in the file
    timestamp_seconds: int
    speaker: str             # speaker label exactly as in the file
    text: str                # exact original text, never altered
    is_expert: bool          # False for interviewer turns
    index: int               # position of the turn within its transcript
    source_line: int         # 1-based line number of the text line in the file


@dataclass
class Transcript:
    expert_id: str
    expert_name: str
    role: str
    market: str
    source_name: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def expert_segments(self) -> list[Segment]:
        return [s for s in self.segments if s.is_expert]

    def label(self) -> str:
        return f"{self.expert_name} ({self.role}, {self.market})"


def timestamp_to_seconds(ts: str) -> int:
    m = TIMESTAMP_RE.match(ts.strip())
    if not m:
        raise TranscriptParseError(f"Not a timestamp: {ts!r}")
    hours, minutes, seconds = m.groups()
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)


def _market_code(market: str, fallback: str) -> str:
    known = {"france": "FR", "germany": "DE", "united kingdom": "UK", "uk": "UK"}
    code = known.get(market.strip().lower())
    if code:
        return code
    letters = re.sub(r"[^A-Za-z]", "", market or fallback).upper()
    return (letters[:2] or "EX")


def parse_transcript_text(raw: str, source_name: str = "<memory>") -> Transcript:
    lines = [ln.rstrip("\r") for ln in raw.replace("\r\n", "\n").split("\n")]

    expert_name = role = market = ""
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if TIMESTAMP_RE.match(stripped):
            body_start = i
            break
        if m := HEADER_EXPERT_RE.match(stripped):
            expert_name = m.group("name").strip()
        elif stripped.lower().startswith("role:"):
            role = stripped.split(":", 1)[1].strip()
        elif stripped.lower().startswith("market:"):
            market = stripped.split(":", 1)[1].strip()
    else:
        raise TranscriptParseError(f"{source_name}: no timestamp lines found")

    if not expert_name:
        raise TranscriptParseError(f"{source_name}: missing 'Expert N – Name' header")

    expert_id = _market_code(market, expert_name)
    transcript = Transcript(expert_id, expert_name, role, market, source_name)

    i = body_start
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if not TIMESTAMP_RE.match(stripped):
            raise TranscriptParseError(
                f"{source_name}:{i + 1}: expected a timestamp line, got {stripped!r}"
            )
        ts = stripped
        # The speaker line is the next non-empty line.
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            raise TranscriptParseError(f"{source_name}:{i + 1}: timestamp with no speaker turn")
        m = SPEAKER_RE.match(lines[j].strip())
        if not m:
            raise TranscriptParseError(
                f"{source_name}:{j + 1}: expected 'Speaker: text', got {lines[j]!r}"
            )
        speaker = m.group("speaker").strip()
        text = m.group("text").strip()
        is_expert = speaker.lower() not in INTERVIEWER_NAMES
        transcript.segments.append(
            Segment(
                segment_id=f"{expert_id}@{ts}",
                expert_id=expert_id,
                expert_name=expert_name,
                role=role,
                market=market,
                timestamp=ts,
                timestamp_seconds=timestamp_to_seconds(ts),
                speaker=speaker,
                text=text,
                is_expert=is_expert,
                index=len(transcript.segments),
                source_line=j + 1,
            )
        )
        i = j + 1

    if not transcript.expert_segments:
        raise TranscriptParseError(f"{source_name}: no expert turns found")
    return transcript


def parse_transcript_file(path: str | Path) -> Transcript:
    path = Path(path)
    return parse_transcript_text(path.read_text(encoding="utf-8-sig"), path.name)


def load_transcripts(data_dir: str | Path) -> list[Transcript]:
    files = sorted(Path(data_dir).glob("Transcript_*.txt"))
    if not files:
        raise FileNotFoundError(f"No Transcript_*.txt files in {data_dir}")
    transcripts = [parse_transcript_file(f) for f in files]
    ids = [t.expert_id for t in transcripts]
    if len(set(ids)) != len(ids):
        raise TranscriptParseError(f"Duplicate expert ids across transcripts: {ids}")
    return transcripts


def parse_interview_guide(path: str | Path) -> list[str]:
    questions = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        m = re.match(r"^\s*\d+\.\s+(.+?)\s*$", line)
        if m:
            questions.append(m.group(1))
    if not questions:
        raise TranscriptParseError(f"{path}: no numbered questions found")
    return questions


def segment_index(transcripts: list[Transcript]) -> dict[str, Segment]:
    return {s.segment_id: s for t in transcripts for s in t.segments}


def preceding_question(transcript: Transcript, segment: Segment) -> Segment | None:
    """The interviewer turn immediately before an expert turn, if any."""
    if segment.index == 0:
        return None
    prev = transcript.segments[segment.index - 1]
    return None if prev.is_expert else prev
