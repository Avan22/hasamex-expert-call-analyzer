"""One object that wires the pipeline together for the UI and the tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .generation import Answer, generate_answer
from .parser import Transcript, load_transcripts, parse_interview_guide
from .retrieval import Retriever
from .themes import CrossExpertAnalysis, analyze
from .verify import CitationVerifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

GUIDE_TOP_K = 5        # expert turns retrieved per (question, expert)
FREEFORM_TOP_K_EACH = 3  # expert turns retrieved per expert for free-form questions


class Pipeline:
    def __init__(
        self,
        transcripts: list[Transcript] | None = None,
        questions: list[str] | None = None,
        data_dir: Path = DATA_DIR,
    ):
        self.transcripts = transcripts or load_transcripts(data_dir)
        self.questions = questions or parse_interview_guide(Path(data_dir) / "Interview_Guide.txt")
        self.retriever = Retriever(self.transcripts)
        self.verifier = CitationVerifier(self.transcripts)
        self.experts = {t.expert_id: t for t in self.transcripts}

    @property
    def expert_ids(self) -> list[str]:
        return [t.expert_id for t in self.transcripts]

    def answer_guide_question(self, question: str, expert_id: str, use_cache: bool = True) -> Answer:
        t = self.experts[expert_id]
        hits = self.retriever.search(question, [expert_id], top_k=GUIDE_TOP_K)
        instruction = (
            f"Answer the interview-guide question below for one expert only: {t.label()}. "
            f"Describe what this expert said; if they did not address part of the question, "
            f"say so in not_covered."
        )
        return generate_answer(question, hits, self.verifier, [expert_id], instruction, use_cache)

    def all_guide_answers(self, use_cache: bool = True, workers: int = 6) -> list[Answer]:
        jobs = [(q, eid) for q in self.questions for eid in self.expert_ids]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(lambda j: self.answer_guide_question(*j, use_cache=use_cache), jobs))

    def cross_expert(self, answers: list[Answer], use_cache: bool = True) -> CrossExpertAnalysis:
        return analyze(self.transcripts, answers, self.verifier, use_cache)

    def ask(self, question: str, use_cache: bool = True) -> Answer:
        question = question.strip()
        hits = self.retriever.search_per_expert(question, self.expert_ids, FREEFORM_TOP_K_EACH)
        instruction = (
            "Answer the user's question across all the expert interviews below. Name the "
            "expert behind each claim. If the experts differ, say so."
        )
        return generate_answer(question, hits, self.verifier, self.expert_ids, instruction, use_cache)
