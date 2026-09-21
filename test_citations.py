"""End-to-end citation audit against the live model.

    python test_citations.py            # uses the on-disk response cache
    python test_citations.py --fresh    # forces new model calls

Runs everything the app shows:
  * 6 interview-guide questions x 3 experts = 18 answers
  * cross-expert themes and disagreements
  * 6 realistic free-form questions
  * 3 out-of-scope questions that must come back "not discussed"

Then audits every citation that would be displayed. The audit deliberately
does NOT reuse core/verify.py: it re-reads the raw .txt files, finds the line
under the cited timestamp, and requires the displayed quote to appear in that
line verbatim, allowing only whitespace and case differences. An independent
check means a bug in the verifier cannot hide itself.

Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

from core.pipeline import DATA_DIR, Pipeline

FREE_FORM = [
    "How long does it take a hospital to purchase a robotic surgery system?",
    "What role does surgeon training play in whether a robotic programme succeeds?",
    "Do the experts expect robotic surgery procedure volumes to grow, and by how much?",
    "Is the purchasing decision driven more by finance or by clinical factors?",
    "How does adoption differ between large and small hospitals?",
    "What does the procurement team look at when evaluating a robotic system?",
]
OUT_OF_SCOPE = [
    "What did the experts say about robotic surgery adoption in Japan?",
    "What is Intuitive Surgical's current share price?",
    "Which robot vendor or specific system model do the experts prefer?",
]


# --- independent raw-file index ------------------------------------------------

def raw_index(data_dir: Path) -> dict[str, dict[str, tuple[str, str]]]:
    """expert name -> timestamp -> (speaker, text), read straight from the files."""
    index = {}
    for f in sorted(data_dir.glob("Transcript_*.txt")):
        lines = f.read_text(encoding="utf-8-sig").replace("\r\n", "\n").split("\n")
        name = re.sub(r"^Expert\s+\d+\s*\W\s*", "", lines[0]).strip()
        turns = {}
        for i, line in enumerate(lines):
            if re.fullmatch(r"\d{1,2}:\d{2}", line.strip()):
                nxt = next(l for l in lines[i + 1:] if l.strip())
                speaker, _, text = nxt.partition(":")
                turns[line.strip()] = (speaker.strip(), text.strip())
        index[name] = turns
    return index


def ws_case(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def audit_citation(c: dict, raw, where: str) -> str | None:
    turns = raw.get(c["expert_name"])
    if turns is None:
        return f"{where}: unknown expert {c['expert_name']!r}"
    if c["timestamp"] not in turns:
        return f"{where}: {c['expert_name']} has no turn at {c['timestamp']}"
    speaker, text = turns[c["timestamp"]]
    if speaker.lower() == "interviewer":
        return f"{where}: {c['timestamp']} is an interviewer turn"
    if ws_case(c["quote"]) not in ws_case(text):
        return (f"{where}: quote not verbatim in {c['expert_name']} @ {c['timestamp']}\n"
                f"      quote: {c['quote']!r}\n      line:  {text!r}")
    if len(c["quote"].split()) < 3:
        return f"{where}: quote too short to be meaningful: {c['quote']!r}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="bypass the response cache")
    ap.add_argument("--report", default="verification_report.json")
    args = ap.parse_args()
    use_cache = not args.fresh

    logging.basicConfig(level=logging.WARNING, format="  [log] %(name)s: %(message)s")
    raw = raw_index(DATA_DIR)
    pipe = Pipeline()
    failures: list[str] = []
    audited = 0
    first_pass_rejections = 0
    final_drops = 0
    report = {"guide": [], "themes": None, "free_form": [], "out_of_scope": []}

    def audit(cites, where):
        nonlocal audited
        for c in cites:
            audited += 1
            if err := audit_citation(c, raw, where):
                failures.append(err)

    t0 = time.time()
    print(f"Model: {__import__('core.llm', fromlist=['x']).model_name()}  cache={'on' if use_cache else 'off'}\n")

    # 1. Interview guide x experts
    print("[1/4] Interview-guide answers (6 questions x 3 experts)")
    answers = pipe.all_guide_answers(use_cache=use_cache)
    for a in answers:
        eid = a.scope[0]
        where = f"guide Q{pipe.questions.index(a.question) + 1} [{eid}]"
        first_pass_rejections += len(a.failures_first_pass)
        final_drops += len(a.failures_final)
        if a.status != "answered":
            failures.append(f"{where}: status={a.status} {a.error or a.not_covered}")
        n_cites = len(a.citations)
        audit(a.citations, where)
        print(f"  {where:<18} {a.status:<10} claims={len(a.points)} citations={n_cites} "
              f"attempts={a.attempts} first-pass-rejected={len(a.failures_first_pass)}")
        report["guide"].append(a.to_dict())

    # 2. Themes and disagreements
    print("\n[2/4] Cross-expert themes and disagreements")
    analysis = pipe.cross_expert(answers, use_cache=use_cache)
    if analysis.error:
        failures.append(f"themes: {analysis.error}")
    if not analysis.themes:
        failures.append("themes: no verified common themes produced")
    if not analysis.disagreements:
        failures.append("themes: no verified disagreements produced")
    first_pass_rejections += len(analysis.failures_first_pass)
    final_drops += len(analysis.failures_final)
    for i, t in enumerate(analysis.themes):
        audit(t["citations"], f"theme {i + 1}")
        print(f"  theme: {t['title']}  (coverage {t['coverage']}, {len(t['citations'])} citations)")
    for i, d in enumerate(analysis.disagreements):
        for p in d["positions"]:
            audit(p["citations"], f"disagreement {i + 1} [{p['expert_id']}]")
            if any(c["expert_id"] != p["expert_id"] for c in p["citations"]):
                failures.append(f"disagreement {i + 1}: position quoted another expert")
        print(f"  disagreement: {d['topic']}  ({len(d['positions'])} positions)")
    for msg in analysis.dropped:
        print(f"  dropped by verifier: {msg}")
    report["themes"] = analysis.__dict__

    # 3. Free-form smoke test
    print("\n[3/4] Free-form questions")
    for q in FREE_FORM:
        a = pipe.ask(q, use_cache=use_cache)
        where = f"free-form {q[:40]!r}"
        first_pass_rejections += len(a.failures_first_pass)
        final_drops += len(a.failures_final)
        if a.status != "answered":
            failures.append(f"{where}: expected an answer, got {a.status} {a.error or a.not_covered}")
        audit(a.citations, where)
        experts = sorted({c["expert_id"] for c in a.citations})
        print(f"  {a.status:<10} citations={len(a.citations)} experts={experts}  {q}")
        report["free_form"].append(a.to_dict())

    # 4. Out-of-scope questions must not be answered
    print("\n[4/4] Out-of-scope questions (must return 'not discussed')")
    for q in OUT_OF_SCOPE:
        a = pipe.ask(q, use_cache=use_cache)
        ok = a.status in ("not_discussed", "no_verified_support") and not a.points
        if not ok:
            failures.append(f"out-of-scope {q!r}: answered with {len(a.points)} claim(s): "
                            f"{[p['claim'] for p in a.points]}")
        print(f"  {'OK ' if ok else 'BAD'} {a.status:<14} {q}")
        report["out_of_scope"].append(a.to_dict())

    Path(args.report).write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    print("\n" + "=" * 72)
    print(f"Citations audited against raw transcript files: {audited}")
    print(f"Quotes rejected by verify.py on first pass (then retried): {first_pass_rejections}")
    print(f"Claims/quotes dropped after retry (never shown to user):  {final_drops}")
    print(f"Elapsed: {time.time() - t0:.1f}s   Full report: {args.report}")
    if failures:
        print(f"\nFAIL: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nPASS: {audited}/{audited} displayed citations verified verbatim at their "
          f"timestamps; 18/18 guide answers, themes, disagreements, "
          f"{len(FREE_FORM)} free-form and {len(OUT_OF_SCOPE)} out-of-scope checks OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
