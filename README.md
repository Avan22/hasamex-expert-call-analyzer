# Expert Call Analyzer

A small, grounded research tool for expert-network calls. It reads three expert
interviews on the European robotic surgery market, answers the six interview-guide
questions for each expert, identifies common themes and disagreements across the
experts, and lets you ask your own questions about the calls. Every claim on screen
carries at least one quote and a timestamp. Code, not the model, has checked each
quote character by character against the transcript, and one click highlights it
in the source.

The design goal is **traceability over cleverness**. A wrong or invented quote is
worse than a missing answer, so the model is treated as a proposer and a
deterministic verifier decides what the user sees. The project has two layers:
a plain Python pipeline (`core/`) that can be run, tested and demoed without a UI,
and a thin Streamlit UI (`app.py`) on top of it. That split keeps the part that
matters (retrieval, generation, verification) inspectable on its own.

---

## Setup and run

Requires Python 3.10+ and an Anthropic API key.

```bash
git clone https://github.com/Avan22/hasamex-expert-call-analyzer.git
cd hasamex-expert-call-analyzer
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then put your key in .env: ANTHROPIC_API_KEY=sk-ant-...
```

Launch the app:

```bash
streamlit run app.py
```

It opens at http://localhost:8501. On the first load the app answers 18
questions (6 guide questions x 3 experts), which takes about a minute. The
results are cached in `.cache/`, so later loads are instant. The sidebar also
accepts uploaded transcripts in the same `.txt` format.

Run the tests:

```bash
pytest -q                          # offline unit tests, no API key needed (~0.1 s)
python test_citations.py           # end-to-end citation audit against the live model
python test_citations.py --fresh   # same, ignoring the response cache
```

To use a different model, set `CLAUDE_MODEL` in `.env` (for example `claude-opus-5`).
No code changes are needed.

---

## What it does (mapped to the brief)

| Requirement | Where |
|---|---|
| Read the 3 transcripts | `core/parser.py`, plus optional upload in the sidebar |
| Answer the 6 guide questions per expert | Tab **Per-expert Q&A** (18 answers) |
| Exact quotes with timestamps | Every claim shows the verified quote, expert and `MM:SS` |
| Common themes and disagreements | Tab **Themes & disagreements**, each side quoted |
| Ask across all transcripts | Tab **Ask a question**, with a visible "Not discussed in the transcripts" fallback |
| Don't invent, stay traceable | `core/verify.py`, a claim-level numbers check, and **View in transcript** on every quote |

---

## Architecture

```
data/*.txt ──► parser ──► segments (immutable, exact text, MM:SS, speaker)
                               │
question ──► retrieval (BM25) ─┴─► top expert turns
                                        │
                              generation (Claude, JSON schema)
                                        │  claims + {segment_id, quote}
                                        ▼
                              verify ──► pass: re-slice exact text from source
                                   └──► fail: log it, send the real text back,
                                              retry once, drop what still fails
                                        │
            themes (map-reduce over verified per-expert answers, verified again)
                                        │
                                  Streamlit UI
```

**Layer 1: `core/`, an importable pipeline with no UI dependency**

- `parser.py` turns each transcript into `Segment` records: `expert_name`, `role`,
  `market`, `timestamp` (the raw `MM:SS`), `timestamp_seconds`, `speaker`, `text`
  (verbatim, never rewritten), plus a stable id such as `DE@06:05`. Interviewer turns
  are kept for context and flagged `is_expert=False`, so they can never be cited.
- `retrieval.py` implements Okapi BM25 in pure Python over expert turns. Each turn is
  indexed together with the interviewer question that prompted it, because a short
  answer only makes sense next to its question. A small, visible synonym map covers
  domain terms (ROI → economic, pay, cost). Free-form questions retrieve the top turns
  per expert, so one expert can't crowd out the others. If nothing matches, the model
  is not called at all.
- `llm.py` is the only place the model is called. It reads the model id from
  `CLAUDE_MODEL` and uses **structured outputs** (`output_config.format` with a JSON
  schema), so answers and citations come back as separate fields instead of prose to
  re-parse. It maps API errors to clear messages and caches responses on disk.
- `generation.py` holds the system prompt, the answer schema and the
  generate → verify → retry loop. The schema restricts each citation's `segment_id` to
  an `enum` of the turns that were actually retrieved, so the model cannot cite a turn
  it wasn't shown. Answers come back as a list of claims, each with its own citations,
  rather than one paragraph with a list of quotes at the end.
- `verify.py` is the gate between the model and the user (details below).
- `themes.py` runs cross-expert analysis as a small map-reduce. The map step is the 18
  verified per-expert answers. The reduce step is one call that sees those findings
  plus the full text of every turn they cite. Its output goes through the same
  verifier, and a theme is kept only if verified quotes from at least two experts
  support it. Its coverage (for example "3/3 experts") is shown.
- `pipeline.py` wires these together for the UI and the tests.

**Layer 2: `app.py`, a Streamlit UI** with three tabs (Per-expert Q&A, Themes &
disagreements, Ask a question) and a sidebar transcript viewer. Clicking
**View in transcript** under any quote switches the sidebar to that expert and
highlights the exact quoted span inside the full turn.

**On proportionality:** this is deliberately a single process with plain modules,
no services, no database and no vector store. Three 7-minute transcripts don't need
more, and building more would spend the time budget on plumbing instead of on
grounding. The seams that would matter at scale (retrieval, the model wrapper,
map-reduce themes) are real and working now, so they can grow without a rewrite.

---

## Model choice

The default is **Claude Sonnet 5** (`claude-sonnet-5`), called through `core/llm.py`
and swappable with one environment variable.

- **The work is extraction, not open-ended reasoning.** It is short context, a fixed
  schema, and "find the sentence that says X and copy it exactly." What matters is
  instruction-following and a low tendency to paraphrase or fill gaps. A mid-sized
  current model does this well at a fraction of the price of the largest one.
- **Cost-aware.** A full cold run (18 answers, themes and a few questions) is about
  25 calls of a few thousand tokens each, well under a dollar on Sonnet 5.
- **The verifier limits the downside.** A weaker model can cost retries, but it can't
  put a bad quote on screen. The test suite measures this directly: in the last run
  the verifier rejected 1 claim on the first pass, the retry fixed it, and nothing
  was dropped.
- **Upgrade path.** If accuracy testing shows more first-pass rejections or weaker
  synthesis, set `CLAUDE_MODEL=claude-opus-5`. Nothing else changes.

---

## How citations and timestamps work (plain-language version)

1. **The transcript is split into speaker turns.** Each turn keeps its exact words
   and the timestamp printed above it in the file, for example Anna Keller at 06:05.
2. **The model never writes a timestamp.** It can only say "this claim is supported
   by turn `DE@06:05`, and here is the sentence from it." The app picks the turn ids
   and the model can only choose from the list it was shown.
3. **The app checks every quote before showing it.** It compares the quote to that
   turn's actual text, ignoring only upper/lower case, spacing and punctuation. There
   is no "close enough": a changed word, a changed number, or two sentences stitched
   together with "..." all fail.
4. **What you see is the transcript's own text,** cut from the source file rather
   than copied from the model's answer, and the timestamp is the one in the file.
5. **If the model was one turn off** (the quote really is in the next turn), the
   citation is moved to the correct turn and marked "timestamp corrected". It is
   never shown at the wrong time.
6. **Every quote has a "View in transcript" link** that opens the full conversation
   with the quote highlighted, so a reviewer can check it with one click.

---

## How hallucination risk is reduced

The main defence is **verification in code**, which does not depend on the model
following instructions. The prompt is a second layer.

1. **Retrieval first.** The model sees only the relevant turns. If retrieval finds
   nothing (for example, "What's the weather in Paris?"), the app answers
   "Not discussed in the transcripts" without calling the model.
2. **A constrained prompt.** Answer only from the excerpts, quote character for
   character, keep the expert's own hedging ("probably", "in some areas"), use the
   quote's own figures, and set `answerable: false` rather than guess.
3. **A constrained schema.** Structured JSON, citation ids limited to the retrieved
   turns, and one citation list per claim.
4. **Quote verification** (`verify.py`). Every quote in every tab, including
   free-form Q&A and themes, is checked as described above. Interviewer lines, quotes
   under 3 words, ellipsis-spliced quotes, and quotes from the wrong expert (for
   example a German quote on a UK position) are all rejected.
5. **A numbers check.** Any number in a claim must appear in that claim's own quotes,
   or in the question itself (such as "3–5 years"). A claim of "25% growth" backed by
   a "15 to 20 percent" quote is rejected, even though the quote is real.
6. **Retry with the real text, then drop.** Failures are logged, then sent back to
   the model once together with the exact source text. Anything that still fails is
   dropped, and so is any claim left without a verified quote. If an answer loses all
   its claims, the UI says "No verified supporting quote found" instead of showing it.
7. **Visible honesty.** Each answer shows a verified-citation badge. Rejections appear
   in a "Verification log" expander. Partial answers state what was not covered.

### Verification suite

`test_citations.py` runs everything the app shows: 18 guide answers, themes and
disagreements, 6 free-form questions, and 3 out-of-scope questions (Japan, a share
price, vendor preference) that must come back as "not discussed". It then audits
every displayed citation with a **separate implementation** that re-reads the raw
`.txt` files, so a bug in `verify.py` cannot hide itself. It exits non-zero if
anything fails.

Latest result (claude-sonnet-5, run 2026-09-21):

```
Citations audited against raw transcript files: 128
Quotes rejected by verify.py on first pass (then retried): 1
Claims/quotes dropped after retry (never shown to user):  0
PASS: 128/128 displayed citations verified verbatim at their timestamps; 18/18 guide
answers, themes, disagreements, 6 free-form and 3 out-of-scope checks OK.
```

*How that run was executed:* no API key was available on the build machine, so
`core.llm.call_json` was routed through the Claude Code CLI (`claude -p --json-schema`)
using the same model and the same prompts and schemas. It is the same pipeline with
a different transport. Running `python test_citations.py` with an API key reproduces
the check through the Anthropic SDK path.

`tests/test_offline.py` (32 tests, no network) covers the parser, retrieval, and
verifier edge cases. It also feeds the generation loop a scripted fake model that
returns paraphrased quotes, invented numbers and fabricated citations, and checks
that the retry-then-drop behaviour handles each one.

---

## Scaling from 3 transcripts to 30+

**Why putting everything in the prompt stops working.** Three transcripts are about
2,000 tokens, so putting them all in one prompt would work today. At 30+ transcripts
(and real calls run 45–60 minutes, about 10k tokens each), the whole corpus reaches
hundreds of thousands of tokens. That is expensive per question and slow. Quote
accuracy also drops as the model searches a huge context, and the list of citable
ids stops being a meaningful constraint. At that point the `retrieval.py` step
becomes load-bearing rather than optional. Each question sees only the turns that
matter, and the rest of the pipeline (schema, verification, UI) doesn't change.

**Pre-compute the index once.** Today the BM25 index is built once per corpus and
reused for every query. At scale I would add dense embeddings: embed each turn once
at ingest time, store the vectors with the transcript's content hash, and re-embed
only new or changed transcripts. A hybrid of BM25 and embeddings works well here,
because keywords catch exact figures and names while embeddings catch paraphrase.

**A real vector store when memory runs out.** In-memory similarity search is fine
up to tens of thousands of turns. Beyond that, use a local FAISS or Chroma index
(the `.gitignore` already excludes the index files), with metadata filters on
expert, market, role and date. "What did German procurement people say" then
becomes a filter plus a search, not a scan. Only `Retriever.search` changes.

**Themes need map-reduce, not one giant prompt.** Thirty full transcripts don't fit
in one call, and a model reading all of them would produce vague themes. The
current `themes.py` already has the right shape:

1. **Map (per transcript, run in parallel):** answer the guide questions for each
   expert, verified. That gives a compact, cited summary of each call, and it is
   cached, so adding transcript 31 does not recompute the first 30.
2. **Reduce (hierarchical):** group summaries by market or role, find themes within
   each group, then merge themes across groups. Each reduce call sees summaries plus
   the specific quoted turns, never whole transcripts.
3. **Verify every level** with the same verifier, so a theme at the top still points
   to an exact line in a specific call.

**Cost and latency.** At 3 transcripts a cold run is about 25 calls and takes about
a minute, well under a dollar. At 30 transcripts, the map step is 30 × 6 = 180 calls,
but they are independent: they parallelise (bounded by rate limits) and can use the
**Message Batches API** at half price, since pre-computing guide answers is not
interactive. Each call stays small because of retrieval, so cost grows roughly
linearly with the number of transcripts instead of per question × corpus size.
Free-form questions stay near constant cost, since retrieval caps the context at
top-k turns. Prompt caching on the fixed system prompt and schema lowers input cost
further. The main new latency is the reduce step, which runs once when transcripts
change, not on every page load.

---

## Known limitations

- **The verifier checks quotes and numbers, not every nuance of a paraphrase.** Each
  claim sentence is written by the model. Its quote is guaranteed real and its
  numbers must match the quote, but a claim can still slightly over- or
  under-state the quote's tone (for example "roughly 15 to 20 percent" for the
  expert's "maybe 15 to 20 percent"). That is why the quote is always shown next
  to the claim. An LLM-as-judge entailment check per claim would be the next step.
- **Retrieval is lexical.** BM25 plus a hand-written synonym list works for this
  vocabulary but will miss paraphrases a reader would catch. Embeddings are the fix
  (see Scaling).
- **"Not discussed" depends partly on the model.** If retrieval finds loosely related
  turns, the model decides whether they actually answer the question. It declined
  correctly in every test, but that is a judgement, not a guarantee. The verifier
  guarantees that nothing unquoted is shown, not that every declinable question gets
  declined.
- **The parser expects this transcript format:** a header, then alternating `MM:SS`
  and `Speaker: text` lines. Real transcripts, with speaker diarisation errors,
  multi-paragraph turns and cross-talk, would need a more tolerant parser.
- **Timestamps are per turn, not per sentence.** A quote from the end of a long turn
  still shows the turn's start time, which is the only time the source file gives.
- **Responses are cached by exact request.** Changing the model, the prompt or a
  transcript invalidates the cache automatically. Otherwise the app replays earlier
  answers until `.cache/` is cleared or "Ignore response cache" is toggled on.
- **The retry is a single round.** A claim that fails twice is dropped, not repaired.
  This is deliberate: a missing claim is preferable to one that has been argued into
  shape.
