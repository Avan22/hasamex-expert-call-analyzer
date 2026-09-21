# Demo talking points (in order)

## 1. What I built (about 1 min, app on screen)
- An expert-call analyzer for 3 robotic-surgery interviews (France, Germany, UK) and the 6-question guide.
- Three tabs: Per-expert Q&A (18 answers), Themes & disagreements, Ask a question, plus a transcript viewer in the sidebar.
- Every claim on screen has at least one verbatim quote with the expert's name and `MM:SS` timestamp.
- Live: click **View in transcript** under Anna Keller's 00:16 quote. The sidebar switches to her call and highlights the exact span.

## 2. Architecture (about 1.5 min, README diagram)
- Two layers: `core/` is a plain Python pipeline that works without the UI. `app.py` is a thin Streamlit layer on top.
- The flow is parse → retrieve (BM25) → generate (`llm.py`: Gemini or Claude, JSON schema) → verify → UI. Themes use map-reduce over the verified per-expert answers.
- The parser stores each turn's text verbatim with a stable id like `DE@06:05`. The interviewer's lines are context and can never be cited.
- Retrieval is real, not a shortcut. It indexes each expert turn together with its question, retrieves per expert for cross-expert questions, and skips the model entirely when nothing matches.
- Kept proportional on purpose: no services, no database, no vector store for 3 files.

## 3. Model choice (about 1 min, say it plainly)
- **Verified default: Gemini `gemini-3.5-flash-lite`.** It is the only provider run end to end against its live API.
- **Claude is fully implemented and unit-tested, but has never been called through the live Anthropic API**, because that account has no billing yet. So I don't present it as verified. The verification is one command: `MODEL_PROVIDER=anthropic python test_citations.py --fresh`.
- Why Claude is still the intended production model (reasoning, not a measured result): instruction-following on exact extraction, and richer synthesis. Flash-Lite gives 2 themes per run.
- Structured outputs (a JSON schema) on both providers keep answers and citations as separate fields, so there is no regex parsing.
- The verifier caps the downside of any model: it can cost a retry, but it can't show a bad quote.

## 4. Citations and timestamps (about 1.5 min)
- The model never writes a timestamp. It picks a `segment_id` from an enum of the retrieved turns, and the timestamp comes from the file.
- The verifier compares the quote to the turn text, ignoring only case, whitespace and punctuation. It accepts no fuzzy matches and no quotes spliced with an ellipsis.
- The UI shows the source's own characters, cut from the transcript, not the model's copy.
- If the model is one turn off, the citation moves to the true turn and is labelled "timestamp corrected". It is never shown at the wrong time.

## 5. Reducing hallucination (about 2 min, the core of the demo)
- The prompt says: excerpts only, quote exactly, keep the expert's hedging, and set `answerable: false` rather than guess.
- The **verification in code** is what actually enforces this. It applies to every tab, including free-form Q&A and themes.
- Numbers check: every figure in a claim must be in that claim's own quotes. For example, "25%" backed by a "15 to 20 percent" quote is rejected.
- On failure, the app logs it, sends the real turn text back, retries once, then drops whatever still fails. A claim with no verified quote is never shown.
- Live: ask "What is Intuitive Surgical's share price?" and it answers **Not discussed in the transcripts**. Then ask a timeline question and point out the answers of 6–12, 9–18 and 6–9 months, each quoted.
- Show `test_citations.py --fresh`: 18 answers, themes, 6 free-form and 3 out-of-scope questions. It audits every citation against the raw `.txt` files with a *separate* implementation.
- Real API results, from two independent fresh runs on live Gemini with no cache:
  - Run 1: 28 live calls, **83/83** citations verified, 0 first-pass rejections, 0 drops.
  - Run 2 (new key): 29 live calls, **74/74** verified, 2 first-pass rejections fixed by the retry, 0 drops.
  - Both runs declined all 3 out-of-scope questions. Logs: `docs/verification_run_gemini_1.txt`, `_2.txt`.
- The count changes run to run because the model picks how many quotes to cite. The 100% pass rate is the invariant.
- Run 2's rejections make a good example: a 2-word quote ("Very important") was replaced with the full sentence, and a claim mentioning "two systems" was rejected because its quote stopped before that phrase.
- The rate-limit fix, found in the live run: blind SDK retries from 3 threads exhausted a free-tier daily quota (5 per minute and 20 per day per model). The fix is a shared rate limiter that honours the server's `RetryInfo` and fails fast on a daily quota.

## 6. Scaling to 30+ transcripts (about 1.5 min)
- Putting every transcript in the prompt breaks down at hundreds of thousands of tokens: cost, latency and quote accuracy all suffer. Retrieval becomes load-bearing, and only `Retriever.search` changes.
- Embed each turn once at ingest, keyed by content hash, and use hybrid BM25 plus embeddings. Move to FAISS or Chroma with metadata filters (market, role) once it outgrows memory.
- Themes: per-transcript map (cached and parallel), then a hierarchical reduce by market or role, then cross-group. Every level is verified.
- Cost: the map step is about 180 small independent calls, so use the Batches API at half price, plus prompt caching on the fixed system prompt. Free-form questions stay near constant cost because retrieval caps the context.

## 7. Honest limitations (about 45 s)
- **Claude is not verified live.** Everything shown was verified on Gemini Flash-Lite. The Claude path is implemented and unit-tested, and pending one command once billing is set up.
- **Traceability, not depth.** The verifier guarantees quotes and numbers are real, not that a claim captures every nuance. For example, an earlier run wrote "roughly 15 to 20 percent" where Dr. Martin said "*maybe* 15 to 20 percent". That's why the quote always sits next to the claim. The next step is a per-claim entailment check with an LLM judge.
- Flash-Lite's cross-expert synthesis is thin (2 themes). A stronger model should do better, which is exactly what the Claude run would show.
