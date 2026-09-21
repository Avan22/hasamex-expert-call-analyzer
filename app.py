"""Streamlit UI for the expert-call analyzer.

    streamlit run app.py

The UI only renders what the core pipeline returns. Every quote it shows has
already passed core/verify.py, and every quote has a "View in transcript"
button that highlights the source turn in the sidebar transcript viewer.
"""

from __future__ import annotations

import html
import logging
import os

import streamlit as st

from core import llm
from core.generation import Answer
from core.parser import TranscriptParseError, parse_transcript_text
from core.pipeline import Pipeline

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")

st.set_page_config(page_title="Expert Call Analyzer", page_icon="🎙️", layout="wide")

st.markdown(
    """
<style>
.quote {border-left: 3px solid #4a7bd0; padding: 0.35rem 0.75rem; margin: 0.25rem 0 0.1rem 0;
        background: rgba(74,123,208,0.07); border-radius: 0 6px 6px 0; font-style: italic;}
.cite-meta {font-size: 0.8rem; opacity: 0.75; margin-bottom: 0.35rem;}
.turn {padding: 0.4rem 0.6rem; border-radius: 6px; margin-bottom: 0.3rem; font-size: 0.88rem;}
.turn.focus {background: rgba(255, 196, 0, 0.18); border: 1px solid rgba(255,196,0,0.6);}
.turn .ts {font-family: monospace; opacity: 0.7; margin-right: 0.4rem;}
.turn.interviewer {opacity: 0.7;}
mark {background: rgba(255,196,0,0.55); padding: 0 2px; border-radius: 3px;}
.badge {display:inline-block; font-size:0.75rem; padding: 1px 8px; border-radius: 10px;
        background: rgba(46,160,67,0.15); color: #2ea043; margin-right: 4px;}
.badge.warn {background: rgba(210,153,34,0.15); color: #d29922;}
</style>
""",
    unsafe_allow_html=True,
)


# --- pipeline -------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_pipeline(uploaded: tuple[tuple[str, str], ...] | None) -> Pipeline:
    if uploaded:
        transcripts = [parse_transcript_text(text, name) for name, text in uploaded]
        return Pipeline(transcripts=transcripts)
    return Pipeline()


@st.cache_data(show_spinner=False)
def guide_answers(_pipe: Pipeline, key: str, use_cache: bool) -> list[Answer]:
    return _pipe.all_guide_answers(use_cache=use_cache)


@st.cache_data(show_spinner=False)
def cross_expert(_pipe: Pipeline, key: str, use_cache: bool):
    return _pipe.cross_expert(guide_answers(_pipe, key, use_cache), use_cache=use_cache)


def set_focus(segment_id: str, quote: str) -> None:
    st.session_state["focus"] = (segment_id, quote)


# --- rendering helpers ----------------------------------------------------------

def render_citation(c: dict, key: str) -> None:
    st.markdown(f'<div class="quote">“{html.escape(c["quote"])}”</div>', unsafe_allow_html=True)
    corrected = " · timestamp corrected by verifier" if c["status"] == "corrected" else ""
    st.markdown(
        f'<div class="cite-meta">{html.escape(c["expert_name"])} · <b>{c["timestamp"]}</b> · '
        f'✓ verified verbatim{corrected}</div>',
        unsafe_allow_html=True,
    )
    st.button("↗ View in transcript", key=key, on_click=set_focus,
              args=(c["segment_id"], c["quote"]), type="tertiary")


def render_answer(a: Answer, key: str, show_experts: bool = False) -> None:
    if a.status == "error":
        st.error(f"Model call failed: {a.error}")
        return
    if a.status == "not_discussed":
        st.info(f"**Not discussed in the transcripts.** {a.not_covered if a.not_covered != 'Not discussed in the transcripts.' else ''}")
        return
    if a.status == "no_verified_support":
        st.warning("**No verified supporting quote found.** The model proposed an answer, but none "
                   "of its quotes matched the transcript, so it is not shown.")
        return
    n = len(a.citations)
    st.markdown(f'<span class="badge">✓ {n} verified citation{"s" if n != 1 else ""}</span>',
                unsafe_allow_html=True)
    for i, p in enumerate(a.points):
        st.markdown(f"**{html.escape(p['claim'])}**")
        for j, c in enumerate(p["citations"]):
            render_citation(c, f"{key}-{i}-{j}")
    if a.not_covered:
        st.caption(f"Not covered: {a.not_covered}")
    if a.failures_first_pass or a.failures_final:
        with st.expander("Verification log"):
            st.write(f"Model attempts: {a.attempts}")
            for f in a.failures_first_pass:
                st.write(f"• First pass rejected [{f['claimed_segment_id']}]: {f['reason']} — “{f['quote']}”")
            for f in a.failures_final:
                st.write(f"• **Dropped after retry** [{f['claimed_segment_id']}]: {f['reason']} — “{f['quote']}”")


def render_transcript(pipe: Pipeline, expert_id: str) -> None:
    focus_id, focus_quote = st.session_state.get("focus", (None, None))
    t = pipe.experts[expert_id]
    st.markdown(f"**{t.expert_name}**  \n{t.role} · {t.market}")
    for s in t.segments:
        text = html.escape(s.text)
        if s.segment_id == focus_id and focus_quote:
            q = html.escape(focus_quote)
            text = text.replace(q, f"<mark>{q}</mark>", 1)
        cls = "turn" + (" focus" if s.segment_id == focus_id else "") + ("" if s.is_expert else " interviewer")
        st.markdown(
            f'<div class="{cls}"><span class="ts">{s.timestamp}</span>'
            f"<b>{html.escape(s.speaker)}:</b> {text}</div>",
            unsafe_allow_html=True,
        )


# --- sidebar --------------------------------------------------------------------

with st.sidebar:
    st.header("Source transcripts")
    uploads = st.file_uploader(
        "Optional: upload transcript .txt files (defaults to the 3 case transcripts)",
        type=["txt"], accept_multiple_files=True,
    )
    uploaded = None
    if uploads:
        try:
            uploaded = tuple(sorted((u.name, u.getvalue().decode("utf-8-sig")) for u in uploads))
            for name, text in uploaded:
                parse_transcript_text(text, name)
        except (TranscriptParseError, UnicodeDecodeError) as e:
            st.error(f"Could not parse upload: {e}")
            uploaded = None

    pipe = get_pipeline(uploaded)
    corpus_key = "|".join(s.segment_id + s.text for t in pipe.transcripts for s in t.segments)

    focus = st.session_state.get("focus")
    default_idx = pipe.expert_ids.index(focus[0].split("@")[0]) if focus and focus[0].split("@")[0] in pipe.expert_ids else 0
    expert_choice = st.selectbox(
        "Transcript", pipe.expert_ids, index=default_idx,
        format_func=lambda e: pipe.experts[e].label(),
    )
    render_transcript(pipe, expert_choice)
    st.divider()
    st.caption(f"Model: `{llm.model_name()}` (set `CLAUDE_MODEL` to change)")
    use_cache = not st.toggle("Ignore response cache (fresh model calls)", value=False)


# --- main -----------------------------------------------------------------------

st.title("Expert Call Analyzer")
st.caption(
    "European robotic surgery market · "
    + " · ".join(t.label() for t in pipe.transcripts)
    + ". Every quote below has been checked verbatim against the transcript; click "
    "**View in transcript** to see it highlighted in the sidebar."
)

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("`ANTHROPIC_API_KEY` is not set. Copy `.env.example` to `.env`, add your key, and restart.")
    st.stop()

tab_qa, tab_themes, tab_ask = st.tabs(
    ["Per-expert Q&A", "Themes & disagreements", "Ask a question"]
)

with tab_qa:
    with st.spinner("Answering 6 guide questions for 3 experts (first run only, then cached)…"):
        answers = guide_answers(pipe, corpus_key, use_cache)
    by_key = {(a.question, a.scope[0]): a for a in answers}
    for qi, q in enumerate(pipe.questions, 1):
        st.subheader(f"Q{qi}. {q}")
        cols = st.columns(len(pipe.expert_ids))
        for col, eid in zip(cols, pipe.expert_ids):
            with col:
                t = pipe.experts[eid]
                st.markdown(f"##### {t.expert_name}")
                st.caption(f"{t.role} · {t.market}")
                render_answer(by_key[(q, eid)], f"qa-{qi}-{eid}")
        st.divider()

with tab_themes:
    with st.spinner("Comparing experts…"):
        analysis = cross_expert(pipe, corpus_key, use_cache)
    if analysis.error:
        st.error(analysis.error)
    st.subheader("Common themes")
    if not analysis.themes:
        st.info("No common theme could be backed by verified quotes from at least two experts.")
    for i, th in enumerate(analysis.themes):
        cov_cls = "badge" if th["coverage"].split("/")[0] == th["coverage"].split("/")[1] else "badge warn"
        with st.container(border=True):
            st.markdown(f"#### {html.escape(th['title'])}")
            st.markdown(f'<span class="{cov_cls}">Shared by {th["coverage"]} experts</span>',
                        unsafe_allow_html=True)
            st.write(th["summary"])
            for j, c in enumerate(th["citations"]):
                render_citation(c, f"theme-{i}-{j}")
    st.subheader("Disagreements and divergences")
    if not analysis.disagreements:
        st.info("No disagreement could be backed by verified quotes on both sides.")
    for i, d in enumerate(analysis.disagreements):
        with st.container(border=True):
            st.markdown(f"#### {html.escape(d['topic'])}")
            st.write(d["summary"])
            cols = st.columns(len(d["positions"]))
            for col, (k, p) in zip(cols, enumerate(d["positions"])):
                with col:
                    st.markdown(f"**{pipe.experts[p['expert_id']].expert_name}**")
                    st.write(p["position"])
                    for j, c in enumerate(p["citations"]):
                        render_citation(c, f"dis-{i}-{k}-{j}")
    if analysis.dropped or analysis.failures_first_pass:
        with st.expander("Verification log"):
            for f in analysis.failures_first_pass:
                st.write(f"• First pass rejected [{f['claimed_segment_id']}]: {f['reason']} — “{f['quote']}”")
            for f in analysis.failures_final:
                st.write(f"• **Dropped after retry**: {f['reason']} — “{f['quote']}”")
            for msg in analysis.dropped:
                st.write(f"• {msg}")

with tab_ask:
    st.caption(
        "Ask anything about the three calls. Answers use only the transcripts; if they do not "
        "cover your question, the app says so instead of guessing."
    )
    history = st.session_state.setdefault("history", [])
    for n, (q, a) in enumerate(history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            render_answer(a, f"ask-{n}")
    question = st.chat_input("e.g. How long does a purchase decision take in each market?")
    if question:
        with st.spinner("Retrieving, answering and verifying…"):
            ans = pipe.ask(question, use_cache=use_cache)
        history.append((question, ans))
        st.rerun()
