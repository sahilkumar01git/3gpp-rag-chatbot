"""Streamlit entry point for the 3GPP RAG chatbot."""

from __future__ import annotations

import os
import uuid

import streamlit as st

if "GROQ_API_KEY" not in os.environ:
    try:
        groq_api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        groq_api_key = None
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

from app.generation.llm_client import LLMError
from app.pipeline import RagPipeline


st.set_page_config(page_title="3GPP RAG Chatbot", page_icon="3G", layout="wide")


@st.cache_resource(show_spinner="Loading the retrieval models and index...")
def load_pipeline() -> RagPipeline:
    """Load the expensive process-wide pipeline once per Streamlit worker."""
    return RagPipeline()


def reset_chat() -> None:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []


if "session_id" not in st.session_state:
    reset_chat()

st.title("3GPP RAG Chatbot")
st.caption("Ask questions about the 3GPP specifications in the configured corpus.")

with st.sidebar:
    st.subheader("Conversation")
    if st.button("New conversation", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()
    st.caption("The answer is checked against retrieved specification passages before it is shown.")

try:
    pipeline = load_pipeline()
except Exception as exc:
    st.error("The chatbot could not load its index or models.")
    st.exception(exc)
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("citations"):
            with st.expander("Evidence and confidence"):
                for citation in message["citations"]:
                    st.markdown(
                        f"**{citation['citation'] or 'Retrieved passage'}**  \n"
                        f"{citation['claim']}  \n"
                        f"Confidence: `{citation['confidence']:.2f}`"
                    )

question = st.chat_input("Ask about the 3GPP corpus...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the specifications and verifying the answer..."):
            try:
                result = pipeline.answer(question, session_id=st.session_state.session_id)
            except LLMError as exc:
                st.error(str(exc))
                st.stop()

        st.markdown(result.answer)
        st.caption(
            f"Confidence: {result.overall_confidence:.2f} · "
            f"Evidence coverage: {result.coverage_ratio:.2f} · "
            f"Latency: {result.latency_seconds:.2f}s"
        )

        citations = [
            {
                "claim": claim.claim,
                "citation": claim.citation,
                "confidence": claim.confidence,
            }
            for claim in result.claims
            if claim.supported
        ]
        if citations:
            with st.expander("Evidence and confidence"):
                for citation in citations:
                    st.markdown(
                        f"**{citation['citation'] or 'Retrieved passage'}**  \n"
                        f"{citation['claim']}  \n"
                        f"Confidence: `{citation['confidence']:.2f}`"
                    )

    st.session_state.session_id = result.session_id
    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "citations": citations}
    )