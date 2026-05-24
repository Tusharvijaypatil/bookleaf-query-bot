"""
Optional Streamlit chat UI — same `responder.answer` pipeline as the CLI.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from src import config, responder


st.set_page_config(page_title="BookLeaf Bot", page_icon=":books:")

config.require_env(strict=True)

st.title(":books: BookLeaf Customer Query Bot")
st.caption("Ask about your book status, royalties, dashboard, add-ons, sales, or shipping.")

with st.sidebar:
    st.markdown("### Settings")
    channel = st.selectbox("Channel (for logging only)", ["cli", "email", "whatsapp", "instagram"], index=0)
    debug = st.toggle("Show debug info", value=False)
    if st.button("Clear chat"):
        st.session_state.pop("history", None)
        st.rerun()

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"|"assistant", "content": str, "meta": dict}

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if debug and msg["role"] == "assistant" and msg.get("meta"):
            m = msg["meta"]
            st.caption(
                f"intent={m['intent']}  •  source={m['source']}  •  "
                f"confidence={m['confidence']:.2f}  •  escalated={m['escalated']}"
            )

prompt = st.chat_input("Type your question…")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt, "meta": {}})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Looking things up…"):
            try:
                result = responder.answer(prompt, channel=channel)
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.stop()
        st.markdown(result.response)
        if debug:
            st.caption(
                f"intent={result.intent}  •  source={result.source}  •  "
                f"confidence={result.confidence:.2f}  •  escalated={result.escalated}"
            )
    st.session_state.history.append({
        "role": "assistant",
        "content": result.response,
        "meta": {
            "intent": result.intent,
            "source": result.source,
            "confidence": result.confidence,
            "escalated": result.escalated,
        },
    })
