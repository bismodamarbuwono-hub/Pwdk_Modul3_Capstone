"""
app.py - Streamlit UI for the Resume Screening Assistant.

Run locally:   streamlit run app.py
Secrets needed (in .env locally, or in Streamlit Cloud > App settings > Secrets):
    OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
"""
import os
import uuid

import streamlit as st

from agents import MAX_HISTORY, ask, get_secret

st.set_page_config(page_title="Resume Screening Assistant", page_icon="📄")


# ---------------------------------------------------------------------------
# Langfuse (tracing). The app still works if Langfuse is not configured.
# ---------------------------------------------------------------------------
def _first_secret(*names):
    """Return the first secret that exists (Langfuse uses two names for the host)."""
    for name in names:
        try:
            return get_secret(name)
        except Exception:
            continue
    return None


def make_langfuse_handler():
    public = _first_secret("LANGFUSE_PUBLIC_KEY")
    secret = _first_secret("LANGFUSE_SECRET_KEY")
    host = _first_secret("LANGFUSE_BASE_URL", "LANGFUSE_HOST")
    if not (public and secret and host):
        return None
    try:
        os.environ["LANGFUSE_PUBLIC_KEY"] = public
        os.environ["LANGFUSE_SECRET_KEY"] = secret
        os.environ["LANGFUSE_BASE_URL"] = host
        os.environ["LANGFUSE_HOST"] = host
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception:
        return None


def flush_langfuse():
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session state: one chat "room" = one list of messages + one Langfuse session id
# ---------------------------------------------------------------------------
def new_chat():
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.total_in = 0
    st.session_state.total_out = 0
    st.session_state.total_idr = 0.0


if "messages" not in st.session_state:
    new_chat()


def render_details(meta):
    """Proof that the answer came from the documents, plus usage info."""
    with st.expander("Retrieved resumes (RAG sources)"):
        if meta["sources"]:
            for s in meta["sources"]:
                st.caption(f"tool: {s['tool']}")
                st.code(s["output"], language=None)
        else:
            st.write("No resumes were retrieved for this answer.")
    with st.expander("Agent and tool calls"):
        st.json(meta["tool_calls"])
    st.caption(
        f"Tokens: {meta['input_tokens']} in / {meta['output_tokens']} out "
        f"| Cost: Rp {meta['price_idr']:.2f}"
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    k = st.slider("Resume chunks retrieved per search", 3, 10, 5)
    if st.button("New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.divider()
    st.subheader("LLM usage (this chat)")
    st.write(f"Input tokens: **{st.session_state.total_in}**")
    st.write(f"Output tokens: **{st.session_state.total_out}**")
    st.write(f"Estimated cost: **Rp {st.session_state.total_idr:.2f}**")

    st.divider()
    st.caption(f"Chat session id: {st.session_state.session_id[:8]}")

# ---------------------------------------------------------------------------
# Chat window: the whole conversation stays visible here
# ---------------------------------------------------------------------------
st.title("Resume Screening Assistant")
st.caption("Ask about the resume database: find candidates, summarize a resume, or compare candidates.")

# Drop-down with the whole conversation, shown as chat messages
with st.expander(f"Chat history ({len(st.session_state.messages)} messages)"):
    if st.session_state.messages:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
        st.caption(f"The agent sees the last {MAX_HISTORY} of these messages on every question.")
    else:
        st.write("No messages yet.")

if not st.session_state.messages:
    st.info(
        "Try: *Find candidates with Python and SQL experience*, then follow up with "
        "*Summarize the first one* or *Compare the first two*."
    )

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("meta"):
            render_details(m["meta"])

if prompt := st.chat_input("Ask about the resumes"):
    # history = everything BEFORE this question
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            handler = make_langfuse_handler()
            try:
                response = ask(
                    prompt,
                    history,
                    k=k,
                    extra_callbacks=[handler] if handler else None,
                    session_id=st.session_state.session_id,
                )
            except Exception as e:
                response = None
                st.error(f"Something went wrong: {e}")
            flush_langfuse()

        if response:
            st.markdown(response["answer"])
            render_details(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"], "meta": response}
            )
            st.session_state.total_in += response["input_tokens"]
            st.session_state.total_out += response["output_tokens"]
            st.session_state.total_idr += response["price_idr"]
            st.rerun()  # refresh the sidebar totals