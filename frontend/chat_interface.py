"""Chat UI: WhatsApp-style bubbles (no avatars), plus a Stop button that lets
the user cancel while waiting for an answer.

HOW THE STOP BUTTON WORKS (worth reading before you touch this file):
Streamlit runs your whole script top-to-bottom on every interaction, so a
plain blocking `requests.post(...)` call can't be interrupted by a button
click that happens *during* that call - the click isn't even processed
until the blocking call returns. To make Stop actually work, the network
request runs in a background thread, and a `st.fragment(run_every=...)`
polls that thread every 0.3s from the main thread - each poll is fast, so
the Stop button stays clickable the whole time the app is waiting.

One honest limitation: clicking Stop makes the UI stop *waiting* for the
answer and discards it when it arrives - it does not abort the request
already in flight to the backend/cloud LLM (this app's /query endpoint
isn't streaming, so there's no partial result to keep either way).
"""

import html as html_lib
import threading

import markdown as md_lib
import requests
import streamlit as st
from api_utils import BACKEND_URL, REQUEST_TIMEOUT

_CHAT_CSS = """
<style>
.chat-row {
    display: flex;
    margin: 0.35rem 0;
}
.chat-row.user {
    justify-content: flex-end;
}
.chat-row.assistant {
    justify-content: flex-start;
}
.chat-bubble {
    max-width: 75%;
    padding: 0.55rem 0.9rem;
    border-radius: 1rem;
    line-height: 1.45;
    font-size: 0.95rem;
    word-wrap: break-word;
}
/* Two clearly different, non-white colors: green for you, blue for the AI */
.chat-bubble.user {
    background-color: #D9FDD3;
    color: #111111;
    border-bottom-right-radius: 0.25rem;
}
.chat-bubble.assistant {
    background-color: #CFE3FF;
    color: #111111;
    border-bottom-left-radius: 0.25rem;
}
.chat-sources {
    max-width: 75%;
    font-size: 0.72rem;
    color: #888888;
    margin: 0.1rem 0 0.4rem 0.2rem;
}
/* The assistant bubble now holds real HTML (converted from the LLM's
   Markdown answer), so give its inner elements sane spacing instead of
   the default browser margins, which look too "spaced out" in a chat
   bubble. */
.chat-bubble p {
    margin: 0 0 0.5rem 0;
}
.chat-bubble p:last-child {
    margin-bottom: 0;
}
.chat-bubble ul, .chat-bubble ol {
    margin: 0.2rem 0 0.5rem 1.1rem;
    padding: 0;
}
.chat-bubble li {
    margin-bottom: 0.15rem;
}
.chat-bubble strong {
    font-weight: 700;
}
.chat-bubble code {
    background-color: rgba(0, 0, 0, 0.08);
    padding: 0.05rem 0.3rem;
    border-radius: 0.25rem;
    font-size: 0.85em;
}
</style>
"""


def _escape(text: str) -> str:
    """Escape plain text before dropping it into raw HTML, and turn
    newlines into <br> so multi-line text still looks right. Used for the
    user's own typed question, which is plain text, not Markdown."""
    return html_lib.escape(text).replace("\n", "<br>")


def _render_markdown(text: str) -> str:
    """Convert the LLM's Markdown answer (**bold**, "- " bullet lists, etc.)
    into real HTML so it renders as bold text / lists instead of showing the
    literal '**' and '-' characters inside the chat bubble."""
    return md_lib.markdown(text, extensions=["nl2br", "sane_lists"])


def _format_sources(sources: list) -> str:
    if not sources:
        return ""
    parts = []
    for source in sources:
        filename = source.get("filename", "unknown")
        page = source.get("page")
        parts.append(f"{filename} (p. {page})" if page is not None else filename)
    return " • ".join(parts)


def _render_bubble(role: str, text: str, sources_text: str = ""):
    # Assistant answers are Markdown from the LLM -> render as real HTML.
    # The user's own question is plain text -> escape + <br> only.
    content_html = _render_markdown(text) if role == "assistant" else _escape(text)
    st.markdown(
        f'<div class="chat-row {role}"><div class="chat-bubble {role}">{content_html}</div></div>',
        unsafe_allow_html=True,
    )
    if sources_text:
        st.markdown(
            f'<div class="chat-row {role}"><div class="chat-sources">'
            f"Sources: {html_lib.escape(sources_text)}</div></div>",
            unsafe_allow_html=True,
        )


def _fetch_answer_in_background(question: str, session_id, result_box: dict):
    """Runs on a background thread - must never call any st.* function
    (Streamlit widgets can only be touched from the main script thread)."""
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id

    response = None
    try:
        response = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result_box["data"] = response.json()
    except requests.exceptions.ConnectionError:
        result_box["error"] = "Could not reach the backend. Is the FastAPI server running?"
    except requests.exceptions.Timeout:
        result_box["error"] = "The request took too long and timed out."
    except requests.exceptions.HTTPError:
        try:
            result_box["error"] = response.json().get("detail", response.text)
        except Exception:
            result_box["error"] = "The backend returned an error."
    except Exception as exc:  # last-resort guard
        result_box["error"] = f"An unexpected error occurred: {exc}"
    finally:
        result_box["done"] = True


def display_chat_interface():
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    for message in st.session_state.messages:
        _render_bubble(message["role"], message["content"], message.get("sources_text", ""))

    pending = st.session_state.get("pending_request")

    # run_every is computed fresh each full rerun: "0.3s" while waiting for
    # an answer, None (no auto-polling) the rest of the time.
    @st.fragment(run_every="0.3s" if pending else None)
    def _poll_pending_request():
        current = st.session_state.get("pending_request")
        if current is None:
            return

        status_col, stop_col = st.columns([6, 1])
        with status_col:
            st.caption("Thinking...")
        with stop_col:
            if st.button("⏹ Stop", key="stop_request_button", use_container_width=True):
                st.session_state.pending_request = None
                st.rerun()
                return

        result_box = current["result_box"]
        if result_box.get("done"):
            data = result_box.get("data")
            error = result_box.get("error")
            st.session_state.pending_request = None

            if error:
                st.error(error)
            elif data:
                st.session_state.session_id = data.get("session_id")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data.get("answer", ""),
                        "sources_text": _format_sources(data.get("sources", [])),
                    }
                )
            st.rerun()

    _poll_pending_request()

    if pending is not None:
        # Keep input disabled while a request is in flight or being stopped,
        # so the chat state stays simple and predictable.
        st.chat_input("Ask a question about your uploaded documents...", disabled=True)
        return

    prompt = st.chat_input("Ask a question about your uploaded documents...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    _render_bubble("user", prompt)

    result_box = {"done": False, "data": None, "error": None}
    thread = threading.Thread(
        target=_fetch_answer_in_background,
        args=(prompt, st.session_state.session_id, result_box),
        daemon=True,
    )
    thread.start()
    st.session_state.pending_request = {"thread": thread, "result_box": result_box}
    st.rerun()