import streamlit as st
from chat_interface import display_chat_interface
from sidebar import display_sidebar

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄")
st.title("📄 RAG Document Assistant")
st.caption(
    "Ask questions about your uploaded PDF, DOCX, or HTML documents. "
    "Answers are grounded in your documents, with cited sources."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

display_sidebar()
display_chat_interface()
