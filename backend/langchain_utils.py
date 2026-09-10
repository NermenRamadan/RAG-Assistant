"""The RAG pipeline: retriever -> cloud LLM -> grounded answer + sources.

This replaces the original ChatOllama/llama3.2 setup with a cloud LLM
reached through an OpenAI-compatible endpoint. By default that's Groq
(fast, has a free tier, no local model download needed), but any
OpenAI-compatible provider works - just change LLM_BASE_URL, LLM_MODEL and
LLM_API_KEY in your .env file.

NOTE ON LANGCHAIN VERSION COMPATIBILITY:
The original starter repo built its chain with
`langchain.chains.create_history_aware_retriever` /
`create_retrieval_chain` / `create_stuff_documents_chain`. Those live in
the top-level `langchain` package's `chains` module, which was removed in
LangChain's 1.x rewrite - so that code breaks on any environment that
already has langchain>=1.0 installed.

To stay compatible with BOTH old (0.x) and new (1.x) LangChain installs
without requiring anyone to upgrade/downgrade anything, the RAG pipeline
below is written as three small, explicit steps (reformulate -> retrieve ->
generate) using only `langchain_core` (prompts) and `langchain_openai`
(the chat model) - both of which have kept a stable API across versions.
No `langchain.chains` import anywhere in this file.
"""

import logging
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from chroma_utils import vectorstore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cloud LLM configuration (all from environment variables - never hardcoded)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or "not-set"

if LLM_API_KEY == "not-set":
    logger.warning(
        "LLM_API_KEY is not set. Copy .env.example to .env and add a real API "
        "key, or /query requests will fail with a 502 error."
    )

llm = ChatOpenAI(
    model=DEFAULT_MODEL_NAME,
    api_key=LLM_API_KEY,
    base_url=LLM_BASE_URL,
    temperature=0.2,
    timeout=30,
    # A flaky connection (antivirus/firewall SSL interception, a dropped
    # Wi-Fi packet, etc.) can reset the TCP connection mid-request. The
    # client already retries automatically; raising this from its default
    # of 2 gives a couple of extra attempts before giving up and surfacing
    # a 502 to the user.
    max_retries=4,
)

# ---------------------------------------------------------------------------
# Retriever - reused across every request, built once at import time
# ---------------------------------------------------------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Rewrites a follow-up question ("what about page 2?") into a standalone
# question using the chat history, WITHOUT answering it.
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question that can be "
    "understood without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# The actual answer-generation prompt. This is what makes answers
# "grounded": the model is explicitly told to only use the retrieved
# context and to admit when it doesn't know.
qa_system_prompt = (
    "You are a helpful assistant that answers questions about the user's "
    "uploaded documents.\n\n"
    "Rules you MUST follow:\n"
    "1. Answer using ONLY the information in the context below.\n"
    "2. Do not invent, assume, or add facts that are not present in the context.\n"
    "3. If the context does not contain enough information to answer the "
    'question, respond exactly with: "I could not find this information in '
    'the provided documents."\n'
    "4. Keep the answer concise and directly useful.\n\n"
    "Context:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _contextualize_question(question: str, chat_history: List[Dict[str, str]]) -> str:
    """Rewrite a follow-up question into a standalone one, using chat history.

    Skipped entirely (no extra LLM call, no dependency on history formatting
    quirks) when there is no prior history yet, which is the common case for
    a brand-new session.
    """
    if not chat_history:
        return question

    prompt_value = contextualize_q_prompt.invoke({"chat_history": chat_history, "input": question})
    response = llm.invoke(prompt_value)
    reformulated = (response.content or "").strip()
    return reformulated or question


def _format_context(documents) -> str:
    """Join retrieved chunks into a single context block for the prompt."""
    return "\n\n".join(doc.page_content for doc in documents)


def _build_sources(documents) -> List[Dict]:
    """Turn retrieved chunks into a deduplicated list of {filename, page}."""
    seen = set()
    sources: List[Dict] = []
    for doc in documents:
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        key = (filename, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"filename": filename, "page": page})
    return sources


def get_answer(question: str, chat_history: List[Dict[str, str]]) -> Tuple[str, List[Dict]]:
    """Run the RAG pipeline by hand: reformulate -> retrieve -> generate.

    Three explicit steps instead of one prebuilt LangChain "chain" object -
    easier to read, easier to debug, and immune to the chains-module churn
    described in the module docstring above. Kept as a small, mockable
    function so tests can monkeypatch it instead of calling the real cloud
    LLM.
    """
    standalone_question = _contextualize_question(question, chat_history)

    documents = retriever.invoke(standalone_question)
    context = _format_context(documents)

    prompt_value = qa_prompt.invoke(
        {"context": context, "chat_history": chat_history, "input": question}
    )
    response = llm.invoke(prompt_value)

    sources = _build_sources(documents)
    return response.content, sources