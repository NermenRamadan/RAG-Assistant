# 📚 RAG Document & Book Assistant

A production-ready **Retrieval-Augmented Generation (RAG)** application. Upload your own documents/books (PDF, DOCX, HTML) and chat with them — all answers are strictly grounded in your context with exact cited sources (filename + page number) to eliminate hallucinations.

---

## 🌟 Key Features

* **Multi-Format Document Support:** Upload PDFs (e.g., textbooks/manuals), DOCX, or HTML files.
* **Strict Grounded Answers:** Answers are generated strictly using the retrieved content to prevent hallucinations. If information isn't present, the assistant explicitly states it.
* **Interactive Chat Interface:** Send single questions or engage in continuous chat sessions with context awareness.
* **Source Attribution:** Every response includes exact source citations (filename and page numbers).
* **Decoupled Architecture:** Clean separation between the FastAPI backend and Streamlit frontend.

---

## 🏗️ Architecture

| Layer | Technology |
| :--- | :--- |
| **Frontend** | Streamlit (Interactive Chat & Document Management) |
| **Backend API** | FastAPI (Endpoints, Validation, & Session Management) |
| **Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`, 384-dim) |
| **Vector Store** | ChromaDB (Persistent local storage) |
| **LLM** | Cloud LLM via OpenAI-compatible API (Groq / OpenAI) |
| **Test Suite** | Pytest with `FastAPI TestClient` & Mocking |

---

## 🚀 RAG Pipeline Flow

1. **Load:** Parse uploaded documents/books using LangChain document loaders (`PyPDFLoader`, `Docx2txtLoader`, `BSHTMLLoader`).
2. **Clean & Chunk:** Normalize whitespace and split text using `RecursiveCharacterTextSplitter` (`chunk_size=1000`, `chunk_overlap=200`).
3. **Embed & Store:** Generate dense vector embeddings and index them into persistent ChromaDB storage.
4. **Retrieve:** Perform top-$k$ similarity search ($k=6$) based on the user's prompt or chat follow-ups.
5. **Generate & Cite:** The Cloud LLM synthesizes an answer using *only* retrieved context, formatted clearly with bullet points and source citations.

---

## 📂 Project Structure

```text
rag-assistant/
├── backend/
│   ├── main.py              # FastAPI endpoints (/health, /query, etc.)
│   ├── langchain_utils.py   # RAG chain, prompts, & LLM configuration
│   ├── chroma_utils.py      # Document loading, chunking, & vectorstore
│   ├── db_utils.py          # Chat history & record management
│   └── pydantic_models.py   # Request/Response schemas
├── frontend/
│   ├── streamlit_app.py     # Streamlit entry point
│   ├── chat_interface.py    # UI for chat & messaging
│   ├── sidebar.py           # Document upload & management UI
│   └── api_utils.py         # HTTP client communicating with backend
├── tests/
│   ├── test_api.py          # Unit tests for FastAPI endpoints
│   └── conftest.py          # Pytest fixtures
├── notebooks/
│   ├── rag_pipeline.ipynb   # Pipeline experiments & evaluation
│   └── evaluation_results.csv
├── docs/                    # Local storage for documents (Git ignored for copyright)
├── .env.example             # Template for required environment variables
├── .gitignore               # Excludes secrets, caches, & raw documents
├── requirements.txt         # Project dependencies
└── README.md