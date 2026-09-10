# RAG Document Assistant

A document-based Retrieval-Augmented Generation (RAG) web app. Upload PDF, DOCX, or
HTML documents and ask questions about them — answers are grounded in your documents
and come with cited sources (filename + page).

## 1. Overview

```
Documents -> Load -> Clean -> Chunk -> Sentence-Transformer Embeddings ->
Persistent ChromaDB -> Retriever -> Cloud LLM -> Grounded Answer + Sources ->
FastAPI -> Streamlit
```

## 2. Problem Statement

Plain LLMs answer from general training data and can hallucinate facts that aren't in
your actual documents, with no way to check where an answer came from. This project
answers questions **only** from a set of documents you provide, and cites the exact
file (and page, when available) each part of the answer came from.

## 3. Proposed Solution

A small FastAPI backend indexes uploaded documents into a local, persistent vector
database (ChromaDB) using Sentence-Transformer embeddings. At query time it retrieves
the most relevant chunks and asks a cloud LLM to answer using only that retrieved
context, returning the answer together with its sources. A Streamlit frontend provides
a chat UI, document upload, and document management.

## 4. Architecture

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend API | FastAPI |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persisted to `api/chroma_db/`) |
| LLM | Cloud LLM via an OpenAI-compatible API (Groq by default) |
| Chat history / document records | SQLite (`api/rag_app.db`) |

## 5. RAG Pipeline

1. **Load** — `PyPDFLoader` (PDF), `Docx2txtLoader` (DOCX), `BSHTMLLoader` (HTML).
2. **Clean** — collapse repeated whitespace/blank lines, strip stray null bytes.
3. **Chunk** — `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`.
   See `notebooks/rag_pipeline.ipynb` §6 for the reasoning behind these values.
4. **Embed** — `all-MiniLM-L6-v2` sentence-transformer, 384-dim vectors.
5. **Store** — persistent Chroma collection at `api/chroma_db/`.
6. **Retrieve** — top-`k=4` similarity search.
7. **Generate** — cloud LLM answers strictly from the retrieved context; a system
   prompt instructs it to say so explicitly when the answer isn't in the documents.
8. **Cite** — every answer returns the `filename` (+ `page`, for PDFs) of each chunk used.

## 6. Technologies

FastAPI · Streamlit · LangChain (`langchain-core`, `langchain-openai`,
`langchain-community`, `langchain-chroma`) · ChromaDB · Sentence-Transformers ·
SQLite · pytest

**Why a cloud LLM instead of local Ollama:** the original assignment specifies a local
Ollama model. This machine doesn't have the disk space/hardware to reasonably run a
local LLM, so this implementation uses a cloud LLM API instead, configured entirely
through environment variables (see §10). This is **not** the same as running Ollama
locally — it requires an internet connection and (depending on provider) an API key
with usage limits/cost. That trade-off is called out explicitly here rather than
glossed over.

## 7. Project Structure

```
rag-assistant/
├── api/
│   ├── __init__.py
│   ├── main.py              FastAPI app: /health, /query, /upload-doc, /list-docs, /delete-doc
│   ├── pydantic_models.py   Request/response models + validation
│   ├── langchain_utils.py   RAG pipeline (retrieval + cloud LLM)
│   ├── db_utils.py          SQLite: chat history + document records
│   └── chroma_utils.py      Document loading/cleaning/chunking/embeddings/Chroma
├── app/
│   ├── streamlit_app.py     Streamlit entry point
│   ├── api_utils.py         HTTP client for the FastAPI backend
│   ├── chat_interface.py    Chat UI (answers + cited sources)
│   └── sidebar.py           Upload / list / delete documents
├── notebooks/
│   └── rag_pipeline.ipynb   Pipeline walkthrough + evaluation (10+ questions)
├── tests/
│   ├── conftest.py
│   └── test_api.py
├── docs/                    Put your source documents here
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 8. Supported File Types

`.pdf`, `.docx`, `.html`

## 9. Installation

You said your virtual environment already has most packages installed. Only install
what's actually missing — see the exact commands in the **PART 3 — Installation**
section of the chat response, or simply run:

```bash
pip install -r requirements.txt --break-system-packages   # if not using a venv
# or, inside an activated venv:
pip install -r requirements.txt
```

pip will skip anything already satisfied.

## 10. Environment Variables

Copy `.env.example` to `.env` and fill in real values. **Never commit `.env`.**

| Variable | Used by | Purpose |
|---|---|---|
| `LLM_API_KEY` | `api/` | API key for your cloud LLM provider |
| `LLM_MODEL` | `api/` | Model name, e.g. `openai/gpt-oss-20b` (Groq) |
| `LLM_BASE_URL` | `api/` | OpenAI-compatible endpoint URL |
| `ALLOWED_ORIGINS` | `api/` | Comma-separated origins allowed by CORS |
| `BACKEND_URL` | `app/` | URL the Streamlit app uses to reach the FastAPI backend |

## 11. Backend Setup

```bash
cd api
uvicorn main:app --reload --port 8000
```

## 12. Frontend Setup

```bash
cd app
streamlit run streamlit_app.py
```

## 13. Running the Application

1. Start the backend (§11), confirm `GET http://localhost:8000/health` returns
   `{"status": "ok"}`.
2. Start the frontend (§12) in a second terminal.
3. Open the Streamlit URL it prints (usually `http://localhost:8501`).
4. Upload a PDF/DOCX/HTML file from the sidebar.
5. Ask a question in the chat box.

## 14. API Reference

### `GET /health`
Returns `{"status": "ok"}`.

### `POST /query`
Request:
```json
{ "question": "What is the warranty period?", "session_id": "optional-existing-id" }
```
Response:
```json
{
  "answer": "The warranty lasts for two years.",
  "session_id": "a1b2c3...",
  "model": "openai/gpt-oss-20b",
  "sources": [ { "filename": "manual.pdf", "page": 4 } ]
}
```

### `POST /upload-doc`
`multipart/form-data`, field `file` — a `.pdf`, `.docx`, or `.html` file.
Response: `{ "message": "...", "file_id": 1 }`

### `GET /list-docs`
Response: `[ { "id": 1, "filename": "manual.pdf", "upload_timestamp": "..." } ]`

### `POST /delete-doc`
Request: `{ "file_id": 1 }`
Response: `{ "message": "Document 1 deleted successfully." }`

## 15. Example curl Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the warranty period?"}'
```

## 16. Evaluation Methodology

`notebooks/rag_pipeline.ipynb` runs 10+ sample questions through the full pipeline and
builds an evaluation table (Question / Expected Answer / Retrieved Context / Generated
Answer / Grounded? / Correct? / Sources). `Grounded?` is computed automatically;
`Expected Answer` and `Correct?` are placeholders filled in by hand after reviewing
each answer against the source documents, since correctness needs human judgment.

## 17. Evaluation Results

Run the notebook end-to-end (`Kernel -> Restart & Run All`) with your own documents in
`docs/`. It saves results to `notebooks/evaluation_results.csv`. Paste a summary table
here once you've filled in the placeholders.

## 18. Limitations

See `notebooks/rag_pipeline.ipynb` §17 for the full list — in short: cloud LLM (not
local), general-purpose small embedding model, fixed-size chunking, no OCR for scanned
documents, simple top-k retrieval with no re-ranking, small manually-judged evaluation.

## 19. Screenshots

*(Add screenshots of the chat interface and document upload here before submitting.)*

## 20. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/query` returns 502 | `LLM_API_KEY` missing or invalid in `.env` |
| `ModuleNotFoundError` on startup | A dependency is genuinely missing — `pip install <name>` |
| HTML upload fails to index | Make sure `beautifulsoup4` is installed |
| Frontend can't reach backend | Check `BACKEND_URL` in `.env` and that the backend is running |
| CORS error in browser console | Add the Streamlit URL to `ALLOWED_ORIGINS` in `.env` |
| First query is slow | First run downloads the ~90MB embedding model — needs internet once |
