"""FastAPI backend for the RAG Document Assistant.

Endpoints:
    GET  /health       liveness check
    POST /query        ask a grounded, cited question (the main RAG endpoint)
    POST /upload-doc   upload + index a PDF/DOCX/HTML file
    GET  /list-docs    list uploaded documents
    POST /delete-doc   delete a document and its vectors
"""

import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from chroma_utils import delete_doc_from_chroma, index_document_to_chroma
from db_utils import (
    create_application_logs,
    create_document_store,
    delete_document_record,
    get_all_documents,
    get_chat_history,
    insert_application_logs,
    insert_document_record,
)
from langchain_utils import DEFAULT_MODEL_NAME, get_answer
from pydantic_models import DeleteFileRequest, DocumentInfo, QueryInput, QueryResponse, SourceInfo

# ---------------------------------------------------------------------------
# Logging - to a file (for later inspection) and to the console.
#
# encoding="utf-8" on BOTH handlers is required: without it, on Windows the
# console/file default to the system codepage (cp1252), which cannot encode
# Arabic (or most non-Latin) characters and crashes with UnicodeEncodeError
# the moment someone asks a question in Arabic - even though the request
# itself succeeds. errors="backslashreplace" is a last-resort safety net so
# logging itself can never crash the app even on a stranger console setup.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8", errors="backslashreplace"),
        logging.StreamHandler(),
    ],
)
# StreamHandler doesn't take an `encoding=` kwarg (it just uses whatever
# stream you give it), so force stdout itself into UTF-8 mode. Needs a
# feature check because this method doesn't exist on very old Python.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
logger = logging.getLogger("rag_api")

# ---------------------------------------------------------------------------
# Make sure the SQLite tables exist. Safe to call on every startup.
# ---------------------------------------------------------------------------
create_application_logs()
create_document_store()

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Document Assistant",
    description="Grounded, cited question answering over your own PDF/DOCX/HTML documents.",
    version="1.0.0",
)

# Never use allow_origins=["*"] - read the allowed frontend origin(s) from
# the environment instead. Comma-separate multiple origins if needed.
_allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".html"}


@app.get("/health")
def health_check():
    """Liveness check used by the frontend and by deployment tooling."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(query_input: QueryInput):
    """The main RAG endpoint: retrieve relevant chunks, ask the cloud LLM,
    and return a grounded answer together with the sources it used."""
    session_id = query_input.session_id or str(uuid.uuid4())

    logger.info("session=%s question=%r", session_id, query_input.question)
    chat_history = get_chat_history(session_id)

    try:
        answer, sources = get_answer(query_input.question, chat_history)
    except Exception:
        # Never leak internal exception details (API keys, stack traces, etc.)
        # to the client - log them server-side and return a generic message.
        logger.exception("RAG pipeline failed for session=%s", session_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI service is currently unavailable. Please try again shortly.",
        )

    insert_application_logs(session_id, query_input.question, answer, DEFAULT_MODEL_NAME)
    logger.info("session=%s sources=%d", session_id, len(sources))

    return QueryResponse(
        answer=answer,
        session_id=session_id,
        model=DEFAULT_MODEL_NAME,
        sources=[SourceInfo(**source) for source in sources],
    )


@app.post("/upload-doc", status_code=status.HTTP_201_CREATED)
def upload_and_index_document(file: UploadFile = File(...)):
    """Upload a PDF/DOCX/HTML file, index it into Chroma, and record it in SQLite."""
    file_extension = Path(file.filename or "").suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{file_extension or 'unknown'}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    # Use a random temp name (not the raw uploaded filename) to avoid path
    # issues, while still passing the *original* filename through for
    # metadata/citations.
    temp_file_path = f"temp_{uuid.uuid4().hex}{file_extension}"
    file_id = None

    try:
        with open(temp_file_path, "wb") as buffer:
            buffer.write(file.file.read())

        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id, file.filename)

        if not success:
            delete_document_record(file_id)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process '{file.filename}'. Make sure it is a valid, text-based file.",
            )

        logger.info("Uploaded and indexed '%s' as file_id=%s", file.filename, file_id)
        return {
            "message": f"File '{file.filename}' has been uploaded and indexed successfully.",
            "file_id": file_id,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error while uploading '%s'", file.filename)
        if file_id is not None:
            delete_document_record(file_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the file.",
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()


@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    chroma_ok = delete_doc_from_chroma(request.file_id)
    if not chroma_ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document {request.file_id} from the vector store.",
        )

    db_ok = delete_document_record(request.file_id)
    if not db_ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Deleted document {request.file_id} from the vector store but failed "
                "to remove its database record."
            ),
        )

    logger.info("Deleted document file_id=%s", request.file_id)
    return {"message": f"Document {request.file_id} deleted successfully."}