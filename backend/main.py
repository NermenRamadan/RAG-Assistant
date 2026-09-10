"""FastAPI backend for the RAG Document Assistant."""

import logging
import os
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
# Logging Config
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)
logger = logging.getLogger("rag_api")

# Ensure DB tables exist
create_application_logs()
create_document_store()

# ---------------------------------------------------------------------------
# App & CORS Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RAG Document Assistant",
    description="Grounded, cited question answering over custom documents.",
    version="1.0.0",
)

_allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # تم التعديل مؤقتاً لتجنب مشاكل الـ CORS في السيرفر المحلي
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".html"}


@app.get("/health")
def health_check():
    """Liveness check endpoint."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(query_input: QueryInput):
    """Main RAG Endpoint."""
    session_id = query_input.session_id or str(uuid.uuid4())

    logger.info("session=%s question=%r", session_id, query_input.question)
    chat_history = get_chat_history(session_id)

    try:
        answer, raw_sources = get_answer(query_input.question, chat_history)
        
        # تحويل قائمة المصادر لتتناسب مع موديل SourceInfo بشكل آمن
        formatted_sources = []
        for src in raw_sources:
            if isinstance(src, dict):
                formatted_sources.append(SourceInfo(
                    filename=src.get("filename", "Unknown"),
                    page=src.get("page")
                ))
            elif isinstance(src, str):
                formatted_sources.append(SourceInfo(filename=src))
                
    except Exception as e:
        logger.exception("RAG pipeline failed for session=%s: %s", session_id, str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The AI service is currently unavailable. Please check your API Key / Grok setup.",
        )

    insert_application_logs(session_id, query_input.question, answer, DEFAULT_MODEL_NAME)
    logger.info("session=%s sources=%d", session_id, len(formatted_sources))

    return QueryResponse(
        answer=answer,
        session_id=session_id,
        model=DEFAULT_MODEL_NAME,
        sources=formatted_sources,
    )


@app.post("/upload-doc", status_code=status.HTTP_201_CREATED)
def upload_and_index_document(file: UploadFile = File(...)):
    """Upload and Index a file."""
    file_extension = Path(file.filename or "").suffix.lower()

    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file_extension}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

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
                detail=f"Failed to process '{file.filename}'. Check file validity.",
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
            detail="An error occurred during file upload.",
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
            detail=f"Failed to delete document {request.file_id} from vector store.",
        )

    db_ok = delete_document_record(request.file_id)
    if not db_ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deleted from vector store but failed in DB record.",
        )

    logger.info("Deleted document file_id=%s", request.file_id)
    return {"message": f"Document {request.file_id} deleted successfully."}