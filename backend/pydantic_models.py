"""Pydantic request/response models used by the FastAPI app."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class QueryInput(BaseModel):
    """Request body for POST /query."""

    question: str = Field(
        ...,
        description="The question to ask about the uploaded documents.",
        examples=["What is the content of this document?"],
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Existing chat session id for history context.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("question must not be empty")
        return value.strip()


class SourceInfo(BaseModel):
    """A single retrieved chunk source info."""

    filename: str = Field(..., description="Original name of the source document.")
    page: Optional[int] = Field(
        default=None,
        description="Page number if available.",
    )


class QueryResponse(BaseModel):
    """Response body for POST /query matching graduation project specs."""

    answer: str
    session_id: str
    model: str
    sources: List[SourceInfo] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """A row from the document_store table."""

    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel):
    """Request body for POST /delete-doc."""

    file_id: int