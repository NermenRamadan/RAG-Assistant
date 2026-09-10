"""Thin HTTP client the Streamlit app uses to talk to the FastAPI backend.

The backend URL always comes from the BACKEND_URL environment variable -
it is never hard-coded, so the same frontend code works locally or deployed.
"""

import os
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 60  # seconds; cloud LLM calls can take a little while


def _extract_error_detail(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


def get_api_response(question: str, session_id: Optional[str]):
    """Call POST /query. Returns the parsed JSON response, or None on error."""
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id

    try:
        response = requests.post(f"{BACKEND_URL}/query", json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not reach the backend. Is the FastAPI server running?")
    except requests.exceptions.Timeout:
        st.error("The request took too long and timed out. Please try again.")
    except requests.exceptions.HTTPError:
        st.error(f"The backend returned an error: {_extract_error_detail(response)}")
    except Exception as exc:  # last-resort guard so the UI never crashes
        st.error(f"An unexpected error occurred: {exc}")
    return None


def upload_document(file):
    """Call POST /upload-doc. Returns the parsed JSON response, or None on error."""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{BACKEND_URL}/upload-doc", files=files, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not reach the backend. Is the FastAPI server running?")
    except requests.exceptions.HTTPError:
        st.error(f"Upload failed: {_extract_error_detail(response)}")
    except Exception as exc:
        st.error(f"An unexpected error occurred while uploading: {exc}")
    return None


def list_documents():
    """Call GET /list-docs. Returns a list (possibly empty) - never None."""
    try:
        response = requests.get(f"{BACKEND_URL}/list-docs", timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not reach the backend. Is the FastAPI server running?")
    except Exception as exc:
        st.error(f"Failed to fetch the document list: {exc}")
    return []


def delete_document(file_id: int):
    """Call POST /delete-doc. Returns the parsed JSON response, or None on error."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/delete-doc", json={"file_id": file_id}, timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not reach the backend. Is the FastAPI server running?")
    except requests.exceptions.HTTPError:
        st.error(f"Delete failed: {_extract_error_detail(response)}")
    except Exception as exc:
        st.error(f"An unexpected error occurred while deleting: {exc}")
    return None
