"""Document ingestion pipeline: load -> clean -> chunk -> embed -> store.

Supports PDF, DOCX, and HTML. Embeddings use Sentence-Transformers
(all-MiniLM-L6-v2) and are stored in a persistent local ChromaDB
collection so the backend never has to rebuild the index on restart.
"""

import logging
import os
import re
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import BSHTMLLoader, Docx2txtLoader, PyPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# chunk_size=1800 / chunk_overlap=350 (characters) - tuned for long,
# section-based technical books (e.g. "Hands-On Machine Learning") rather
# than short prose documents. A full chapter explains one concept across
# several paragraphs (and often a code block), so a small 1000-char chunk
# cuts that explanation in half and the LLM only ever sees one half. Bigger
# chunks keep a whole explanation (or a code example + the paragraph
# introducing it) together. The 350-character overlap (~20%) still protects
# against losing a sentence at a chunk boundary. See
# notebooks/rag_pipeline.ipynb for a discussion with real examples.
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 350

# ---------------------------------------------------------------------------
# Expensive resources - created ONCE at import time and reused for every
# request (loading the embedding model and opening Chroma per-request would
# be slow and pointless).
# ---------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    # Try to break on headings/paragraphs/sentences before falling back to
    # a hard character cut - keeps code blocks and paragraphs intact more
    # often than the default separator list.
    separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
)

logger.info("Loading embedding model '%s' (first run may download it)...", EMBEDDING_MODEL_NAME)
embedding_function = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_function)


def _clean_text(text: str) -> str:
    """Light, format-agnostic cleanup applied to every loaded page/section."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)       # collapse repeated spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)    # collapse 3+ blank lines to 1
    return text.strip()


def load_and_split_document(file_path: str, filename: str) -> List[Document]:
    """Load a file, clean its text, and split it into chunks with metadata.

    `filename` is the *original* uploaded name (the file on disk is a
    temporary, randomly-named copy), so it's passed separately and stamped
    onto every chunk's metadata for citations later.
    """
    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif extension == ".html":
        # BSHTMLLoader (BeautifulSoup) instead of UnstructuredHTMLLoader:
        # same result for plain HTML pages, but only needs the small
        # `beautifulsoup4` package instead of the much heavier
        # `unstructured` dependency tree - kinder to limited disk space.
        # bs_kwargs forces Python's built-in html.parser instead of
        # BSHTMLLoader's default (`lxml`, a much larger C-extension
        # package we don't want to require).
        loader = BSHTMLLoader(
            file_path, open_encoding="utf-8", bs_kwargs={"features": "html.parser"}
        )
    else:
        raise ValueError(f"Unsupported file type: {extension}")

    documents = loader.load()

    for doc in documents:
        doc.page_content = _clean_text(doc.page_content)
        doc.metadata["filename"] = filename

        # PyPDFLoader sets metadata['page'] starting at 0; humans count
        # pages from 1, so normalize it here for nicer citations.
        page: Optional[int] = doc.metadata.get("page")
        if page is not None:
            try:
                doc.metadata["page"] = int(page) + 1
            except (TypeError, ValueError):
                doc.metadata["page"] = None
        else:
            doc.metadata["page"] = None

    return text_splitter.split_documents(documents)


def index_document_to_chroma(file_path: str, file_id: int, filename: str) -> bool:
    """Run the full pipeline for one uploaded file and add it to Chroma."""
    try:
        splits = load_and_split_document(file_path, filename)

        if not splits:
            logger.warning("No extractable text found in '%s'", filename)
            return False

        for split in splits:
            split.metadata["file_id"] = file_id
            split.metadata.setdefault("filename", filename)

        vectorstore.add_documents(splits)
        logger.info("Indexed %d chunk(s) from '%s' (file_id=%s)", len(splits), filename, file_id)
        return True
    except Exception:
        logger.exception("Error indexing document '%s'", filename)
        return False


def delete_doc_from_chroma(file_id: int) -> bool:
    """Remove every chunk belonging to a given file_id from Chroma."""
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        logger.info("Deleting %d chunk(s) for file_id=%s", len(docs["ids"]), file_id)
        vectorstore._collection.delete(where={"file_id": file_id})
        return True
    except Exception:
        logger.exception("Error deleting file_id=%s from Chroma", file_id)
        return False