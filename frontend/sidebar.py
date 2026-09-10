import streamlit as st
from api_utils import delete_document, list_documents, upload_document

ALLOWED_TYPES = ["pdf", "docx", "html"]


def display_sidebar():
    # --- Upload ------------------------------------------------------------
    st.sidebar.header("Upload a Document")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a PDF, DOCX, or HTML file", type=ALLOWED_TYPES
    )
    if uploaded_file is not None and st.sidebar.button("Upload"):
        with st.spinner("Uploading and indexing..."):
            upload_response = upload_document(uploaded_file)
            if upload_response:
                st.sidebar.success(
                    f"'{uploaded_file.name}' uploaded (file_id {upload_response['file_id']})."
                )
                st.session_state.documents = list_documents()

    # --- List ----------------------------------------------------------------
    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        with st.spinner("Refreshing..."):
            st.session_state.documents = list_documents()

    if "documents" not in st.session_state:
        st.session_state.documents = list_documents()

    documents = st.session_state.documents
    if not documents:
        st.sidebar.info("No documents uploaded yet.")
        return

    for doc in documents:
        st.sidebar.text(f"{doc['filename']} (ID: {doc['id']})")

    # --- Delete --------------------------------------------------------------
    selected_file_id = st.sidebar.selectbox(
        "Select a document to delete",
        options=[doc["id"] for doc in documents],
        format_func=lambda x: next(doc["filename"] for doc in documents if doc["id"] == x),
    )
    if st.sidebar.button("Delete Selected Document"):
        with st.spinner("Deleting..."):
            delete_response = delete_document(selected_file_id)
            if delete_response:
                st.sidebar.success(f"Document {selected_file_id} deleted.")
                st.session_state.documents = list_documents()
