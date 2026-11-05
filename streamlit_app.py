import hashlib
import os
import sqlite3
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from qa_system import (
    NOT_FOUND_ANSWER,
    create_embedding_model,
    generate_grounded_answer,
    index_document,
    retrieve_passages,
)
from storage import (
    delete_document,
    document_exists,
    get_messages,
    init_db,
    list_conversations,
    list_documents,
    save_exchange,
)


APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env", override=False)
DB_PATH = Path(os.environ.get("QA_DB_PATH", APP_DIR / ".data" / "document_qa.sqlite3"))
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return create_embedding_model(EMBEDDING_MODEL_NAME)


def load_groq_api_key():
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets["GROQ_API_KEY"]
    except (KeyError, st.errors.StreamlitSecretNotFoundError):
        return None


def render_sources(sources):
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"**{source['filename']} — {source['location']}**")
            st.caption(f"Relevance: {source['score']:.2f}")
            st.write(f"> {source['text']}")


st.set_page_config(
    page_title="Document Q&A",
    page_icon=":material/menu_book:",
    layout="centered",
)
init_db(DB_PATH)

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "history_selector" not in st.session_state:
    st.session_state.history_selector = st.session_state.conversation_id
elif st.session_state.history_selector is None and st.session_state.conversation_id:
    st.session_state.history_selector = st.session_state.conversation_id

st.title("Document Q&A")
st.caption("Ask questions about your files and verify every answer against its sources. Answers are generated with Groq.")

with st.sidebar:
    st.header("Document library")
    uploads = st.file_uploader(
        "Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        max_upload_size=20,
        help="Files and extracted passages stay on this machine.",
    )

    if uploads:
        for uploaded in uploads:
            uploaded_bytes = uploaded.getvalue()
            content_hash = hashlib.sha256(uploaded_bytes).hexdigest()
            if document_exists(DB_PATH, content_hash):
                continue
            try:
                with st.status(f"Preparing {uploaded.name}...", expanded=True) as status:
                    status.update(
                        label="Loading a local search model..."
                    )
                    embedding_model = load_embedding_model()
                    if getattr(embedding_model, "fallback_notice", None):
                        st.session_state.embedding_notice = embedding_model.fallback_notice
                    progress = st.progress(0, text="Starting document indexing")

                    def report_indexing_progress(stage, completed, total):
                        progress.progress(
                            completed / total if total else 0,
                            text=(
                                f"{stage}: {completed:,}/{total:,}"
                                if total > 1
                                else stage
                            ),
                        )

                    added = index_document(
                        DB_PATH,
                        uploaded.name,
                        uploaded_bytes,
                        embedding_model,
                        progress_callback=report_indexing_progress,
                    )
                    progress.empty()
                    status.update(
                        label=(
                            f"Indexed {uploaded.name}"
                            if added
                            else f"{uploaded.name} is already indexed"
                        ),
                        state="complete",
                    )
            except (ImportError, OSError, RuntimeError, ValueError, sqlite3.Error) as error:
                st.error(f"Could not index {uploaded.name}: {error}")

    documents = list_documents(DB_PATH)
    st.caption(f"{len(documents)} document{'s' if len(documents) != 1 else ''} indexed")
    if st.session_state.get("embedding_notice"):
        st.warning(st.session_state.embedding_notice, icon=":material/wifi_off:")

    if documents:
        document_to_remove = st.selectbox(
            "Select a document to remove",
            documents,
            format_func=lambda document: document["filename"],
            key="document_to_remove",
        )
        if st.button("Remove document", icon=":material/delete:"):
            delete_document(DB_PATH, document_to_remove["id"])
            st.rerun()

    st.divider()
    st.header("Conversation history")
    conversations = list_conversations(DB_PATH)
    conversation_ids = [conversation["id"] for conversation in conversations]
    options = [None, *conversation_ids]
    if st.session_state.history_selector not in options:
        st.session_state.history_selector = st.session_state.conversation_id
    selected_conversation = st.selectbox(
        "Open a conversation",
        options,
        format_func=lambda conversation_id: (
            "New chat"
            if conversation_id is None
            else next(
                item["title"]
                for item in conversations
                if item["id"] == conversation_id
            )
        ),
        key="history_selector",
        label_visibility="collapsed",
    )
    st.session_state.conversation_id = selected_conversation

    if st.button("Start a new chat", icon=":material/add_comment:"):
        st.session_state.conversation_id = None
        st.session_state.history_selector = None
        st.rerun()

active_conversation_id = st.session_state.conversation_id
messages = (
    get_messages(DB_PATH, active_conversation_id)
    if active_conversation_id
    else []
)

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message["sources"])

if not documents:
    st.info("Upload at least one document in the sidebar to get started.")

question = st.chat_input(
    "Ask a question about your documents",
    disabled=not documents,
)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.status("Searching your documents...", type="compact") as status:
                embedding_model = load_embedding_model()
                passages = retrieve_passages(
                    DB_PATH,
                    question,
                    embedding_model,
                    history=messages,
                )
                if passages:
                    status.update(label="Generating a grounded answer...")
                    streamed_answer = []
                    answer_placeholder = st.empty()

                    def show_answer_chunk(chunk):
                        streamed_answer.append(chunk)
                        answer_placeholder.markdown("".join(streamed_answer))

                    answer = generate_grounded_answer(
                        question,
                        passages,
                        history=messages,
                        model_name=os.environ.get(
                            "GROQ_MODEL",
                            "openai/gpt-oss-120b",
                        ),
                        api_key=load_groq_api_key(),
                        stream_callback=show_answer_chunk,
                    )
                else:
                    answer = NOT_FOUND_ANSWER
                status.update(label="Done", state="complete")

            if answer == NOT_FOUND_ANSWER:
                passages = []
            if passages:
                answer_placeholder.markdown(answer)
            else:
                st.markdown(answer)
            render_sources(passages)
            st.session_state.conversation_id = save_exchange(
                DB_PATH,
                active_conversation_id,
                question,
                answer,
                passages,
            )
        except (ImportError, OSError, RuntimeError, ValueError, sqlite3.Error) as error:
            st.error(f"Unable to answer this question: {error}")
