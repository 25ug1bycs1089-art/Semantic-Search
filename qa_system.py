import hashlib
import math
import os
import re
from io import BytesIO
from pathlib import Path

from storage import (
    document_exists,
    get_passages,
    store_document,
)


NOT_FOUND_ANSWER = "Answer not found in the provided documents."
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
MAX_PASSAGES = 5
MIN_RELEVANCE = 0.28
EMBEDDING_BATCH_SIZE = 64
LEXICAL_STOP_WORDS = frozenset(
    "a about an and are as at be been being but by can did do does for from had has have "
    "how i in into is it its of on or that the their them then there these they this "
    "those to was we were what when where which who why will with you your make one "
    "say said tell".split()
)


class HashingEmbeddingModel:
    model_id = "offline-word-ngram-hashing-v1"
    fallback_notice = (
        "The semantic model is unavailable locally or Hugging Face is unreachable. "
        "This document was indexed using offline keyword matching instead, which is "
        "less effective for paraphrases."
    )

    def __init__(self):
        try:
            from sklearn.feature_extraction.text import HashingVectorizer
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Offline search requires scikit-learn. From the project folder, run "
                "`py -m pip install -r requirements.txt` and restart the app."
            ) from error

        self._vectorizer = HashingVectorizer(
            n_features=512,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            strip_accents="unicode",
        )

    @staticmethod
    def _terms(text):
        terms = set()
        for word in re.findall(r"\w+", text.casefold()):
            if len(word) <= 1 or word in LEXICAL_STOP_WORDS:
                continue
            if len(word) > 5 and word.endswith("ies"):
                word = f"{word[:-3]}y"
            elif len(word) > 5 and word.endswith(("ing", "ed")):
                word = word[:-3] if word.endswith("ing") else word[:-2]
                if len(word) > 3 and word[-1] == word[-2]:
                    word = word[:-1]
            elif len(word) > 4 and word.endswith("es"):
                word = word[:-2]
            elif len(word) > 3 and word.endswith("s"):
                word = word[:-1]
            terms.add(word)
        return terms

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return self._vectorizer.transform(texts).toarray()

    def score(self, query, passage):
        query_terms = self._terms(query)
        if not query_terms:
            return 0.0
        passage_terms = self._terms(passage)
        return len(query_terms & passage_terms) / len(query_terms)


def create_embedding_model(model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError:
        return HashingEmbeddingModel()

    try:
        model = SentenceTransformer(model_name, local_files_only=True)
    except OSError:
        return HashingEmbeddingModel()

    model.model_id = f"sentence-transformers:{model_name}"
    return model


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than it.")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n\n", start + chunk_size // 2, end),
                text.rfind(" ", start + chunk_size // 2, end),
            )
            if boundary > start:
                end = boundary

        content = text[start:end].strip()
        if content:
            chunks.append((start, end, content))
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _chunks_for_section(text, location):
    chunks = []
    line_cursor = 0
    line_at_cursor = 1
    for start, end, content in split_text(text):
        first_line = line_at_cursor + text.count("\n", line_cursor, start)
        last_line = first_line + text.count("\n", start, end)
        if location == "Text":
            chunk_location = (
                f"Line {first_line}"
                if first_line == last_line
                else f"Lines {first_line}-{last_line}"
            )
        else:
            chunk_location = location
        chunks.append({"text": content, "location": chunk_location})
        line_cursor = start
        line_at_cursor = first_line
    return chunks


def _parse_docx(data):
    try:
        from docx import Document
    except ImportError as error:
        raise ImportError("Install the project requirements to read DOCX files.") from error

    document = Document(BytesIO(data))
    sections = []
    current_title = None
    current_paragraphs = []

    def finish_section():
        nonlocal current_paragraphs
        if current_paragraphs:
            title = current_title or f"Section {len(sections) + 1}"
            sections.append(("\n\n".join(current_paragraphs), f"Section: {title}"))
            current_paragraphs = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style and paragraph.style.name.startswith("Heading"):
            finish_section()
            current_title = text
        else:
            current_paragraphs.append(text)

    for table in document.tables:
        for row in table.rows:
            values = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if values:
                current_paragraphs.append(" | ".join(values))
    finish_section()

    return sections


def parse_document(filename, data):
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise ImportError("Install the project requirements to read PDF files.") from error

        try:
            reader = PdfReader(BytesIO(data))
            sections = [
                (page.extract_text() or "", f"Page {page_number}")
                for page_number, page in enumerate(reader.pages, start=1)
            ]
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not read PDF: {error}") from error
    elif extension == ".docx":
        sections = _parse_docx(data)
    elif extension == ".txt":
        try:
            sections = [(data.decode("utf-8-sig"), "Text")]
        except UnicodeDecodeError as error:
            raise ValueError("TXT files must use UTF-8 encoding.") from error
    else:
        raise ValueError("Only PDF, DOCX, and TXT files are supported.")

    passages = [
        chunk
        for text, location in sections
        for chunk in _chunks_for_section(text, location)
    ]
    if not passages:
        raise ValueError("This document contains no extractable text.")
    return passages


def index_document(db_path, filename, data, embedding_model, progress_callback=None):
    digest = hashlib.sha256(data).hexdigest()
    if document_exists(db_path, digest):
        return False

    if progress_callback:
        progress_callback("Extracting text", 0, 1)
    passages = parse_document(filename, data)
    vectors = []
    passage_count = len(passages)
    if progress_callback:
        progress_callback("Creating semantic embeddings", 0, passage_count)

    for start in range(0, passage_count, EMBEDDING_BATCH_SIZE):
        batch = passages[start : start + EMBEDDING_BATCH_SIZE]
        batch_vectors = embedding_model.encode(
            [passage["text"] for passage in batch],
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors.extend(batch_vectors)
        if progress_callback:
            progress_callback(
                "Creating semantic embeddings",
                min(start + len(batch), passage_count),
                passage_count,
            )

    if len(vectors) != len(passages):
        raise RuntimeError("The embedding model returned an unexpected number of vectors.")

    stored_passages = []
    for passage, vector in zip(passages, vectors):
        values = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        stored_passages.append(
            {
                **passage,
                "embedding": {
                    "model": getattr(embedding_model, "model_id", "custom"),
                    "vector": [float(value) for value in values],
                },
            }
        )
    if progress_callback:
        progress_callback("Saving document index", passage_count, passage_count)
    store_document(db_path, filename, digest, stored_passages)
    return True


def cosine_similarity(left, right):
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match.")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def build_retrieval_query(question, history):
    recent = history[-4:]
    if not recent:
        return question
    context = "\n".join(
        f"{message['role']}: {message['content']}" for message in recent
    )
    return f"Conversation context for resolving references:\n{context}\nCurrent question: {question}"


def build_lexical_query(question, history):
    question_terms = HashingEmbeddingModel._terms(question)
    if len(question_terms) > 2:
        return question

    previous_question = next(
        (message["content"] for message in reversed(history) if message["role"] == "user"),
        "",
    )
    return f"{previous_question} {question}".strip()


def retrieve_passages(
    db_path,
    question,
    embedding_model,
    history=None,
    limit=MAX_PASSAGES,
    min_relevance=MIN_RELEVANCE,
):
    passages = get_passages(db_path)
    if not passages:
        return []

    model_id = getattr(embedding_model, "model_id", "custom")
    stored_model_ids = {
        passage["embedding"].get("model", "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2")
        if isinstance(passage["embedding"], dict)
        else "sentence-transformers:sentence-transformers/all-MiniLM-L6-v2"
        for passage in passages
    }
    if stored_model_ids != {model_id}:
        raise RuntimeError(
            "The indexed passages use a different embedding model. Remove and re-upload "
            "the documents to rebuild their indexes with the currently available model."
        )

    if isinstance(embedding_model, HashingEmbeddingModel):
        query = build_lexical_query(question, history or [])
        score_passage = lambda passage: embedding_model.score(query, passage["text"])
        relevance_threshold = max(min_relevance, 0.25)
    else:
        query = build_retrieval_query(question, history or [])
        vectors = embedding_model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector = vectors[0]
        query_vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        score_passage = lambda passage: cosine_similarity(
            query_vector,
            passage["embedding"]["vector"]
            if isinstance(passage["embedding"], dict)
            else passage["embedding"],
        )
        relevance_threshold = min_relevance

    ranked = sorted(
        ({**passage, "score": score_passage(passage)} for passage in passages),
        key=lambda passage: passage["score"],
        reverse=True,
    )
    return [
        {key: value for key, value in passage.items() if key != "embedding"}
        for passage in ranked[:limit]
        if passage["score"] >= relevance_threshold
    ]


def _format_citations(answer, passages):
    if "answer not found" in answer.lower():
        return NOT_FOUND_ANSWER

    source_by_id = {f"S{index}": passage for index, passage in enumerate(passages, start=1)}
    cited_ids = []

    def replace_citation(match):
        source_id = match.group(1).upper()
        passage = source_by_id.get(source_id)
        if passage is None:
            return ""
        if source_id not in cited_ids:
            cited_ids.append(source_id)
        return f"[{source_id}: {passage['filename']}, {passage['location']}]"

    formatted = re.sub(r"\[(S\d+)\]", replace_citation, answer, flags=re.IGNORECASE).strip()
    if not cited_ids:
        references = "; ".join(
            f"[{source_id}: {source_by_id[source_id]['filename']}, "
            f"{source_by_id[source_id]['location']}]"
            for source_id in source_by_id
        )
        formatted = f"{formatted}\n\nSources: {references}"
    return formatted


def generate_grounded_answer(
    question,
    passages,
    history=None,
    model_name=None,
    api_key=None,
    stream_callback=None,
):
    if not passages:
        return NOT_FOUND_ANSWER

    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Groq API key is not configured. Set GROQ_API_KEY in your environment "
            "or add it to Streamlit secrets, then restart the app."
        )

    try:
        from openai import APIConnectionError, APIStatusError, OpenAI
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The OpenAI Python SDK is missing. From the project folder, run "
            "`py -m pip install -r requirements.txt` and restart the app."
        ) from error

    history_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in (history or [])[-4:]
        if message["role"] in {"user", "assistant"}
    ]
    source_text = "\n\n".join(
        f"[S{index}] {passage['filename']} — {passage['location']}\n{passage['text']}"
        for index, passage in enumerate(passages, start=1)
    )
    system_prompt = (
        "Answer the user's question using only the source passages below. "
        "Conversation history is only for resolving references; it is not evidence. "
        f"If the passages do not contain the answer, reply exactly: {NOT_FOUND_ANSWER} "
        "Be direct and concise. Cite each factual claim using one or more source markers "
        "such as [S1]. Do not invent facts or source markers.\n\n"
        f"Source passages:\n{source_text}"
    )
    request_body = {
        "model": model_name or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        "messages": [
            {"role": "system", "content": system_prompt},
            *history_messages,
            {"role": "user", "content": question},
        ],
        "temperature": 1,
        "max_completion_tokens": 2048,
        "top_p": 1,
        "reasoning_effort": "medium",
        "stream": True,
        "stop": None,
    }
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        max_retries=0,
        timeout=60,
    )
    try:
        stream = client.chat.completions.create(**request_body)
        answer_chunks = []
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                answer_chunks.append(content)
                if stream_callback:
                    stream_callback(content)
    except APIStatusError as error:
        detail = str(getattr(error, "body", None) or error)
        status_code = getattr(error, "status_code", "unknown")
        if status_code == 403 and "1010" in detail:
            raise RuntimeError(
                "Groq returned HTTP 403 (Cloudflare error 1010): Groq's edge network "
                "blocked this request. This is usually a network/IP access restriction, "
                "not a model parameter. Try an allowed network or contact Groq support."
            ) from error
        raise RuntimeError(f"Groq returned HTTP {status_code}: {detail}") from error
    except APIConnectionError as error:
        raise RuntimeError(
            "Could not connect to Groq. Check your internet connection and Groq service status."
        ) from error
    answer = "".join(answer_chunks).strip()
    if not answer:
        raise RuntimeError("Groq returned an empty answer.")
    return _format_citations(answer, passages)
