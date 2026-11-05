from types import SimpleNamespace

import pytest

from qa_system import (
    HashingEmbeddingModel,
    NOT_FOUND_ANSWER,
    _chunks_for_section,
    _format_citations,
    cosine_similarity,
    create_embedding_model,
    generate_grounded_answer,
    index_document,
    parse_document,
    retrieve_passages,
    split_text,
)
from storage import get_messages, init_db, list_documents, save_exchange


class KeywordEmbeddingModel:
    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "solar" in normalized or "renewable" in normalized or "sunlight" in normalized:
                vectors.append([1.0, 0.0, 0.0])
            elif "battery" in normalized or "lithium" in normalized:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


class FakeGroqChunk:
    def __init__(self, content=None, choices=True):
        self.choices = (
            [SimpleNamespace(delta=SimpleNamespace(content=content))]
            if choices
            else []
        )


class FakeOpenAICompletions:
    def __init__(self, stream, observed):
        self.stream = stream
        self.observed = observed

    def create(self, **kwargs):
        self.observed["request"] = kwargs
        return iter(self.stream)


class FakeOpenAIClient:
    def __init__(self, stream, observed, **client_options):
        observed["client"] = client_options
        self.chat = SimpleNamespace(
            completions=FakeOpenAICompletions(stream, observed)
        )


def install_fake_openai(monkeypatch, stream, observed):
    import openai

    class Client(FakeOpenAIClient):
        def __init__(self, **client_options):
            super().__init__(stream, observed, **client_options)

    monkeypatch.setattr(openai, "OpenAI", Client)


def test_groq_answer_request_is_grounded_and_formats_citations(monkeypatch):
    observed = {}
    install_fake_openai(
        monkeypatch,
        [FakeGroqChunk("Warranty lasts "), FakeGroqChunk("two years. [S1]")],
        observed,
    )
    streamed_chunks = []
    passages = [
        {
            "filename": "warranty.pdf",
            "location": "Page 3",
            "text": "The warranty lasts two years.",
            "score": 0.9,
        }
    ]

    answer = generate_grounded_answer(
        "How long is the warranty?",
        passages,
        model_name="test-groq-model",
        api_key="test-key",
        stream_callback=streamed_chunks.append,
    )

    assert answer == "Warranty lasts two years. [S1: warranty.pdf, Page 3]"
    assert streamed_chunks == ["Warranty lasts ", "two years. [S1]"]
    assert observed["client"] == {
        "api_key": "test-key",
        "base_url": "https://api.groq.com/openai/v1",
        "max_retries": 0,
        "timeout": 60,
    }
    request = observed["request"]
    assert request["model"] == "test-groq-model"
    assert request["temperature"] == 1
    assert request["max_completion_tokens"] == 2048
    assert request["top_p"] == 1
    assert request["reasoning_effort"] == "medium"
    assert request["stream"] is True
    assert request["stop"] is None
    assert "The warranty lasts two years." in request["messages"][0]["content"]


def test_groq_uses_gpt_oss_120b_by_default(monkeypatch):
    observed = {}
    install_fake_openai(monkeypatch, [FakeGroqChunk("Two years. [S1]")], observed)
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    generate_grounded_answer(
        "How long is the warranty?",
        [{"filename": "warranty.txt", "location": "Line 1", "text": "Two years."}],
        api_key="test-key",
    )

    assert observed["request"]["model"] == "openai/gpt-oss-120b"


def test_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        generate_grounded_answer(
            "How long is the warranty?",
            [{"filename": "warranty.txt", "location": "Line 1", "text": "Two years."}],
        )


def test_groq_http_errors_are_reported(monkeypatch):
    class FakeStatusError(Exception):
        status_code = 401
        body = "Unauthorized"

    class FailingCompletions:
        def create(self, **_kwargs):
            raise FakeStatusError("Unauthorized")

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FailingCompletions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setattr(openai, "APIStatusError", FakeStatusError)

    with pytest.raises(RuntimeError, match="Groq returned HTTP 401"):
        generate_grounded_answer(
            "How long is the warranty?",
            [{"filename": "warranty.txt", "location": "Line 1", "text": "Two years."}],
            api_key="invalid",
        )


def test_groq_cloudflare_1010_error_has_network_guidance(monkeypatch):
    class FakeStatusError(Exception):
        status_code = 403
        body = "error code: 1010"

    class FailingCompletions:
        def create(self, **_kwargs):
            raise FakeStatusError("Forbidden")

    class Client:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FailingCompletions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", Client)
    monkeypatch.setattr(openai, "APIStatusError", FakeStatusError)

    with pytest.raises(RuntimeError, match="Cloudflare error 1010"):
        generate_grounded_answer(
            "How long is the warranty?",
            [{"filename": "warranty.txt", "location": "Line 1", "text": "Two years."}],
            api_key="test-key",
        )


def test_missing_cached_model_uses_offline_search(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def import_without_sentence_transformers(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ModuleNotFoundError("No module named 'sentence_transformers'", name=name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sentence_transformers)

    model = create_embedding_model("test-model")

    assert isinstance(model, HashingEmbeddingModel)
    assert model.encode(["solar panels", "battery storage"]).shape == (2, 512)


def test_unavailable_hugging_face_model_falls_back_offline(monkeypatch):
    import sys
    import types

    class UnavailableModel:
        def __init__(self, _model_name, **_kwargs):
            raise OSError("Hugging Face is unreachable")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=UnavailableModel),
    )

    model = create_embedding_model("test-model")

    assert isinstance(model, HashingEmbeddingModel)
    assert "keyword matching" in model.fallback_notice


def test_offline_embedding_model_indexes_and_retrieves(tmp_path):
    db_path = tmp_path / "offline.sqlite3"
    init_db(db_path)
    model = HashingEmbeddingModel()
    index_document(
        db_path,
        "energy.txt",
        b"Solar panels turn sunlight into renewable electricity.",
        model,
    )

    matches = retrieve_passages(db_path, "solar panels", model)

    assert len(matches) == 1
    assert matches[0]["filename"] == "energy.txt"


def test_offline_search_finds_relevant_passage_despite_low_hash_cosine(tmp_path):
    db_path = tmp_path / "offline-relevance.sqlite3"
    init_db(db_path)
    model = HashingEmbeddingModel()
    index_document(
        db_path,
        "handbook.txt",
        b"Warranty claims must be submitted to the service department within thirty days.",
        model,
    )

    matches = retrieve_passages(
        db_path,
        "Where do I submit a warranty claim?",
        model,
    )

    assert matches
    assert matches[0]["filename"] == "handbook.txt"
    assert matches[0]["score"] >= 0.25


def test_offline_search_does_not_match_when_no_query_terms_overlap(tmp_path):
    db_path = tmp_path / "offline-unrelated.sqlite3"
    init_db(db_path)
    model = HashingEmbeddingModel()
    index_document(db_path, "handbook.txt", b"Warranty claims go to the service department.", model)

    matches = retrieve_passages(db_path, "How tall is Mount Everest?", model)

    assert matches == []


def test_offline_search_uses_previous_user_question_for_follow_up(tmp_path):
    db_path = tmp_path / "offline-follow-up.sqlite3"
    init_db(db_path)
    model = HashingEmbeddingModel()
    index_document(db_path, "guide.txt", b"Warranty claims go to the service department.", model)

    matches = retrieve_passages(
        db_path,
        "Where do I submit one?",
        model,
        history=[{"role": "user", "content": "How do I make a warranty claim?"}],
    )

    assert matches
    assert matches[0]["filename"] == "guide.txt"


def test_split_text_chunks_and_overlaps():
    text = "alpha beta gamma delta epsilon zeta eta theta"
    chunks = split_text(text, chunk_size=20, overlap=5)

    assert len(chunks) > 1
    assert chunks[0][2]
    assert chunks[0][1] > chunks[1][0]
    assert "".join(chunk[2] for chunk in chunks)


def test_parse_txt_includes_line_citations():
    passages = parse_document("notes.txt", b"first line\nsecond line\nthird line")

    assert passages == [
        {"text": "first line\nsecond line\nthird line", "location": "Lines 1-3"}
    ]


def test_large_txt_chunk_line_citations_remain_correct():
    text = "a" * 1199 + "\n" + "b" * 1199 + "\n" + "c" * 10

    passages = _chunks_for_section(text, "Text")

    assert [passage["location"] for passage in passages] == [
        "Lines 1-2",
        "Lines 1-2",
        "Lines 2-3",
    ]


def test_unsupported_document_type_is_rejected():
    with pytest.raises(ValueError, match="Only PDF, DOCX, and TXT"):
        parse_document("notes.md", b"text")


def test_index_retrieve_and_persist_conversation(tmp_path):
    db_path = tmp_path / "history.sqlite3"
    init_db(db_path)
    model = KeywordEmbeddingModel()
    assert index_document(
        db_path,
        "energy.txt",
        b"Solar panels convert sunlight into renewable electricity.",
        model,
    )
    assert not index_document(
        db_path,
        "energy.txt",
        b"Solar panels convert sunlight into renewable electricity.",
        model,
    )

    passages = retrieve_passages(
        db_path,
        "How does solar energy produce power?",
        model,
    )
    assert len(passages) == 1
    assert passages[0]["filename"] == "energy.txt"
    assert passages[0]["location"] == "Line 1"
    assert list_documents(db_path)[0]["passage_count"] == 1

    conversation_id = save_exchange(
        db_path,
        None,
        "How does solar energy produce power?",
        "Solar panels turn sunlight into electricity. [S1: energy.txt, Line 1]",
        passages,
    )
    messages = get_messages(db_path, conversation_id)
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["sources"][0]["filename"] == "energy.txt"


def test_index_reports_progress_and_embeds_in_batches(tmp_path):
    class RecordingEmbeddingModel:
        def __init__(self):
            self.batch_sizes = []

        def encode(self, texts, **_kwargs):
            self.batch_sizes.append(len(texts))
            return [[1.0, 0.0] for _ in texts]

    db_path = tmp_path / "history.sqlite3"
    init_db(db_path)
    model = RecordingEmbeddingModel()
    progress = []
    text = "\n".join(f"Passage {index}: " + "x" * 1150 for index in range(70))

    index_document(
        db_path,
        "large.txt",
        text.encode(),
        model,
        progress_callback=lambda stage, completed, total: progress.append(
            (stage, completed, total)
        ),
    )

    assert len(model.batch_sizes) > 1
    assert max(model.batch_sizes) <= 64
    assert progress[0][0] == "Extracting text"
    assert progress[-1][0] == "Saving document index"
    assert progress[-1][1] == progress[-1][2]


def test_retrieval_uses_relevance_threshold(tmp_path):
    db_path = tmp_path / "history.sqlite3"
    init_db(db_path)
    model = KeywordEmbeddingModel()
    index_document(db_path, "energy.txt", b"Solar energy uses sunlight.", model)

    passages = retrieve_passages(
        db_path,
        "What is in this unrelated passage?",
        model,
        min_relevance=0.5,
    )
    assert passages == []


def test_retrieval_requires_reindex_when_embedding_model_changes(tmp_path):
    db_path = tmp_path / "history.sqlite3"
    init_db(db_path)
    model = KeywordEmbeddingModel()
    index_document(db_path, "energy.txt", b"Solar energy uses sunlight.", model)

    with pytest.raises(RuntimeError, match="Remove and re-upload"):
        retrieve_passages(db_path, "solar power", HashingEmbeddingModel())


def test_citations_are_expanded_to_verifiable_sources():
    passages = [
        {
            "filename": "guide.pdf",
            "location": "Page 4",
            "text": "A verified passage.",
            "score": 0.8,
        }
    ]

    assert _format_citations("The process takes two days. [S1]", passages) == (
        "The process takes two days. [S1: guide.pdf, Page 4]"
    )
    assert _format_citations("Answer not found in the provided documents.", passages) == (
        NOT_FOUND_ANSWER
    )


def test_cosine_similarity_rejects_different_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        cosine_similarity([1.0, 0.0], [1.0])
