"""Offline stand-ins for the Gemini API (no test may reach the real network).

* :class:`LexicalEmbedding` - deterministic hashed bag-of-words embedder, so real ChromaDB
  retrieval still ranks lexically similar chunks first.
* :class:`ScriptedLLM`     - answers every LLM entry point with schema-valid, document-grounded JSON.
* ``gemini_*`` helpers     - real :class:`GeminiLLMService` / :class:`GeminiEmbeddingService` instances
  wired to an ``httpx.MockTransport`` plus builders for Gemini-shaped responses.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, AsyncIterator, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

import httpx

from app.repositories.chroma_repository import ChromaRepository
from app.services.embedding_service import GeminiEmbeddingService
from app.services.llm_service import GeminiLLMService, LLMService

GEMINI_TEST_URL = "http://gemini.test/v1beta"
GEMINI_TEST_KEY = "test-gemini-key-not-real"


def gemini_body(text: str, *, thought: str | None = None) -> dict[str, Any]:
    """A ``generateContent`` response body carrying ``text`` (optionally preceded by a thought part)."""
    parts: list[dict[str, Any]] = [{"text": thought, "thought": True}] if thought else []
    parts.append({"text": text})
    return {"candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]}


def gemini_ok(text: str = "ok") -> httpx.Response:
    return httpx.Response(200, json=gemini_body(text))


def gemini_sse(*texts: str, done: bool = False) -> bytes:
    """``streamGenerateContent?alt=sse`` body: one ``data:`` frame per text piece."""
    frames = [f"data: {json.dumps(gemini_body(t))}\r\n\r\n" for t in texts]
    if done:
        frames.append("data: [DONE]\r\n\r\n")
    return "".join(frames).encode()


def gemini_llm(handler: Any, *, model: str = "gemini-test", backoff: float = 0.0, **kw: Any) -> GeminiLLMService:
    """Real GeminiLLMService talking to ``handler`` through httpx.MockTransport (never the network)."""
    svc = GeminiLLMService(api_key=kw.pop("api_key", GEMINI_TEST_KEY), base_url=GEMINI_TEST_URL, model=model, **kw)
    svc._retry_backoff_s = backoff
    svc._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=GEMINI_TEST_URL,
                                    headers={"x-goog-api-key": svc._api_key} if svc._api_key else {})
    return svc


def gemini_embeddings_body(vectors: list[list[float]]) -> dict[str, Any]:
    return {"embeddings": [{"values": v} for v in vectors]}


def gemini_embedding(handler: Any, *, model: str = "embed-test", api_key: str = GEMINI_TEST_KEY) -> GeminiEmbeddingService:
    """Real GeminiEmbeddingService talking to ``handler`` through httpx.MockTransport."""
    svc = GeminiEmbeddingService(api_key=api_key, base_url=GEMINI_TEST_URL, model=model)
    svc._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=GEMINI_TEST_URL)
    return svc


def embed_request_texts(request: httpx.Request) -> list[str]:
    """Texts of a ``batchEmbedContents`` request (in order)."""
    return [r["content"]["parts"][0]["text"] for r in json.loads(request.content)["requests"]]

_WORD = re.compile(r"[a-z0-9]+")
DIM = 512


def _vector(text: str) -> list[float]:
    vec = [0.0] * DIM
    for word in _WORD.findall(text.lower()):
        if len(word) < 3:
            continue
        idx = int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class LexicalEmbedding:
    model_name = "lexical-test-embedding"

    async def embed_text(self, text: str) -> list[float]:
        return _vector(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vector(t) for t in texts]

    async def close(self) -> None:
        return None


def make_repo(name: str) -> ChromaRepository:
    """Real ChromaDB (in-memory) repository on a uniquely named collection."""
    return ChromaRepository(client=chromadb.Client(ChromaSettings(anonymized_telemetry=False)), collection_name=name)


EVAL_RESULT = {
    "score": 8, "correct": True,
    "feedback": "Your answer captures the main idea of the document.",
    "strengths": ["Identifies the core concept."], "missing_points": ["No example given."],
    "errors": [],
}


class ScriptedLLM(LLMService):
    """Schema-valid outputs; question text reuses words from the document so grounding passes."""

    model_name = "scripted-test-llm"

    def __init__(self) -> None:
        self.stream_calls = 0

    async def generate(self, system_prompt: str, user_prompt: str, **_: Any) -> str:
        return "OK"

    async def generate_json(self, system_prompt: str, user_prompt: str, *,
                            required_keys: Optional[list[str]] = None, **_: Any) -> Any:
        keys = required_keys or []
        if "questions" in keys:
            return {"questions": [{
                "question": "What do computers learn from data in machine learning?",
                "expected_answer": "Computers learn patterns from data to make predictions.",
                "difficulty": "medium", "type": "short_answer",
            }]}
        if "answer" in keys:
            return {"answer": "Machine learning lets computers learn patterns from data."}
        return dict(EVAL_RESULT)

    async def generate_stream(self, system_prompt: str, user_prompt: str, **_: Any) -> AsyncIterator[str]:
        self.stream_calls += 1
        text = json.dumps(EVAL_RESULT)
        for i in range(0, len(text), 12):
            yield text[i:i + 12]

    async def check_health(self) -> dict[str, Any]:
        return {"status": "ok"}


def sample_pdf_bytes(pages: list[str]) -> bytes:
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 545, 790), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


class RecordingLLM(ScriptedLLM):
    """ScriptedLLM that records every prompt pair and lets tests override raw outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, str, str]] = []  # (kind, system_prompt, user_prompt)
        self.eval_payload: dict[str, Any] = dict(EVAL_RESULT)
        self.answer_payload: dict[str, Any] = {"answer": "Machine learning lets computers learn patterns from data."}
        self.questions_payload: Optional[dict[str, Any]] = None

    async def generate(self, system_prompt: str, user_prompt: str, **kw: Any) -> str:
        self.calls.append(("generate", system_prompt, user_prompt))
        return await super().generate(system_prompt, user_prompt, **kw)

    async def generate_json(self, system_prompt: str, user_prompt: str, *,
                            required_keys: Optional[list[str]] = None, **kw: Any) -> Any:
        self.calls.append(("json", system_prompt, user_prompt))
        keys = required_keys or []
        if "questions" in keys and self.questions_payload is not None:
            return self.questions_payload
        if "answer" in keys:
            return self.answer_payload
        if "score" in keys:
            return self.eval_payload
        return await super().generate_json(system_prompt, user_prompt, required_keys=required_keys, **kw)

    async def generate_stream(self, system_prompt: str, user_prompt: str, **kw: Any) -> AsyncIterator[str]:
        self.calls.append(("stream", system_prompt, user_prompt))
        self.stream_calls += 1
        text = json.dumps(self.eval_payload)
        for i in range(0, len(text), 12):
            yield text[i:i + 12]

    @property
    def prompts(self) -> str:
        """Every prompt sent so far, joined (for 'this text never reached the model' checks)."""
        return "\n".join(f"{s}\n{u}" for _, s, u in self.calls)


class Stack:
    """Services wired to a real in-memory ChromaDB with fake embedding + recording LLM."""

    def __init__(self, llm: Optional[RecordingLLM] = None, *, threshold: float = 0.0,
                 max_context_chars: int = 12000, max_top_k: int = 20) -> None:
        import uuid

        from app.services.evaluation_service import EvaluationService
        from app.services.question_service import QuestionGenerationService
        from app.services.rag_service import RAGService
        from app.services.vector_service import VectorService

        self.embedding = LexicalEmbedding()
        self.llm = llm or RecordingLLM()
        self.repo = make_repo(f"stack_{uuid.uuid4().hex[:8]}")
        self.vector = VectorService(embedding_service=self.embedding, chroma_repo=self.repo)
        self.rag = RAGService(embedding_service=self.embedding, chroma_repo=self.repo,
                              llm_service=self.llm, similarity_threshold=threshold,
                              max_context_chars=max_context_chars, max_top_k=max_top_k)
        self.questions = QuestionGenerationService(rag_service=self.rag, llm_service=self.llm)
        self.evaluator = EvaluationService(rag_service=self.rag, llm_service=self.llm)

    def install(self, monkeypatch: Any) -> "Stack":
        monkeypatch.setattr("app.api.documents._get_vector_service", lambda: self.vector)
        monkeypatch.setattr("app.api.rag._get_rag_service", lambda: self.rag)
        monkeypatch.setattr("app.api.questions._get_question_service", lambda: self.questions)
        monkeypatch.setattr("app.api.evaluations._get_evaluation_service", lambda: self.evaluator)
        return self
