"""Prompt registry: stable ids, explicit versions and content fingerprints for every prompt task.

The prompt TEXT still lives in :mod:`app.core.prompts` (single source, nothing is duplicated); this
module wraps it with a ``(task, version)`` key, a SHA-256 content fingerprint and a build entry point
so services build prompts through one place and the configuration fingerprint covers prompt text.

Rules enforced here
* A ``(task, version)`` pair is immutable: registering the same pair with different text is an error,
  and :func:`verify_lock` compares the shipped v1 prompts with the committed fingerprints so a prompt
  edited WITHOUT bumping its version is detected (tests + startup log).
* :func:`lint_prompt_texts` is a cheap first filter (injection hierarchy statement, no external
  knowledge, no request for backend-owned fields).  It is NOT a safety proof; the behavioural tests are.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable

from app.core import prompts as P

TASK_RAG_ANSWER = "rag_answer"
TASK_QUESTION_GENERATION = "question_generation"
TASK_ANSWER_EVALUATION = "answer_evaluation"
TASK_RUBRIC_GENERATION = "rubric_generation"
TASK_RUBRIC_REVIEW = "rubric_review"
TASK_RUBRIC_GRADING = "rubric_grading"
TASK_QUESTION_GENERATION_NO_DOC = "question_generation_nodoc"
TASK_HANDWRITING_TRANSCRIPTION = "handwriting_transcription"
TASKS = (TASK_RAG_ANSWER, TASK_QUESTION_GENERATION, TASK_ANSWER_EVALUATION,
         TASK_RUBRIC_GENERATION, TASK_RUBRIC_REVIEW, TASK_RUBRIC_GRADING, TASK_QUESTION_GENERATION_NO_DOC,
         TASK_HANDWRITING_TRANSCRIPTION)

DEFAULT_PROMPT_VERSIONS: dict[str, str] = {
    TASK_RAG_ANSWER: P.PROMPT_VERSION,
    TASK_QUESTION_GENERATION: P.QUESTION_GENERATION_PROMPT_VERSION,
    TASK_ANSWER_EVALUATION: P.EVALUATION_PROMPT_VERSION,
    TASK_RUBRIC_GENERATION: P.RUBRIC_GENERATION_PROMPT_VERSION,
    TASK_RUBRIC_REVIEW: P.RUBRIC_REVIEW_PROMPT_VERSION,
    TASK_RUBRIC_GRADING: P.RUBRIC_GRADING_PROMPT_VERSION,
    TASK_QUESTION_GENERATION_NO_DOC: P.QUESTION_GENERATION_NO_DOC_PROMPT_VERSION,
    TASK_HANDWRITING_TRANSCRIPTION: P.HANDWRITING_TRANSCRIPTION_PROMPT_VERSION,
}

# Committed fingerprints of the shipped prompts. Changing prompt text requires a NEW version
# (and a new entry here), never an edit of an existing one.
PROMPT_LOCK: dict[tuple[str, str], str] = {
    (TASK_RAG_ANSWER, "v1"): "3237d0def1c1f0a29c4e5b2d9dcbb57e50809d7da576bec8ffdeb115a62d0f29",
    (TASK_QUESTION_GENERATION, "v1"): "9d40603f91d40796ece2c741e91aa7ffd8afffd5f2903e0ec1a6e6ef8e4d6135",
    (TASK_ANSWER_EVALUATION, "v1"): "d37c9a3b7943f21efba25c96bea3a8ddd87be94031f403400603b03329f56423",
    (TASK_RUBRIC_GENERATION, "v1"): "69aa16ed6580a0fe94bb94c308ecec2cbe4270a623054f1708085871d4629c30",
    (TASK_RUBRIC_REVIEW, "v1"): "90e2eb93e224702646384a69dc1568be07a5b81de9d4df10e2a85be86b1136ad",
    (TASK_RUBRIC_GRADING, "v1"): "36b407e4abe1c215dd361366c975e4ee16029c1f08b3717349e93d42ee854bf9",
    (TASK_QUESTION_GENERATION_NO_DOC, "v1"): "7bdd7e5b2d12ae5a16bcfb23170faec1f8b298d57a443d4be8897ff38a609e0b",
    (TASK_HANDWRITING_TRANSCRIPTION, "v1"): "2ce3c20f3ae9ef799df46fde5492835764c7e4043e2de89bd4fe413b0173ebeb",
}


class PromptRegistryError(ValueError):
    """Unknown prompt, duplicate registration with different content, or failed lint."""


@dataclass(frozen=True)
class PromptSpec:
    task: str
    version: str
    builder: Callable[..., tuple[str, str]]
    texts: tuple[str, ...]          # every static text that shapes the prompt (fingerprint input)
    system_texts: tuple[str, ...]   # the subset that forms the system prompt (lint input)
    description: str = ""

    @property
    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for part in (self.task, self.version, *self.texts):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()


_REGISTRY: dict[tuple[str, str], PromptSpec] = {}

_REQUIRED = {
    "untrusted input is data, not instructions": re.compile(r"data, never as instructions|untrusted", re.I),
    "no external knowledge": re.compile(r"external knowledge", re.I),
    "backend owns sources/ids": re.compile(
        r"(do not|never)\s+(invent|output|make up|fabricate)|the backend adds", re.I),
}
_FORBIDDEN = {
    "asks to ignore the context": re.compile(r"ignore (the |all )?(context|document)", re.I),
    "asks to use own knowledge": re.compile(r"(use|rely on) (your|its) (own|prior|general) knowledge", re.I),
    "asks for citations/ids from the model": re.compile(
        r"(cite|include|provide|return)\s+(the\s+)?(page|chunk[ _]?id|document[ _]?id|source)", re.I),
    "asks to obey embedded instructions": re.compile(r"follow (the )?instructions (in|inside|from) the (document|context|student)", re.I),
}


def lint_prompt_texts(system_texts: tuple[str, ...] | list[str]) -> list[str]:
    """Return human-readable problems (empty = lint passed)."""
    blob = "\n".join(system_texts)
    problems = [f"missing statement: {name}" for name, rx in _REQUIRED.items() if not rx.search(blob)]
    problems += [f"forbidden instruction: {name}" for name, rx in _FORBIDDEN.items() if rx.search(blob)]
    return problems


def register_prompt(spec: PromptSpec, *, lint: bool = True) -> PromptSpec:
    if spec.task not in TASKS:
        raise PromptRegistryError(f"unknown prompt task {spec.task!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", spec.version):
        raise PromptRegistryError(f"invalid prompt version label {spec.version!r}")
    key = (spec.task, spec.version)
    existing = _REGISTRY.get(key)
    if existing is not None:
        if existing.fingerprint != spec.fingerprint:
            raise PromptRegistryError(
                f"prompt {spec.task}/{spec.version} already registered with different content; "
                "published prompt versions are immutable - register a new version"
            )
        return existing
    if lint:
        problems = lint_prompt_texts(spec.system_texts)
        if problems:
            raise PromptRegistryError(f"prompt {spec.task}/{spec.version} rejected by lint: {'; '.join(problems)}")
    _REGISTRY[key] = spec
    return spec


def content_fingerprint(task: str, version: str) -> str:
    """Fingerprint of a registered prompt, or ``unregistered`` (never raises)."""
    spec = _REGISTRY.get((task, version))
    return spec.fingerprint if spec else "unregistered"


def build(task: str, version: str, *args: Any, **kwargs: Any) -> tuple[str, str]:
    """``(system_prompt, user_prompt)`` for ``task`` at ``version``."""
    try:
        spec = _REGISTRY[(task, version)]
    except KeyError as exc:
        raise PromptRegistryError(f"prompt {task}/{version} is not registered") from exc
    return spec.builder(*args, **kwargs)


def verify_lock() -> list[str]:
    """Differences between registered v1 prompts and the committed fingerprints."""
    out = []
    for key, expected in PROMPT_LOCK.items():
        spec = _REGISTRY.get(key)
        if spec is None:
            out.append(f"{key[0]}/{key[1]}: not registered")
        elif spec.fingerprint != expected:
            out.append(f"{key[0]}/{key[1]}: content changed without a version bump")
    return out


def _register_builtin() -> None:
    register_prompt(PromptSpec(
        TASK_RAG_ANSWER, P.PROMPT_VERSION, P.build_rag_messages,
        texts=(P.RAG_SYSTEM_PROMPT, P.RAG_USER_PROMPT_TEMPLATE),
        system_texts=(P.RAG_SYSTEM_PROMPT,), description="RAG grounded answer"))
    register_prompt(PromptSpec(
        TASK_QUESTION_GENERATION, P.QUESTION_GENERATION_PROMPT_VERSION, P.build_question_generation_prompt,
        texts=(P.QUESTION_GENERATION_SYSTEM_PROMPT, P.QUESTION_GENERATION_USER_PROMPT_TEMPLATE,
               *(f"{k}={v}" for k, v in sorted(P.QUESTION_TYPE_FORMAT_INSTRUCTIONS.items()))),
        system_texts=(P.QUESTION_GENERATION_SYSTEM_PROMPT,), description="Question generation"))
    register_prompt(PromptSpec(
        TASK_ANSWER_EVALUATION, P.EVALUATION_PROMPT_VERSION, P.build_evaluation_prompt,
        texts=(P.EVALUATION_SYSTEM_PROMPT, P.EVALUATION_USER_PROMPT_TEMPLATE),
        system_texts=(P.EVALUATION_SYSTEM_PROMPT,), description="Answer evaluation (grading)"))
    register_prompt(PromptSpec(
        TASK_RUBRIC_GENERATION, P.RUBRIC_GENERATION_PROMPT_VERSION, P.build_rubric_generation_prompt,
        texts=(P.RUBRIC_GENERATION_SYSTEM_PROMPT, P.RUBRIC_GENERATION_USER_PROMPT_TEMPLATE,
               *(f"{k}={v}" for k, v in sorted(P.RUBRIC_ASSESSMENT_NOTES.items()))),
        system_texts=(P.RUBRIC_GENERATION_SYSTEM_PROMPT,), description="Analytic rubric generation"))
    register_prompt(PromptSpec(
        TASK_RUBRIC_REVIEW, P.RUBRIC_REVIEW_PROMPT_VERSION, P.build_rubric_review_prompt,
        texts=(P.RUBRIC_REVIEW_SYSTEM_PROMPT, P.RUBRIC_REVIEW_USER_PROMPT_TEMPLATE),
        system_texts=(P.RUBRIC_REVIEW_SYSTEM_PROMPT,), description="Rubric review (coverage / observability)"))
    register_prompt(PromptSpec(
        TASK_RUBRIC_GRADING, P.RUBRIC_GRADING_PROMPT_VERSION, P.build_rubric_grading_prompt,
        texts=(P.RUBRIC_GRADING_SYSTEM_PROMPT, P.RUBRIC_GRADING_USER_PROMPT_TEMPLATE),
        system_texts=(P.RUBRIC_GRADING_SYSTEM_PROMPT,), description="Rubric-based grading (level per criterion)"))
    register_prompt(PromptSpec(
        TASK_QUESTION_GENERATION_NO_DOC, P.QUESTION_GENERATION_NO_DOC_PROMPT_VERSION,
        P.build_question_generation_no_doc_prompt,
        texts=(P.QUESTION_GENERATION_NO_DOC_SYSTEM_PROMPT, P.QUESTION_GENERATION_NO_DOC_USER_PROMPT_TEMPLATE,
               *(f"{k}={v}" for k, v in sorted(P.QUESTION_TYPE_FORMAT_INSTRUCTIONS.items()))),
        system_texts=(P.QUESTION_GENERATION_NO_DOC_SYSTEM_PROMPT,),
        description="Question generation from course_context only (no document)"))
    register_prompt(PromptSpec(
        TASK_HANDWRITING_TRANSCRIPTION, P.HANDWRITING_TRANSCRIPTION_PROMPT_VERSION,
        P.build_handwriting_transcription_prompt,
        texts=(P.HANDWRITING_TRANSCRIPTION_SYSTEM_PROMPT, P.HANDWRITING_TRANSCRIPTION_USER_PROMPT_TEMPLATE),
        system_texts=(P.HANDWRITING_TRANSCRIPTION_SYSTEM_PROMPT,),
        description="Handwritten answer transcription (vision)"))


_register_builtin()
