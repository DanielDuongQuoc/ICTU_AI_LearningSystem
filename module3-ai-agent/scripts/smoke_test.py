"""Production smoke test for a running Module 3 (or an in-process ASGI client in tests).

    python scripts/smoke_test.py [--base-url http://localhost:8003]

Flow: health -> ready -> upload -> get -> retrieve -> generate question -> evaluate ->
stream evaluation -> delete -> verify deletion. Uses a generated sample PDF. It needs a live
Ollama (question generation / evaluation call Qwen), so it is NOT part of ``pytest``; the
offline suite runs the same steps against fakes via ``run_smoke``.
Module 1 / Module 2 are not involved (their contracts are unconfirmed): this is a Module 3 smoke test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

import httpx

SAMPLE_TEXT = (
    "Photosynthesis is the process by which green plants convert light energy into chemical energy. "
    "Chlorophyll in the chloroplasts absorbs sunlight. Plants take in carbon dioxide and water, "
    "produce glucose and release oxygen."
)
QUESTION = "What does photosynthesis do?"
EXPECTED = "Plants convert light energy into chemical energy, producing glucose and oxygen."


def sample_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 790), SAMPLE_TEXT, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def _parse_sse(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


async def run_smoke(client: httpx.AsyncClient, *, require_ready: bool = True) -> list[tuple[str, bool, str]]:
    """Run the flow; returns ``[(step, ok, detail)]`` and never raises for step failures."""
    results: list[tuple[str, bool, str]] = []
    state: dict[str, Any] = {}

    def record(step: str, ok: bool, detail: str = "") -> bool:
        results.append((step, ok, detail))
        return ok

    async def step(name: str, coro) -> Any:
        try:
            return await coro
        except Exception as exc:  # network / unexpected: report, keep going where possible
            record(name, False, f"{type(exc).__name__}: {exc}")
            return None

    r = await step("health", client.get("/health"))
    if r is not None:
        record("health", r.status_code == 200 and "status" in r.json(), f"status={r.json().get('status')}")
    r = await step("ready", client.get("/health/ready"))
    if r is not None:
        ready = r.status_code == 200
        record("ready", ready or not require_ready, f"HTTP {r.status_code}")
        if not ready and require_ready:
            return results  # dependencies down: later steps would only produce noise

    r = await step("upload", client.post("/api/documents/upload",
                                         files={"file": ("smoke.pdf", sample_pdf(), "application/pdf")}))
    if r is None or not record("upload", r.status_code in (200, 201), f"HTTP {r.status_code}"):
        return results
    doc = state["doc"] = r.json()["document_id"]
    try:
        r = await step("get", client.get(f"/api/documents/{doc}"))
        if r is not None:
            record("get", r.status_code == 200 and r.json()["document_id"] == doc, f"HTTP {r.status_code}")

        r = await step("retrieve", client.post("/api/rag/retrieve", json={"document_id": doc, "question": QUESTION}))
        if r is not None:
            body = r.json() if r.status_code == 200 else {}
            record("retrieve", r.status_code == 200 and bool(body.get("chunks"))
                   and all(c["document_id"] == doc for c in body["chunks"]), f"HTTP {r.status_code}")

        r = await step("questions", client.post("/api/questions/generate",
                                                json={"document_id": doc, "number_of_questions": 1}))
        if r is not None:
            record("questions", r.status_code == 200 and bool(r.json().get("questions")), f"HTTP {r.status_code}")

        payload = {"document_id": doc, "question": QUESTION, "student_answer": "Plants turn sunlight into sugar.",
                   "expected_answer": EXPECTED, "max_score": 10}
        r = await step("evaluate", client.post("/api/evaluations/evaluate", json=payload))
        if r is not None:
            ok = r.status_code == 200 and 0 <= r.json()["score"] <= 10
            record("evaluate", ok, f"HTTP {r.status_code}")

        r = await step("stream", client.post("/api/evaluations/evaluate/stream", json=payload))
        if r is not None:
            types = [e["type"] for e in _parse_sse(r.text)]
            ok = (r.headers.get("content-type", "").startswith("text/event-stream") and types[-1:] == ["done"]
                  and types.count("done") == 1 and types.count("result") == 1)
            record("stream", ok, "events=" + ",".join(dict.fromkeys(types)))
    finally:
        r = await step("delete", client.delete(f"/api/documents/{doc}"))
        if r is not None:
            record("delete", r.status_code == 200, f"HTTP {r.status_code}")
        r = await step("verify_deleted", client.get(f"/api/documents/{doc}"))
        if r is not None:
            record("verify_deleted", r.status_code == 404, f"HTTP {r.status_code}")
    return results


async def _main(base_url: str) -> int:
    async with httpx.AsyncClient(base_url=base_url, timeout=300) as client:
        results = await run_smoke(client)
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:15s} {detail}")
    return 0 if results and all(ok for _, ok, _ in results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8003")
    return asyncio.run(_main(parser.parse_args(argv).base_url))


if __name__ == "__main__":
    sys.exit(main())
