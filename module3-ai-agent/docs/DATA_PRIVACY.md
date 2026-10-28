# Data privacy and retention (Module 3)

This describes what the code does today. It does **not** assert compliance with any specific law or policy; no such
requirement was provided. Retention periods below are *behaviour*, not a policy decision.

## Lifecycle

| Stage | Data | Where | Lifetime |
|---|---|---|---|
| Upload | PDF bytes | temp file `m3_pdf_*/<uuid>.pdf` in the OS temp dir | removed in `finally` after the request (also on error) |
| Extract / chunk | page text, chunks | process memory | request lifetime |
| Embed | chunk text sent to the Ollama embedding model | Ollama (local, same host / compose network) | Ollama does not persist prompts by default; not verified beyond that |
| Index | chunk text, embedding, metadata (`document_id`, source filename, page, chunk id) | ChromaDB (`CHROMA_PERSIST_DIRECTORY`, volume `chroma_data`) | until `DELETE /api/documents/{id}` or the volume is removed |
| RAG / questions | question text, retrieved chunks, generated text | memory; prompt sent to Ollama | request lifetime; **not stored** |
| Evaluation | question, student answer, expected answer, result | memory; prompt sent to Ollama | request lifetime; **not stored** by Module 3 (Module 2 owns persistence of results) |
| AI governance (Phase 18) | version manifests, evidence reports, audit trail (actor label, reasons, timestamps) | `AI_VERSIONS_DIR` (volume `ai_versions_data`) | kept until manually removed; contains **no** student or document content, only configuration and metric reports |
| Logs | request id, method, path, status, duration, error code, document id, upload filename | stdout / container logs | per log driver; **no** bodies, answers, prompts, PDF text or tokens (enforced by tests) |
| Metrics | counters / histograms by route and error code | memory | process lifetime |

## Deletion

* `DELETE /api/documents/{id}` removes **all chunks and embeddings** of that document from ChromaDB and nothing
  else (`tests/security/test_isolation_secrets_admin.py`). Afterwards GET / retrieve / evaluate return 404 and no
  model is called.
* Not covered by deletion: **backups** made with `scripts/backup.py` (they contain the chunk text and remain
  until removed by the operator), container/host logs (upload processing logs include the document id and the
  uploader-supplied filename, never document content), and anything Module 1/2 stored.
* There is no automatic expiry: documents stay indexed until deleted. Operators who need a retention period must
  schedule deletions externally.

## Data that should not be stored or logged

Student answers, generated answers, prompts, PDF text, admin tokens, `.env` contents. Raw-payload debug logging
does not exist; do not add it. `LOG_LEVEL=DEBUG` is rejected in production configuration.

## Limitations

* The source filename supplied by the uploader is stored in Chroma metadata and written to upload logs; it may
  itself contain personal data.
* ChromaDB files are not encrypted at rest by this module; use disk/volume encryption if required.
* No per-user data ownership: anyone who can call the API can delete any document whose id they know.
* Ollama is trusted to stay local; Module 3 never sends data to an external LLM service.
