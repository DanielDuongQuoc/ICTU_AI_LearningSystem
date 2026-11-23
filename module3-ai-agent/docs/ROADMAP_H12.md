# LỘ TRÌNH 6 TIẾN ĐỘ NÂNG CẤP MODULE 3 THEO ĐỀ CƯƠNG H12

> Tài liệu quản lý tiến độ triển khai dự án **Module 3 — AI Agent** bám sát Đề cương hướng dẫn **H12. AI Agent Development** (GVHD: TS. Nguyễn Toàn Thắng).

## 1. Tổng quan 6 Tiến độ

| Tiến độ | Tên tiến độ | Chỉ số PI mục tiêu | Trạng thái | Kết quả cốt lõi |
|---|---|---|---|---|
| **Tiến độ 1** | Nền tảng Agent Framework, Model Adapter, Ranh giới Vai trò & Bảo vệ Chế độ Thi | PI 1.1, 1.2, 1.3, 2.1, 2.2 (nền), 3.1 | ✅ **ĐÃ HOÀN THÀNH** | Tích hợp `langgraph==1.2.24`, gói `app/h12_agent/`, phân biệt tường minh `mock` vs `live`, khóa cứng `classroom_exam`, cấm trắc nghiệm/voice, Service Key auth, `InstitutionalRef`, `IdempotencyStore`, 2 endpoint mới (`/api/agent/capabilities`, `/api/agent/execute-skeleton`) & bộ test `tests/test_h12_milestone1.py`. |
| **Tiến độ 2** | Authorized LCMS Tool Layer, Versioned Lesson Knowledge & Isolated Session Memory | PI 2.2, 2.3, hạ tầng 3.3 | ✅ **ĐÃ HOÀN THÀNH** | Xây `LCMSToolClient` (`MockLcmsToolClient` + `HttpLcmsToolClient`) & `AuthorizedToolExecutor`, phiên bản hóa tri thức trên ChromaDB (`lesson_id`, `knowledge_version`, `visibility: student_safe | teacher_only` chống lộ đáp án mẫu), `SessionMemoryStore` SQLite cô lập theo `(student_id, lesson_id, attempt_id)` với retention & context window policy, 4 endpoint mới và bộ test `tests/test_h12_milestone2.py`. |
| **Tiến độ 3** | Teacher Question Drafting & Knowing Fixed Essay Grading Support | PI 3.2, 3.3, 3.4 (Tuần 5–6) | ✅ **ĐÃ HOÀN THÀNH** | Chuẩn hóa 2 workflow trên LangGraph (`question_drafting.py`, `knowing_grading.py`): soạn câu hỏi **tự luận** bám bài học + bàn giao nháp qua `record_question_draft`; chấm bài tự luận cố định Knowing theo rubric đã duyệt trả về `draft_suggestion` (`pending_teacher_review` / `needs_teacher_escalation`, điểm tiêu chí + dẫn chứng + `teacher_handoff=true`), bỏ tự chốt điểm ngầm; bổ sung `tests/test_h12_milestone3.py`. |
| **Tiến độ 4** | Understanding Adaptive Text Chat & Bảo mật Hội thoại | PI 3.2, 3.3, 3.4 (Tuần 7) | ✅ **ĐÃ HOÀN THÀNH** | Xây bộ bảo vệ hội thoại `app/h12_agent/dialogue_guards.py` và workflow LangGraph rẽ nhánh có điều kiện 7 node (`app/h12_agent/understanding_dialogue.py`), mở rộng `DeterministicMockModelAdapter`, 3 endpoint `/api/agent/understanding/sessions/{start,turns,close}`, vượt qua đủ 4 kịch bản PI 3.4 (`redirect`, `reveal_answers`, `alter_scores`, `continue_after_closure`) cùng cô lập chéo sinh viên & khóa thi `classroom_exam` giữa phiên (`tests/test_h12_milestone4.py`). |
| **Tiến độ 5** | Đồng bộ tích hợp với LCMS (Thưởng) & Desktop App (Kiên) | PI 3.3, 4.2 (Tuần 8) | ✅ **ĐÃ HOÀN THÀNH** | Chuẩn hóa `HttpLcmsToolClient` (camelCase ASP.NET Core, ProblemDetails error mapping, cached singleton + shutdown cleanup, CORS headers), xây LCMS HTTP Simulator (`app/h12_agent/lcms_simulator.py`), viết tài liệu & bản vá đồng bộ C# (`docs/INTEGRATION_SYNC_LCMS_DESKTOP.md`) cho `backend_api` & `student_desktop_app`, và hoàn thiện kiểm thử E2E qua HTTP (`tests/test_h12_milestone5.py`) phủ `cross-session isolation`, `classroom-mode denial`, `no silent grade publication`, auth & availability. |
| **Tiến độ 6** | Bộ đánh giá 6 tiêu chí H12, Triển khai & Hoàn thiện Khóa luận | PI 4.1, 4.2 (Tuần 9–10) | ✅ **ĐÃ HOÀN THÀNH** | Xây bộ dữ liệu đa bài học `tests/golden/h12_eval_dataset.json` (36 ca) & `scripts/evaluate_h12.py` đo đủ 6 tiêu chí PI 4.1 (`reports/h12_evaluation_offline.{md,json}` đạt 100% toàn bộ 6 tiêu chí), bổ sung cấu hình lưu phiên Docker (`Dockerfile`, `docker-compose.yml`), cập nhật `SECURITY.md`, `README.md`, tài liệu khóa luận 4 chương (`docs/THESIS_H12_GUIDE.md`) & bộ kiểm thử `tests/test_h12_milestone6.py`. |

---

## 2. Chi tiết những gì đã hoàn thành ở Tiến độ 1

1. **Tích hợp Agent Framework chính thức (PI 3.1):**
   - Ghim `langgraph==1.2.24` trong `requirements.txt` và cài đặt vào `.venv`.
   - Xây dựng đồ thị trạng thái có kiểm soát 5 nút bằng `langgraph.graph.StateGraph` tại `app/h12_agent/graph.py`:
     `validate_mode_and_scope` → `select_allowed_tools` → `retrieve_grounded_evidence` → `invoke_model_adapter` → `validate_draft_boundary`.
2. **Replaceable Model Adapter tách biệt rõ `mock` vs `live` (PI 3.1):**
   - `app/h12_agent/model_adapter.py` định nghĩa `AgentModelAdapter` với 2 bản cài đặt `DeterministicMockModelAdapter` (đầu ra luôn gắn `[MOCK]`, không gọi mạng) và `LiveGeminiModelAdapter` (bọc `LLMService`).
   - Cấu hình qua biến môi trường `AGENT_MODEL_ADAPTER=gemini|mock`, trả nhãn rõ ràng trong `trace.adapter_mode`, `trace.is_mock` và response headers `X-AI-Adapter-Mode`, `X-AI-Framework`.
3. **Ranh giới Vai trò, Chế độ Thi & Định dạng đánh giá (PI 1.1, PI 1.2, PI 1.3):**
   - `app/h12_agent/policy.py` tách rõ 3 vai trò (`question_drafting`, `grading_support`, `dialogue_assessment`), `allowed_scopes`, `allowed_modes`, `allowed_tools`, `visibility_scope` (`teacher_authorized` vs `student_safe`) và `stopping_conditions`.
   - Chặn cứng chế độ thi có kiểm soát `classroom_exam` với HTTP `403 ASSESSMENT_MODE_FORBIDDEN` trước khi chạm tới RAG hay LLM.
   - Chặn thay thế bằng trắc nghiệm (`multiple_choice`, `mcq`, `true_false`, `quiz`) và kênh giọng nói (`voice`, `audio`, `speech`, `oral`) với HTTP `422 UNSUPPORTED_ASSESSMENT_FORMAT`.
   - Cố định ranh giới đầu ra: mọi kết quả của Agent đều mang `publication_status="draft_suggestion"`, `released=false`, `teacher_review_required=true`.
4. **Định danh ổn định, Xác thực dịch vụ & Idempotency (PI 2.2):**
   - `InstitutionalRef` mang các định danh nghiệp vụ ổn định (`lesson_id`, `lesson_version`, `course_id`, `assignment_id`, `rubric_id`, `student_id`, `attempt_id`).
   - Hỗ trợ xác thực `X-Agent-Service-Key` qua `hmac.compare_digest` (`401 AGENT_UNAUTHORIZED` khi `AGENT_AUTH_REQUIRED=true`).
   - `app/h12_agent/idempotency.py` xử lý gửi lặp theo `Idempotency-Key` + SHA-256 payload fingerprint (`X-Idempotent-Replay: true` khi khớp, `409 IDEMPOTENCY_KEY_CONFLICT` khi lệch payload).
5. **Tài liệu hợp đồng & Kiểm thử:**
   - Mở `GET /api/agent/capabilities` và `POST /api/agent/execute-skeleton` (`app/api/agent_h12.py`), cập nhật `docs/INTEGRATION_CONTRACT.md`, `scripts/release_check.py`, `tests/test_api_contract.py` và bổ sung bộ test `tests/test_h12_milestone1.py`.

---

## 3. Chi tiết những gì đã hoàn thành ở Tiến độ 2

1. **Tầng Authorized LCMS Tool Layer & Tool Executor (PI 2.1, PI 2.2, PI 3.3):**
   - `app/h12_agent/lcms_tools.py` định nghĩa `LCMSToolClient` với 2 bản cài đặt:
     - `MockLcmsToolClient`: seed sẵn `lesson-db-01` (`v1` & `v2`), `lesson-se-02` (`v1`), bài chưa duyệt `lesson-draft-99` (`approved=False`), và sinh viên đang thi `SV_IN_EXAM_01`; hỗ trợ 3 công cụ ghi nháp `record_question_draft`, `record_grading_suggestion`, `record_dialogue_outcome` có idempotency và luôn trả `publication_status="draft_suggestion"`, `released=False`.
     - `HttpLcmsToolClient`: gọi HTTP tới `AGENT_LCMS_BASE_URL` kèm `X-Agent-Service-Key`, `X-Request-ID`, `Idempotency-Key` và ánh xạ lỗi chuẩn (`503 LCMS_UNAVAILABLE`, `502 LCMS_TOOL_ERROR`, `404 LESSON_NOT_FOUND`, `409 IDEMPOTENCY_KEY_CONFLICT`, `422 UNAPPROVED_KNOWLEDGE`).
   - Áp dụng sanitization đa tầng: khi `visibility_scope == "student_safe"`, gỡ bỏ toàn bộ `teacher_only_notes`, `answer_key`, và rubric `key_points`.
   - `app/h12_agent/tool_executor.py` (`AuthorizedToolExecutor`): kiểm soát chặt tool được phép theo `RolePolicy`, chặn ngay `classroom_exam` từ LCMS (`403 ASSESSMENT_MODE_FORBIDDEN`) và ghi `ToolOutcome`.
2. **Tri thức bài học có phiên bản & Chống rò rỉ đáp án mẫu (PI 2.3):**
   - Mở rộng `app/repositories/chroma_repository.py` (`query_lesson`, `delete_lesson_version`, `list_lesson_versions` dùng `$and`) và `app/services/vector_service.py` (`index_chunks(..., extra_metadata=...)`).
   - `app/h12_agent/knowledge.py` (`LessonKnowledgeService`): chỉ chấp nhận học liệu đã duyệt (`approved=True`, ngược lại `422 UNAPPROVED_KNOWLEDGE`), phân vùng chunk theo `lesson_id`, `knowledge_version` và `visibility` (`student_safe` vs `teacher_only`), kiểm tra phòng thủ nhiều lớp khi truy xuất.
3. **Bộ nhớ phiên cô lập SQLite & Chính sách Ngữ cảnh/Lưu giữ (PI 2.3):**
   - `app/h12_agent/session_store.py` (`SessionMemoryStore`): lưu bền vững bằng SQLite (`h12_sessions`, `h12_session_turns`, `h12_tool_audit`), khóa cô lập tuyệt đối theo `(student_id, lesson_id, attempt_id, assessment_mode)`.
   - Chặn đọc chéo sinh viên/bài học/lượt (`403 AGENT_SCOPE_FORBIDDEN`), chặn ghi vào phiên đã đóng hoặc quá `max_turns` (`409 SESSION_STATE_CONFLICT`), tóm tắt lượt cũ khi vượt `AGENT_CONTEXT_WINDOW_TURNS`, cắt giới hạn `AGENT_SESSION_MAX_CONTEXT_CHARS`, và dọn phiên quá hạn bằng `purge_expired()`.
4. **Endpoints mới & Kiểm thử:**
   - Mở 4 endpoint trong `app/api/agent_h12.py`: `POST /api/agent/knowledge/index`, `POST /api/agent/knowledge/retrieve`, `POST /api/agent/sessions`, `GET /api/agent/sessions/{session_id}`.
   - Cập nhật `docs/INTEGRATION_CONTRACT.md`, `scripts/release_check.py`, `tests/test_api_contract.py` và bổ sung bộ kiểm thử `tests/test_h12_milestone2.py`.

---

## 4. Chi tiết những gì đã hoàn thành ở Tiến độ 3

1. **Workflow 1 — Teacher Lesson-Grounded Essay Question Drafting (`app/h12_agent/question_drafting.py`):**
   - Đồ thị LangGraph 6 nút (`validate_mode_and_scope` → `fetch_authorized_lesson` → `retrieve_lesson_evidence` → `draft_essay_questions` → `handoff_drafts_to_lcms` → `finalize_draft_boundary`).
   - Ép định dạng `essay` (cấm trắc nghiệm/MCQ), kéo bài học đã duyệt từ LCMS, tự đồng bộ & truy xuất ChromaDB theo phiên bản (`422 INSUFFICIENT_CONTEXT` và 0 lần gọi LLM khi thiếu ngữ cảnh), loại bỏ item có `options` hoặc trùng lặp, gắn `draft_id` + `evidence_refs` từ backend và bàn giao bản nháp qua `record_question_draft`.
2. **Workflow 2 — Knowing Fixed Essay Grading Support (`app/h12_agent/knowing_grading.py`):**
   - Đồ thị LangGraph 7 nút (`validate_mode_and_scope` → `verify_assessment_context` → `fetch_lesson_and_rubric` → `retrieve_lesson_evidence` → `evaluate_essay_against_rubric` → `record_grading_suggestion` → `finalize_handoff_boundary`).
   - Kiểm tra ngữ cảnh thi từ LCMS (chặn `SV_IN_EXAM_01` với `403 ASSESSMENT_MODE_FORBIDDEN`), kiểm duyệt rubric (`RUBRIC_NOT_APPROVED`, `validate_rubric` `R1..R6`), xử lý bài trống/chép đề không cần gọi LLM, bắt Prompt Injection (`needs_teacher_escalation` + `injection_suspected`), tính toàn bộ điểm gợi ý từ bảng rubric bằng Python, ghi gợi ý qua `record_grading_suggestion` và trả về `pending_teacher_review` / `needs_teacher_escalation` với `released=False`, `teacher_handoff=True`.
3. **Mở rộng Mock Model Adapter & Endpoints:**
   - Nâng cấp `DeterministicMockModelAdapter` (`app/h12_agent/model_adapter.py`) để hỗ trợ cả sinh đề tự luận `[MOCK]` và gợi ý chấm rubric `[MOCK]` bám theo tiêu chí thực.
   - Mở `POST /api/agent/questions/draft` và `POST /api/agent/knowing/grade-suggestion` tại `app/api/agent_h12.py`, đồng bộ hợp đồng API & bổ sung bộ kiểm thử `tests/test_h12_milestone3.py`.

---

## 5. Chi tiết những gì đã hoàn thành ở Tiến độ 4

1. **Lớp bảo mật hội thoại Understanding (`app/h12_agent/dialogue_guards.py`):**
   - Cài đặt phát hiện và chặn 4 nhóm tấn công/tín hiệu hội thoại PI 3.4: `detect_redirect_attempt`, `detect_answer_reveal_attempt`, `detect_score_alter_attempt`, `detect_student_closure_request`, và `inspect_student_message` (`block_rubric_progress=True` không cho nâng tiêu chí khi có tấn công).
   - Cài đặt `guard_assistant_reply` kiểm tra `enforce_no_prompt_leak`, trùng lắp shingle `SHINGLE_WORDS` với `system_prompt` (`400/502 LLM_OUTPUT_REJECTED`), và chặn/thay thế an toàn nếu phản hồi vô tình lộ marker đáp án mẫu (`answer_leak_prevented=True`).
2. **Workflow Understanding rẽ nhánh có điều kiện trên LangGraph (`app/h12_agent/understanding_dialogue.py`):**
   - Triển khai `start_understanding_session`, đồ thị rẽ nhánh có điều kiện (`validate_mode_and_session` → `fetch_safe_lesson_and_rubric` → `retrieve_student_safe_evidence` → `inspect_security_and_evaluate_turn` → `decide_next_action` → `close_and_record_outcome` | `finalize_active_turn`) và `close_understanding_session`.
   - Khóa chặt ngữ cảnh `student_safe` (không bao giờ tải `teacher_only_notes` hay `answer_key`), kiểm tra thời gian thực trạng thái `classroom_exam`, cô lập sở hữu phiên theo `student_id`, giữ nguyên trạng thái phiên `active` khi LLM trả JSON lỗi (`502 EVALUATION_INVALID_RESPONSE`), và bàn giao `DialogueOutcomeRecordInput` về LCMS (`teacher_review_required=true`, `released=false`) khi đạt một trong các điều kiện dừng (`criteria_covered`, `max_turns_reached`, `student_requested_closure`, `repeated_injection_escalation`).
3. **Mở rộng Adapter, Endpoints & Bộ kiểm thử (`tests/test_h12_milestone4.py`):**
   - Bổ sung `_mock_understanding_turn` vào `DeterministicMockModelAdapter` (`app/h12_agent/model_adapter.py`).
   - Mở 3 endpoint mới (`POST /api/agent/understanding/sessions/start`, `POST /api/agent/understanding/sessions/{session_id}/turns` hỗ trợ `Idempotency-Key`, `POST /api/agent/understanding/sessions/{session_id}/close`) tại `app/api/agent_h12.py`, đồng bộ `INTEGRATION_CONTRACT.md`, `release_check.py`, `test_api_contract.py` và bổ sung `tests/test_h12_milestone4.py`.

---

## 6. Chi tiết những gì đã hoàn thành ở Tiến độ 5

1. **Chuẩn hóa cầu nối HTTP với LCMS ASP.NET Core (`app/h12_agent/lcms_tools.py`, `app/main.py`):**
   - Bổ sung `_CAMEL_TO_SNAKE_KEYS` và `_normalize_lcms_keys` để `HttpLcmsToolClient` đọc chuẩn xác phản hồi JSON `camelCase` (`System.Text.Json`) từ C# LCMS cũng như `snake_case`, không làm biến dạng dữ liệu tiêu chí rubric.
   - Mở rộng `_raise_for_http_error` hỗ trợ cả lỗi chuẩn H12 và RFC 7807 ASP.NET Core ProblemDetails (`detail`, `title`, `code`).
   - Cache singleton `HttpLcmsToolClient` trong `get_lcms_tool_client()`, tự làm mới khi cấu hình đổi và đóng cleanly qua `close_lcms_tool_client()` trong `_close_shared_clients` (`app/main.py`).
   - Mở rộng `CORSMiddleware` cho `X-Agent-Service-Key`, `Idempotency-Key` (`allow_headers`) và `X-AI-Adapter-Mode`, `X-AI-Framework`, `X-Idempotent-Replay` (`expose_headers`).
2. **Documented LCMS HTTP Simulator (`app/h12_agent/lcms_simulator.py`):**
   - Cài đặt ứng dụng FastAPI độc lập (`create_lcms_simulator_app`) mô phỏng đủ 6 internal LCMS endpoints, hỗ trợ xác thực `X-Agent-Service-Key`, `Idempotency-Key`, cờ `camel_case=True` và CLI runner (`python -m app.h12_agent.lcms_simulator`).
3. **Tài liệu đồng bộ `backend_api` / `student_desktop_app` & Kiểm thử E2E (`docs/INTEGRATION_SYNC_LCMS_DESKTOP.md`, `tests/test_h12_milestone5.py`):**
   - Đối chiếu mã nguồn C# trên nhánh `ai-integration` và cung cấp bản vá chi tiết: chặn `oral-chat`/`ai-chat` khi sinh viên đang `InExam`/`InProgress`, bỏ auto-final điểm `Result = "Pass"/"Fail"` trong `HomeworkAiGrading.cs`, nối `Module3Client.cs` và thay `SendOralChatAsync` trên `ExamForm.cs`/`ApiClient.cs` của Desktop App.
   - Bổ sung `tests/test_h12_milestone5.py` chạy E2E qua HTTP (`HttpLcmsToolClient` ↔ LCMS Simulator) cho cả 3 workflow, kiểm chứng `cross-session isolation`, `classroom-mode denial`, `no silent grade publication`, auth `401`, lỗi `503 LCMS_UNAVAILABLE`, ProblemDetails, không rò rỉ `GEMINI_API_KEY`, CORS và lifecycle singleton.

---

## 7. Chi tiết những gì đã hoàn thành ở Tiến độ 6

1. **Tập Golden Dataset & Bộ đánh giá 6 tiêu chí H12 (`tests/golden/h12_eval_dataset.json`, `scripts/evaluate_h12.py`):**
   - Bổ sung `MockLcmsToolClient.register_lesson(...)` (`app/h12_agent/lcms_tools.py`) và xây dựng tập mẫu đa bài học `tests/golden/h12_eval_dataset.json` (`v1.0.0`, 36 ca đánh giá phủ 4 phiên bản bài học đã duyệt + 1 bài chưa duyệt) cùng ngưỡng chấp nhận tường minh cho đủ 6 tiêu chí PI 4.1.
   - Viết `scripts/evaluate_h12.py` (chế độ `--mode mock` và `--mode gemini`) chạy trực tiếp qua HTTP giữa FastAPI Module 3 và LCMS Simulator `camelCase`, xuất báo cáo Markdown/JSON tại `reports/h12_evaluation_offline.md` và `reports/h12_evaluation_offline.json`.
   - Khắc phục 2 lỗi nhỏ phát hiện khi chạy evaluator trên `DeterministicMockModelAdapter` (`model_adapter.py`: lọc dòng header `[Source: ...]` trước khi tách câu và loại bỏ stopword tiếng Anh 3 chữ cái khi đối chiếu tiêu chí rubric).
2. **Triển khai Container, Bảo mật & Tài liệu Khóa luận (`Dockerfile`, `docker-compose.yml`, `docs/SECURITY.md`, `README.md`, `docs/THESIS_H12_GUIDE.md`):**
   - Cập nhật `Dockerfile` và `docker-compose.yml` để khai báo `AGENT_SESSION_DB_PATH=/app/data/sessions/h12_sessions.db` và volume bền vững `session_data`.
   - Cập nhật `docs/SECURITY.md` (`H12 AI Agent Security & Boundaries`), `README.md` (`8. H12 AI Agent Module`), viết tài liệu tổng hợp 4 chương khóa luận & 12 PI tại `docs/THESIS_H12_GUIDE.md`, và thêm `tests/test_h12_milestone6.py`.

---

## 8. Tổng kết Mức độ Hoàn thiện 12 Chỉ số PI (`PI 1.1` → `PI 4.2`)

Toàn bộ 6 tiến độ trong lộ trình H12 đã hoàn tất trong `module3-ai-agent` (nhánh `ai-module`), bao phủ đầy đủ 12 chỉ số chuẩn đầu ra PI 1.1 → PI 4.2 với bằng chứng kiểm thử tự động offline (`tests/test_h12_milestone1..6.py` + `scripts/evaluate_h12.py` + `scripts/release_check.py --run-tests`) và bộ patch đồng bộ C# sẵn sàng cho `backend_api` & `student_desktop_app` trên nhánh `ai-integration` (`docs/INTEGRATION_SYNC_LCMS_DESKTOP.md`). Chi tiết ánh xạ từng PI xem tại `docs/THESIS_H12_GUIDE.md`.
