# Tài liệu Tổng hợp Phục vụ Khóa luận H12 (AI Agent Development)

> **Đề tài:** *Development of an AI Agent Module for an LMS/LCMS Supporting Blended Learning at the Academy of International Education*  
> **Sinh viên:** Dương Quốc Toản (`DTC235210055`) — **GVHD:** TS. Nguyễn Toàn Thắng  
> Tài liệu này ánh xạ trực tiếp toàn bộ **4 chương khóa luận** và **12 chỉ số chuẩn đầu ra (PI 1.1 → PI 4.2)** tới mã nguồn, API, kiểm thử và số liệu thực nghiệm đã kiểm chứng trong `module3-ai-agent`.

---

## 1. Ánh xạ 4 Chương Khóa luận

| Chương khóa luận | Nội dung trọng tâm | Tài liệu & Mã nguồn đối chiếu trong `module3-ai-agent` |
|---|---|---|
| **Chương 1 — Tổng quan & Phân tích bài toán** | Tách vai trò AI Agent (`question_drafting`, `grading_support`, `dialogue_assessment`) khỏi LCMS; chuẩn hóa Knowing = tự luận cố định, Understanding = hội thoại văn bản thích ứng; khóa cứng `classroom_exam`; mọi đầu ra chỉ là `draft_suggestion`. | `PHAN_TICH_DOI_CHIEU_DE_CUONG_H12.md`, `app/h12_agent/types.py`, `app/h12_agent/policy.py`, `app/grading/policy.py` |
| **Chương 2 — Thiết kế Kiến trúc Agent, Tool Layer, Tri thức & Bộ nhớ Phiên** | Kiến trúc phân tầng Gateway → LangGraph Orchestrator → `AuthorizedToolExecutor` → `LCMSToolClient` (`mock` \| `http`) → `LessonKnowledgeService` (`lesson_id`, `knowledge_version`, `student_safe` vs `teacher_only`) → SQLite `SessionMemoryStore` cô lập theo `(student_id, lesson_id, attempt_id)`. | `app/h12_agent/lcms_tools.py`, `app/h12_agent/tool_executor.py`, `app/h12_agent/knowledge.py`, `app/h12_agent/session_store.py`, `app/h12_agent/idempotency.py`, `app/repositories/chroma_repository.py`, `docs/INTEGRATION_CONTRACT.md` |
| **Chương 3 — Hiện thực hóa Workflow & Bảo mật Hội thoại** | Cài đặt `langgraph==1.2.24`, `AgentModelAdapter` (`DeterministicMockModelAdapter` vs `LiveGeminiModelAdapter`), 3 workflow có kiểm soát: Question Drafting, Knowing Grading Support và Understanding Adaptive Text Chat 7 node rẽ nhánh có điều kiện + `dialogue_guards.py` (chống redirect, lộ đáp án, sửa điểm, chat sau đóng phiên). | `app/h12_agent/graph.py`, `app/h12_agent/model_adapter.py`, `app/h12_agent/question_drafting.py`, `app/h12_agent/knowing_grading.py`, `app/h12_agent/understanding_dialogue.py`, `app/h12_agent/dialogue_guards.py`, `app/api/agent_h12.py`, `docs/SECURITY.md` |
| **Chương 4 — Tích hợp, Đánh giá Thực nghiệm & Triển khai** | Cầu nối ASP.NET Core `camelCase` + `lcms_simulator.py`, tài liệu & bản vá đồng bộ `backend_api` / `student_desktop_app`, bộ đánh giá 6 tiêu chí `scripts/evaluate_h12.py`, bộ test `tests/test_h12_milestone1..6.py`, đóng gói Docker (`Dockerfile`, `docker-compose.yml`). | `app/h12_agent/lcms_simulator.py`, `docs/INTEGRATION_SYNC_LCMS_DESKTOP.md`, `tests/golden/h12_eval_dataset.json`, `scripts/evaluate_h12.py`, `reports/h12_evaluation_offline.md`, `Dockerfile`, `docker-compose.yml` |

---

## 2. Bảng Đối chiếu Đủ 12 Chỉ số PI (`PI 1.1` → `PI 4.2`)

| PI | Trọng số | Cách hệ thống đáp ứng | Bằng chứng kiểm thử / File |
|---|---:|---|---|
| **PI 1.1** | 5% | Chỉ định rõ vai trò Agent/Tool soạn đề bám bài, gợi ý chấm Knowing tự luận (`essay`) và chat Understanding (`text_chat`); từ chối MCQ/quiz & kênh `voice`/`oral` (`422 UNSUPPORTED_ASSESSMENT_FORMAT`). | `app/h12_agent/policy.py`, `tests/test_h12_milestone1.py` |
| **PI 1.2** | 5% | Mô hình hóa trọn vẹn luồng GV & SV gồm phục hồi lỗi (`EVALUATION_INVALID_RESPONSE`) và khóa cứng `classroom_exam` cả lúc bắt đầu lẫn giữa phiên (`403 ASSESSMENT_MODE_FORBIDDEN`). | `app/h12_agent/knowing_grading.py`, `understanding_dialogue.py`, `tests/test_h12_milestone4.py`, `tests/test_h12_milestone5.py` |
| **PI 1.3** | 10% | Phân tách rõ `question_drafting`, `grading_support`, `dialogue_assessment`, allowed tools, stopping conditions (`criteria_covered`, `max_turns_reached`, `student_requested_closure`, `repeated_injection_escalation`), luôn trả `publication_status="draft_suggestion"`, `released=false`. | `app/h12_agent/policy.py`, `tests/test_h12_milestone3.py` |
| **PI 2.1** | 15% | Tách bạch lớp Gateway, Workflow Orchestration, Authorized Tool Execution, Knowledge/Session Store và Model/LCMS Adapters (`mock` & `live`/`http`). | `app/h12_agent/`, `docs/INTEGRATION_CONTRACT.md` |
| **PI 2.2** | 10% | `InstitutionalRef` ổn định, xác thực `X-Agent-Service-Key` (`401`), scope check (`403`), `Idempotency-Key` SHA-256 (`409` khi lệch payload), cấm tự công bố điểm. | `app/h12_agent/idempotency.py`, `app/h12_agent/lcms_tools.py`, `tests/test_h12_milestone2.py` |
| **PI 2.3** | 5% | ChromaDB theo `(lesson_id, knowledge_version, visibility)` và SQLite `SessionMemoryStore` cô lập `(student_id, lesson_id, attempt_id)` kèm chính sách cửa sổ ngữ cảnh & retention; ngăn rò rỉ `teacher_only_notes` / `answer_key`. | `app/h12_agent/knowledge.py`, `session_store.py`, `tests/test_h12_milestone2.py`, `tests/test_h12_milestone5.py` |
| **PI 3.1** | 5% | Chọn `langgraph==1.2.24` (state machine tường minh), gắn nhãn `adapter_mode` / `is_mock` trên mọi `AgentTrace` và response header (`X-AI-Adapter-Mode`, `X-AI-Framework`). | `app/h12_agent/graph.py`, `model_adapter.py`, `tests/test_h12_milestone1.py` |
| **PI 3.2** | 5% | Structured prompts + JSON schema validation cho question drafts, Knowing rubric grading và Understanding turns; retry có kiểm soát và trả `502` fail-closed khi model lỗi. | `app/h12_agent/question_drafting.py`, `knowing_grading.py`, `understanding_dialogue.py` |
| **PI 3.3** | 20% | `AuthorizedToolExecutor` + `HttpLcmsToolClient` / `MockLcmsToolClient` + `create_lcms_simulator_app` kết nối đầy đủ 3 workflow qua HTTP `camelCase`. | `app/h12_agent/tool_executor.py`, `lcms_simulator.py`, `tests/test_h12_milestone5.py` |
| **PI 3.4** | 10% | Đồ thị LangGraph rẽ nhánh có điều kiện, ghi `observable_steps` & `h12_tool_audit`, vượt qua đủ 4 bài test bảo mật: *redirect*, *reveal answers*, *alter scores*, *continue after closure*. | `app/h12_agent/dialogue_guards.py`, `understanding_dialogue.py`, `tests/test_h12_milestone4.py` |
| **PI 4.1** | 5% | Tập dữ liệu đa bài học `tests/golden/h12_eval_dataset.json` (36 ca, 4 phiên bản bài học đã duyệt + 1 bài chưa duyệt) & `scripts/evaluate_h12.py` đo đủ 6 tiêu chí, bảo mật & cô lập. | `scripts/evaluate_h12.py`, `reports/h12_evaluation_offline.md`, `tests/test_h12_milestone6.py` |
| **PI 4.2** | 5% | Cấu hình Docker/Compose/Render/SECURITY/README/INTEGRATION_CONTRACT, không lộ credential LLM, đóng gói bản vá C# tích hợp với `backend_api` & `student_desktop_app`. | `Dockerfile`, `docker-compose.yml`, `docs/SECURITY.md`, `docs/INTEGRATION_SYNC_LCMS_DESKTOP.md` |

---

## 3. Kết quả Đánh giá Thực nghiệm Offline (Baseline `mock` Adapter)

Trích từ `reports/h12_evaluation_offline.md` (chạy hoàn toàn qua HTTP giữa FastAPI Module 3 và LCMS Simulator `camelCase`):

- **Bộ dữ liệu:** `tests/golden/h12_eval_dataset.json` (`v1.0.0`, 36 kịch bản trên 4 bài học đã duyệt: `lesson-db-01@v1`, `lesson-db-01@v2`, `lesson-se-02@v1`, `lesson-net-03@v1`, cùng bài chưa duyệt `lesson-draft-99@v1`).
- **`question_relevance`:** `6/6` (`100.0%`, ngưỡng `>= 90%`) — sinh đề tự luận bám bài học và từ chối MCQ/quiz (`422`).
- **`lesson_grounding`:** `6/6` (`100.0%`, ngưỡng `>= 100%`) — đúng `knowledge_version`, không lộ `ANSWER_KEY_CONFIDENTIAL`, chặn bài chưa duyệt (`422 UNAPPROVED_KNOWLEDGE`) và từ chối ngữ cảnh không liên quan.
- **`rubric_coverage`:** `4/4` (`100.0%`, ngưỡng `>= 100%`) — phủ đủ tiêu chí rubric, cộng điểm tất định đúng 100%, không tự phát hành điểm.
- **`scoring_agreement`:** `MAE = 0.0` (ngưỡng `<= 1.5`), khớp trạng thái `100.0%` (`8/8`, ngưỡng `>= 90%`).
- **`understanding_followup_quality` & Bảo mật PI 3.4:** `6/6` (`100.0%`, ngưỡng `>= 90%`), kiểm tra bảo mật PI 3.4 đạt `100.0%` (`uf_sec_redirect` + `uf_sec_reveal_and_alter_escalation`).
- **`failure_handling`:** `6/6` (`100.0%`, ngưỡng `>= 100%`) — cô lập chéo phiên, khóa `classroom_exam`, chặn `voice`/`oral`, phục hồi lỗi JSON `502`, chặn `503 LCMS_UNAVAILABLE` & `401`, idempotency `409`.
- **Bất biến toàn cục:** `zero_silent_grade_publication = True`, `zero_answer_key_leakage = True`.

---

## 4. Hạn chế Cần Trình bày Trung thực Khi Bảo vệ

1. **Phạm vi kết quả `--mode mock` vs `--mode gemini`:** Kết quả trong `reports/h12_evaluation_offline.md` kiểm chứng đúng kiến trúc Agent, tool calls, rẽ nhánh LangGraph, chính sách `student_safe`, cô lập phiên và xử lý lỗi mà không gọi API Gemini thật. Khi demo live hoặc đo chất lượng diễn đạt tự nhiên trên `live` LLM, chạy `python scripts/evaluate_h12.py --mode gemini` với `GEMINI_API_KEY` hợp lệ và lưu ý giới hạn quota free-tier.
2. **Vị trí mã nguồn C# của nhóm phối hợp:** Trong workspace nhánh `ai-module`, toàn bộ thay đổi mã nguồn được thực hiện gọn trong `module3-ai-agent/`. Các thay đổi phía C# (`backend_api` của Thưởng và `student_desktop_app` của Kiên trên nhánh `ai-integration`) được cung cấp sẵn dưới dạng đặc tả và patch cụ thể tại `docs/INTEGRATION_SYNC_LCMS_DESKTOP.md` và đã được mô phỏng HTTP hai chiều tương thích ASP.NET Core `camelCase` qua `app/h12_agent/lcms_simulator.py`.
3. **Các endpoint legacy không thuộc phạm vi H12:** Các route cũ như `/api/evaluations/evaluate`, `/api/questions/generate-matrix`, `/api/submissions/handwritten/transcribe` được giữ lại để không làm gãy tương thích ngược các bài kiểm thử trước đó (`test_api_contract.py`), trong khi toàn bộ luồng H12 chạy qua nhóm `/api/agent/*`.
