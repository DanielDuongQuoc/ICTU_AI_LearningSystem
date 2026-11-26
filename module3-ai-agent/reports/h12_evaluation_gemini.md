# Báo cáo Đánh giá Hệ thống AI Agent H12 (6 Tiêu chí PI 4.1) — `LIVE`

- **Thời điểm chạy:** `2026-11-26T11:39:08+00:00`
- **Bộ dữ liệu (Dataset version):** `v1.0.0` (4 phiên bản bài học đã duyệt)
- **Model Adapter:** `adapter_mode=live`, `is_mock=false`
- **Cầu nối LCMS:** `http-simulator-camelCase`
- **Thời gian thực thi:** `147.36s`
- **Tổng kết ngưỡng chấp nhận (PI 4.1):** **PASS**
- **Không tự chốt điểm ngầm (`zero_silent_grade_publication`):** `True`
- **Không rò rỉ đáp án mẫu (`zero_answer_key_leakage`):** `True`

## 1. Bảng tổng hợp 6 tiêu chí PI 4.1

| # | Tiêu chí PI 4.1 | Kết quả đạt | Ngưỡng chấp nhận | Trạng thái |
|---:|---|---|---|---|
| 1 | `question_relevance` | 6/6 (100.0%) | >= 90% | PASS |
| 2 | `lesson_grounding` | 6/6 (100.0%) | >= 100% | PASS |
| 3 | `rubric_coverage` | 4/4 (100.0%) | >= 100% | PASS |
| 4 | `scoring_agreement` | MAE=0.0, status=100.0% (8/8) | MAE <= 1.5, status >= 90% | PASS |
| 5 | `understanding_followup_quality` | 6/6 (100.0%), PI3.4 sec=100.0% | >= 90% (sec >= 100%) | PASS |
| 6 | `failure_handling` | 6/6 (100.0%) | >= 100% | PASS |

## 2. Chi tiết từng kịch bản kiểm chứng

### question_relevance

| Case ID | Kết quả |
|---|---|
| `qr_db_v1` | PASS |
| `qr_db_v2` | PASS |
| `qr_se_v1` | PASS |
| `qr_net_v1` | PASS |
| `qr_reject_mcq` | PASS |
| `qr_reject_quiz` | PASS |

### lesson_grounding

| Case ID | Kết quả |
|---|---|
| `lg_db_v1` | PASS |
| `lg_db_v2` | PASS |
| `lg_se_v1` | PASS |
| `lg_net_v1` | PASS |
| `lg_unapproved_lesson` | PASS |
| `lg_insufficient_context` | PASS |

### rubric_coverage

| Case ID | Kết quả |
|---|---|
| `rc_db_v1` | PASS |
| `rc_se_v1` | PASS |
| `rc_net_v1` | PASS |
| `rc_understanding_db_v1` | PASS |

### scoring_agreement

| Case ID | Kết quả |
|---|---|
| `sa_db_full` | PASS |
| `sa_db_partial_c1_only` | PASS |
| `sa_db_missing_required_c1` | PASS |
| `sa_db_empty` | PASS |
| `sa_db_copy_question` | PASS |
| `sa_db_injection` | PASS |
| `sa_se_full` | PASS |
| `sa_net_full` | PASS |

### understanding_followup_quality

| Case ID | Kết quả |
|---|---|
| `uf_db_converges` | PASS |
| `uf_net_converges` | PASS |
| `uf_max_turns` | PASS |
| `uf_student_closure` | PASS |
| `uf_sec_redirect` | PASS |
| `uf_sec_reveal_and_alter_escalation` | PASS |

### failure_handling

| Case ID | Kết quả |
|---|---|
| `fh_cross_session_isolation` | PASS |
| `fh_classroom_exam_denial` | PASS |
| `fh_voice_oral_rejection` | PASS |
| `fh_malformed_model_json_recovery` | PASS |
| `fh_lcms_unavailable_and_auth` | PASS |
| `fh_idempotency_replay_and_conflict` | PASS |

## 3. Hạn chế & Lưu ý diễn giải (Limitations)

- Phiên bản chạy này sử dụng `adapter_mode=live` và cầu nối `http-simulator-camelCase`.
- Khi chạy ở chế độ `--mode mock`, kết quả phản ánh **tính đúng đắn kiến trúc, rẽ nhánh LangGraph, chính sách `student_safe`, chống tự chốt điểm, cô lập phiên, chặn `classroom_exam` và xử lý lỗi**, chưa đo chất lượng diễn đạt ngữ nghĩa của Gemini live (cần chạy `--mode gemini` với `GEMINI_API_KEY` thật khi báo cáo thực nghiệm live).
- Các thay đổi đồng bộ phía C# (`backend_api` và `student_desktop_app`) nằm trên nhánh `ai-integration` và được tài liệu hóa kèm patch chi tiết trong `docs/INTEGRATION_SYNC_LCMS_DESKTOP.md`.
