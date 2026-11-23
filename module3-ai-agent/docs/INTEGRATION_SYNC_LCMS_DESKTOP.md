# Tài liệu Đồng bộ Tích hợp Module 3 (AI Agent H12) ↔ LCMS (`backend_api` — Thưởng) & Desktop App (`student_desktop_app` — Kiên)

> **Mục tiêu (Tiến độ 5 — Tuần 8, PI 3.3 & PI 4.2):** Đặc tả hợp đồng HTTP hai chiều giữa Module 3 (`module3-ai-agent`) với LCMS (`backend_api` trên nhánh `ai-integration`) và Desktop App (`student_desktop_app` trên nhánh `ai-integration`), kèm các bản vá C# mẫu để Thưởng và Kiên áp dụng trực tiếp mà không làm lộ key LLM về phía client.

---

## 1. Kiến trúc kết nối chuẩn H12 & Ranh giới Bảo mật

1. **Không nhúng model credential vào client (PI 4.2):**
   - `GEMINI_API_KEY` chỉ nằm ở biến môi trường của `module3-ai-agent`.
   - Giao tiếp giữa `backend_api` ↔ `module3-ai-agent` xác thực qua header nội bộ `X-Agent-Service-Key` (khớp biến môi trường `AGENT_SERVICE_SECRET` khi `AGENT_AUTH_REQUIRED=true`).
2. **Hỗ trợ chống gửi lặp & truy vết đầu-cuối (PI 2.2):**
   - `Idempotency-Key` (regex `^[A-Za-z0-9._:-]{1,128}$`) dùng cho mọi thao tác ghi/lượt hội thoại (`POST /api/agent/questions/draft`, `POST /api/agent/knowing/grade-suggestion`, `POST /api/agent/understanding/sessions/{session_id}/turns`, và 3 endpoint ghi nội bộ của LCMS).
   - `X-Request-ID` được chuyển tiếp từ request ngoài vào toàn bộ lời gọi `HttpLcmsToolClient`.
3. **CORS Headers trên Module 3 (`app/main.py`):**
   - `allow_headers`: `Content-Type`, `Accept`, `X-Request-ID`, `X-Agent-Service-Key`, `Idempotency-Key`.
   - `expose_headers`: `X-Request-ID`, `X-AI-Adapter-Mode`, `X-AI-Framework`, `X-Idempotent-Replay`.

---

## 2. Hợp đồng 6 Internal LCMS Endpoints mà `backend_api` cần cung cấp cho Module 3

Khi `AGENT_LCMS_MODE=http`, `HttpLcmsToolClient` (`module3-ai-agent/app/h12_agent/lcms_tools.py`) sẽ gọi 6 API nội bộ tới `AGENT_LCMS_BASE_URL` (mặc định tương thích hoàn toàn với JSON `camelCase` chuẩn `System.Text.Json` của ASP.NET Core hoặc `snake_case`):

| # | Method & Path trên LCMS (`backend_api`) | Ánh xạ bảng C# (`ApplicationDbContext`) | Dữ liệu trả về / Mục đích |
|---|---|---|---|
| 1 | `GET /api/internal/agent/lessons/{lessonId}?version=v1&visibility=student_safe` | `Documents` (`AcademicDocument`) + `Subjects` | Trả bài học đã duyệt (`approved == true`). Nếu `visibility == "student_safe"`, **bắt buộc** để `teacherOnlyNotes = null`. |
| 2 | `GET /api/internal/agent/lessons/{lessonId}/assessment-materials?version=v1&rubric_id=...&visibility=student_safe` | `AiRubrics` (`Status == "Approved"`) + `Questions` | Trả rubric đã duyệt + danh sách câu hỏi Knowing/Understanding. Khi `visibility == "student_safe"`, **xóa** `answerKey` và `keyPoints`. |
| 3 | `POST /api/internal/agent/assessment-context` | `ExamSessions` (`InExam`/`Reopened`) + `ExamRoomSessions` (`InProgress`) + `HomeAssignments` | Kiểm tra ngữ cảnh đánh giá: nếu sinh viên đang thi trên lớp, **bắt buộc** trả `classroomExamActive: true`, `allowed: false`, `permittedMode: "classroom_exam"`. |
| 4 | `POST /api/internal/agent/drafts/questions` | `Question` / bản nháp ngân hàng câu hỏi | Lưu bản nháp câu hỏi **tự luận** do AI đề xuất (`publicationStatus: "draft_suggestion"`, `released: false`, `teacherReviewRequired: true`). Hỗ trợ `Idempotency-Key`. |
| 5 | `POST /api/internal/agent/suggestions/grading` | `HomeSubmissions` (`AiSuggestedResult`, `AiCriteriaJson`, `AiFeedback`, `Result = "Pending"`) | Lưu gợi ý chấm bài Knowing gắn tiêu chí để giảng viên duyệt (**không tự chốt điểm**). Hỗ trợ `Idempotency-Key`. |
| 6 | `POST /api/internal/agent/outcomes/dialogue` | Bản ghi kết quả hội thoại Understanding | Lưu tóm tắt phiên hội thoại Understanding đã đóng (`stoppingReason`, `criterionAssessments`, `flags`, `released: false`, `teacherReviewRequired: true`). |

### Gợi ý khung `Controllers/InternalAgentController.cs` cho Thưởng (`backend_api`)

```csharp
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using backend_api.Data;

namespace backend_api.Controllers;

[ApiController]
[Route("api/internal/agent")]
public class InternalAgentController : ControllerBase
{
    private readonly ApplicationDbContext _db;
    private readonly IConfiguration _config;

    public InternalAgentController(ApplicationDbContext db, IConfiguration config)
    {
        _db = db;
        _config = config;
    }

    private bool HasValidServiceKey()
    {
        var expected = _config["AiService:AgentServiceSecret"] ?? "";
        if (string.IsNullOrWhiteSpace(expected)) return true;
        var provided = Request.Headers["X-Agent-Service-Key"].ToString().Trim();
        return string.Equals(provided, expected.Trim(), StringComparison.Ordinal);
    }

    [HttpPost("assessment-context")]
    public async Task<IActionResult> FetchAssessmentContext([FromBody] AssessmentContextRequest req, CancellationToken ct)
    {
        if (!HasValidServiceKey())
            return Unauthorized(new { success = false, code = "AGENT_UNAUTHORIZED", message = "Invalid service key." });

        var code = (req.StudentId ?? "").Trim().ToUpperInvariant();
        var inClassroomExam = !string.IsNullOrEmpty(code) && (
            await _db.ExamSessions.AsNoTracking()
                .AnyAsync(s => s.StudentCode.Trim().ToUpper() == code && (s.Status == "InExam" || s.Status == "Reopened"), ct)
            || await _db.ExamRoomJoins.AsNoTracking()
                .Include(j => j.ExamRoomSession)
                .AnyAsync(j => j.StudentCode.Trim().ToUpper() == code && j.ExamRoomSession != null && j.ExamRoomSession.Status == "InProgress", ct)
        );

        if (inClassroomExam || req.RequestedMode == "classroom_exam")
        {
            return Ok(new
            {
                lessonId = req.LessonId,
                lessonVersion = req.LessonVersion ?? "v1",
                studentId = req.StudentId,
                attemptId = req.AttemptId,
                permittedMode = "classroom_exam",
                classroomExamActive = true,
                maxTurns = 6,
                allowed = false,
                denialReason = "Sinh viên đang trong phiên thi kiểm soát trên lớp (InExam/InProgress).",
                sourceMode = "http"
            });
        }

        return Ok(new
        {
            lessonId = req.LessonId,
            lessonVersion = req.LessonVersion ?? "v1",
            studentId = req.StudentId,
            attemptId = req.AttemptId,
            permittedMode = req.RequestedMode ?? "home_understanding",
            classroomExamActive = false,
            maxTurns = 6,
            allowed = true,
            sourceMode = "http"
        });
    }

    public sealed record AssessmentContextRequest(
        string LessonId,
        string? LessonVersion,
        string? StudentId,
        string? AttemptId,
        string? RequestedMode);
}
```

---

## 3. Các điểm cần vá trên nhánh `ai-integration` cho Thưởng (`backend_api`)

### 3.1. Chặn AI tuyệt đối khi sinh viên đang thi (`StudentController.cs` & `StudentAiController.cs`)
- **Hiện trạng trên `ai-integration`:**
  - `backend_api/Controllers/StudentController.cs` (`[HttpPost("oral-chat")]`) hiện **cho phép** gọi `TryOralRagFromSubjectAsync(...)` khi `ExamSession.Status` là `"InExam"` hoặc `"Reopened"`.
  - `backend_api/Controllers/StudentAiController.cs` (`[HttpPost("ai-chat")]`, `StudyAiChat`) chưa kiểm tra `ExamSession` hay `ExamRoomSession`.
- **Cách vá bắt buộc theo PI 1.2 & Tuần 8:** Thêm hàm dùng chung kiểm tra phòng thi trước mọi endpoint AI cho sinh viên:

```csharp
private async Task<bool> IsStudentInClassroomExamAsync(string studentCode, CancellationToken ct)
{
    var code = studentCode.Trim().ToUpperInvariant();
    var hasExam = await _db.ExamSessions.AsNoTracking()
        .AnyAsync(s => s.StudentCode.Trim().ToUpper() == code && (s.Status == "InExam" || s.Status == "Reopened"), ct);
    if (hasExam) return true;

    return await _db.ExamRoomJoins.AsNoTracking()
        .Include(j => j.ExamRoomSession)
        .AnyAsync(j => j.StudentCode.Trim().ToUpper() == code
                    && j.ExamRoomSession != null
                    && j.ExamRoomSession.Status == "InProgress", ct);
}

// Trong StudyAiChat ([HttpPost("ai-chat")]) & OralChat ([HttpPost("oral-chat")]):
if (await IsStudentInClassroomExamAsync(AuthenticatedStudentCode!, HttpContext.RequestAborted))
{
    return StatusCode(403, new
    {
        success = false,
        code = "ASSESSMENT_MODE_FORBIDDEN",
        message = "Không được phép sử dụng trợ lý AI trong thời gian làm bài thi trên lớp."
    });
}
```

### 3.2. Bỏ tự chốt điểm `final` trong `backend_api/Services/HomeworkAiGrading.cs`
- **Hiện trạng trên `ai-integration`:** Khi `Module 3` trả về không có cờ lỗi, `HomeworkAiGrader.GradeAsync` tự ghi `Result = "Pass"/"Fail"` và `GradedBy = "AI"` vào `HomeSubmissions` mà không qua giảng viên duyệt.
- **Cách vá bắt buộc theo PI 1.3 & PI 2.2:** Chuyển sang gọi `/api/agent/knowing/grade-suggestion` qua `Module3Client` và **luôn giữ `Result = "Pending"`**, chỉ ghi các trường gợi ý cho giảng viên phê duyệt:

```csharp
// Thay thế nhánh auto-final bằng bản nháp chờ giảng viên xác nhận (teacher_handoff = true):
updated = await _db.HomeSubmissions
    .Where(s => s.Id == submissionId && s.Result == "Pending" && s.Status == "Submitted")
    .ExecuteUpdateAsync(set => set
        .SetProperty(s => s.RawPercent, metPercent)
        .SetProperty(s => s.AiSuggestedResult, suggested)
        .SetProperty(s => s.AiFlags, flags.Length > 0 ? flags : "pending_teacher_review")
        .SetProperty(s => s.AiFeedback, feedback)
        .SetProperty(s => s.AiCriteriaJson, criteriaJson)
        .SetProperty(s => s.AiError, (string?)null), ct);
return new(HomeworkAiOutcome.NeedsReview, suggested, flags);
```

### 3.3. Nối `Module3Client.cs` & chuẩn hóa định dạng bài tập (`AppModels.cs`)
- Trong `backend_api/Models/AppModels.cs`, chuẩn hóa `HomeAssignment.Format` nhận `"FixedEssay"` (Knowing) và `"AdaptiveTextChat"` (Understanding) thay cho `"Quiz"` / `"AIOralInterview"`.
- Trong `backend_api/Services/Module3Client.cs`, gắn thêm header `X-Agent-Service-Key` (nếu cấu hình) và `Idempotency-Key`, đồng thời bổ sung 5 wrapper cho các route H12:
  - `POST /api/agent/questions/draft`
  - `POST /api/agent/knowing/grade-suggestion`
  - `POST /api/agent/understanding/sessions/start`
  - `POST /api/agent/understanding/sessions/{sessionId}/turns`
  - `POST /api/agent/understanding/sessions/{sessionId}/close`

---

## 4. Các điểm cần vá cho Kiên (`student_desktop_app` trên `ai-integration`)

1. **Khóa khung chat AI khỏi giao diện thi (`ExamForm.cs` & `ExamKioskForm.cs`):**
   - Gỡ bỏ hoặc ẩn/khóa hoàn toàn nút gọi `_api.SendOralChatAsync(...)` trong `ExamForm.cs` khi sinh viên đang ở màn hình thi kiểm soát trên lớp (`classroom_exam`), hiển thị thông báo chế độ thi không hỗ trợ AI.
2. **Chuyển `ApiClient.cs` sang luồng Understanding Adaptive Text Chat (làm tại nhà):**
   - Thay thế `SendOralChatAsync("/api/student/oral-chat", ...)` bằng 3 hàm gọi qua Backend LCMS tới Module 3:
     - `StartUnderstandingSessionAsync(lessonId, lessonVersion, studentId, attemptId)`
     - `SendUnderstandingTurnAsync(sessionId, studentId, message, idempotencyKey)`
     - `CloseUnderstandingSessionAsync(sessionId, studentId)`
   - Kèm `channel: "text_chat"` và `assessment_mode: "home_understanding"` (không gửi `voice`/`oral`).

---

## 5. Chạy thử tích hợp HTTP độc lập với LCMS Simulator (không cần chờ DB thật)

Module 3 đã tích hợp sẵn **C#-compatible LCMS HTTP Simulator** (`app/h12_agent/lcms_simulator.py`) hỗ trợ cả `camelCase` của ASP.NET Core và `X-Agent-Service-Key`:

```powershell
# Terminal 1: Khởi động LCMS Simulator ở cổng 5099 (xuất JSON camelCase giống ASP.NET Core)
cd module3-ai-agent
$env:LCMS_SIM_CAMEL_CASE="1"
$env:AGENT_SERVICE_SECRET="<replace-with-shared-secret>"
.\.venv\Scripts\python.exe -m app.h12_agent.lcms_simulator

# Terminal 2: Khởi động Module 3 gọi LCMS qua HTTP thật ở cổng 8003
cd module3-ai-agent
$env:AGENT_LCMS_MODE="http"
$env:AGENT_LCMS_BASE_URL="http://127.0.0.1:5099"
$env:AGENT_AUTH_REQUIRED="true"
$env:AGENT_SERVICE_SECRET="<replace-with-shared-secret>"
.\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8003
```
