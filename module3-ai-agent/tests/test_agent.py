"""Grading Agent: scan, accepted-format filtering, zip safety, resume, export (Gemini faked)."""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import pytest

from agent.config import AgentConfig, load_config
from agent.exporter import export, write_scan_report
from agent.runner import load_state, run
from agent.scanner import scan
from agent.selector import build_answer, select_files
from app.core.exceptions import AppError, ErrorCode
from app.schemas.grading import GradeRequest, GradeResult
from app.services.grading_service import GradingService
from tests.test_grading import FakeLLM, judgement
from tests.test_rubric import make_rubric

MAIN_CS = "using System;\nclass Program { static void Main() { Console.WriteLine(1+2); } }\n"


class ServiceBackend:
    """Same grading core as production, with a fake LLM."""

    def __init__(self, llm: FakeLLM, fail_on: dict[str, AppError] | None = None) -> None:
        self.svc = GradingService(llm_service=llm)
        self.fail_on = fail_on or {}
        self.requests: list[GradeRequest] = []

    async def grade(self, req: GradeRequest) -> GradeResult:
        self.requests.append(req)
        if req.submission_ref in self.fail_on:
            raise self.fail_on[req.submission_ref]
        return await self.svc.grade(req)

    async def close(self) -> None:
        return None


def write(p: Path, text: str | bytes) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.write_bytes if isinstance(text, bytes) else lambda t: p.write_text(t, encoding="utf-8"))(text)
    return p


@pytest.fixture()
def assignment(tmp_path: Path):
    subs = tmp_path / "submissions"
    # good: code + junk that must be filtered out
    write(subs / "SV001_NguyenVanA" / "Program.cs", MAIN_CS)
    write(subs / "SV001_NguyenVanA" / "Utils.cs", "static class Utils { public static int Add(int a,int b)=>a+b; }")
    write(subs / "SV001_NguyenVanA" / "bin" / "Debug" / "app.cs", "// compiled junk")
    write(subs / "SV001_NguyenVanA" / "obj" / "x.cs", "// junk")
    write(subs / "SV001_NguyenVanA" / "README.txt", "hello")
    write(subs / "SV001_NguyenVanA" / "app.exe", b"MZ\x00\x00binary")
    # only binaries -> no_valid_files
    write(subs / "SV002_TranB" / "app.exe", b"MZ\x00\x00binary")
    write(subs / "SV002_TranB" / "notes.docx", b"PK\x03\x04")
    # zip submission
    zpath = subs / "SV003_LeC.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("src/Main.java", "public class Main { public static void main(String[] a){} }")
        zf.writestr("src/bin/Main.class", "junk")
    # bad name / loose file
    write(subs / "khong-dung-ten" / "a.cs", MAIN_CS)
    write(subs / "loose.cs", MAIN_CS)

    rubric = make_rubric()
    rubric.status = "approved"
    write(tmp_path / "rubric.json", rubric.model_dump_json())
    write(tmp_path / "de.md", "Viết chương trình cộng hai số.")
    cfg_path = write(tmp_path / "agent.yaml", "\n".join([
        "assessment_context: homework", "submissions_dir: ./submissions", "question_file: ./de.md",
        "rubric_file: ./rubric.json", "output_dir: ./results",
        "accepted_extensions: [cs, .JAVA]",
    ]))
    return load_config(cfg_path)


def test_config_resolves_paths_and_normalises_extensions(assignment: AgentConfig, tmp_path: Path):
    assert assignment.submissions_dir == (tmp_path / "submissions").resolve()
    assert assignment.accepted_extensions == [".cs", ".java"]
    with pytest.raises(ValueError):
        AgentConfig.model_validate({**assignment.model_dump(), "naming_pattern": "^(.*)$"})


def test_scan_finds_folders_and_zips_and_reports_bad_entries(assignment: AgentConfig):
    res = scan(assignment)
    assert [s.student_ref for s in res.submissions] == ["SV001", "SV002", "SV003"]
    assert dict(res.ignored) == {"khong-dung-ten": "bad_name", "loose.cs": "not_a_folder_or_zip"}
    zip_sub = res.submissions[2]
    assert zip_sub.root is not None and (zip_sub.root / "src" / "Main.java").is_file()


def test_selector_keeps_only_accepted_formats(assignment: AgentConfig):
    sub = scan(assignment).submissions[0]
    sel = select_files(sub.root, assignment)
    assert [f.rel_path for f in sel.accepted] == ["Program.cs", "Utils.cs"]   # entry point first
    reasons = dict(sel.rejected)
    assert reasons["bin/"] == "excluded_dir" and reasons["obj/"] == "excluded_dir"
    assert reasons["README.txt"] == "extension_not_accepted" and reasons["app.exe"] == "extension_not_accepted"


def test_selector_rejects_binary_too_large_and_empty(tmp_path: Path, assignment: AgentConfig):
    root = tmp_path / "x"
    write(root / "bin.cs", b"abc\x00def")
    write(root / "big.cs", "a" * (assignment.max_file_kb * 1024 + 1))
    write(root / "empty.cs", "   \n")
    write(root / "ok.cs", "class A {}")
    sel = select_files(root, assignment)
    assert dict(sel.rejected) == {"bin.cs": "binary", "big.cs": "too_large", "empty.cs": "empty"}
    assert [f.rel_path for f in sel.accepted] == ["ok.cs"]


def test_build_answer_headers_and_truncation(tmp_path: Path, assignment: AgentConfig):
    root = tmp_path / "y"
    write(root / "a.cs", "A" * 500)
    write(root / "b.cs", "B" * 5000)
    files = select_files(root, assignment).accepted
    full, truncated, inc = build_answer(files, 100_000)
    assert not truncated and "// FILE: a.cs" in full and "// FILE: b.cs" in full
    part, truncated, inc = build_answer(files, 2000)
    assert truncated and len(part) <= 2000 and inc[-1] == "b.cs (truncated)"


@pytest.mark.parametrize("name", ["../evil.cs", "/abs.cs", "C:/win.cs"])
def test_zip_slip_is_blocked(tmp_path: Path, assignment: AgentConfig, name: str):
    z = assignment.submissions_dir / "SV009_Evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(name, "x")
    sub = next(s for s in scan(assignment).submissions if s.student_ref == "SV009")
    assert sub.root is None and sub.errors and "unsafe path" in sub.errors[0]
    assert not (tmp_path / "evil.cs").exists()


def test_zip_bomb_limit(assignment: AgentConfig):
    z = assignment.submissions_dir / "SV010_Bomb.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.cs", "0" * (2 * 1024 * 1024))
    cfg = assignment.model_copy(update={"max_unzipped_mb": 1})
    sub = next(s for s in scan(cfg).submissions if s.student_ref == "SV010")
    assert sub.root is None and "max_unzipped_mb" in sub.errors[0]


async def test_run_grades_homework_and_exports(assignment: AgentConfig):
    backend = ServiceBackend(FakeLLM(judgement({"C1": "Đầy đủ", "C2": "Một phần", "C3": "Không đạt"})))
    summary = await run(assignment, backend)
    st = summary.entries
    assert st["SV001"]["result"] == "PASS" and st["SV001"]["status"] == "final"
    assert st["SV001"]["files"]["included"] == ["Program.cs", "Utils.cs"]
    assert st["SV002"]["status"] == "needs_review" and st["SV002"]["flags"] == ["no_valid_files"]
    assert st["SV003"]["result"] == "PASS"
    assert all(r.submission_kind == "code" and r.assessment_context == "homework" for r in backend.requests)
    assert len(backend.requests) == 2                         # SV002 never reached the model
    assert "bin/Debug" not in backend.requests[0].answer      # filtered files never sent

    paths = export(assignment, st)
    rows = list(csv.reader(open(paths[0], encoding="utf-8-sig")))
    assert rows[0][:3] == ["student_ref", "ket_qua", "trang_thai"]
    assert rows[1][:3] == ["SV001", "Đạt", "Đã chốt"] and rows[2][2] == "Chờ GV xem"
    fb = (assignment.output_dir / "feedback" / "SV001.md").read_text(encoding="utf-8")
    assert "Kết quả:** Đạt" in fb
    assert write_scan_report(assignment, scan(assignment)).is_file()


async def test_run_in_class_exports_points(assignment: AgentConfig):
    cfg = assignment.model_copy(update={"assessment_context": "in_class"})
    backend = ServiceBackend(FakeLLM(judgement({"C1": "Đầy đủ", "C2": "Một phần", "C3": "Không đạt"})))
    st = (await run(cfg, backend)).entries
    assert st["SV001"]["suggested_total"] == 6.0 and st["SV001"]["status"] == "pending_review"
    rows = list(csv.reader(open(export(cfg, st)[0], encoding="utf-8-sig")))
    assert rows[0][:5] == ["student_ref", "diem_de_xuat", "diem_toi_da", "C1", "C2"]
    assert rows[1][1] == "6.0" and rows[1][3] == "4.0"


async def test_run_stops_on_quota_and_resumes_without_regrading(assignment: AgentConfig):
    llm = FakeLLM(judgement({"C1": "Đầy đủ", "C2": "Đầy đủ", "C3": "Đầy đủ"}))
    quota = AppError(ErrorCode.LLM_GENERATION_ERROR, "Language model returned HTTP 429.")
    first = await run(assignment, ServiceBackend(llm, fail_on={"SV003": quota}))
    assert first.stopped and "SV001" in first.entries and "SV003" not in first.entries
    assert json.loads(assignment.state_file.read_text(encoding="utf-8"))["SV001"]["result"] == "PASS"

    backend = ServiceBackend(llm)
    second = await run(assignment, backend)
    assert not second.stopped and second.skipped_done == 2          # SV001 graded, SV002 local
    assert [r.submission_ref for r in backend.requests] == ["SV003"]
    assert set(load_state(assignment)) == {"SV001", "SV002", "SV003"}


async def test_non_quota_error_is_recorded_and_run_continues(assignment: AgentConfig):
    llm = FakeLLM(judgement({"C1": "Đầy đủ", "C2": "Đầy đủ", "C3": "Đầy đủ"}))
    bad = AppError(ErrorCode.GRADING_INVALID_RESPONSE, "bad")
    st = (await run(assignment, ServiceBackend(llm, fail_on={"SV001": bad}))).entries
    assert st["SV001"]["status"] == "needs_review" and st["SV001"]["flags"] == ["grading_error"]
    assert st["SV003"]["result"] == "PASS"


async def test_truncated_submission_needs_review(assignment: AgentConfig):
    cfg = assignment.model_copy(update={"max_total_kb": 1})
    write(cfg.submissions_dir / "SV001_NguyenVanA" / "Big.cs", "// x\n" * 400)
    st = (await run(cfg, ServiceBackend(FakeLLM(judgement({"C1": "Đầy đủ", "C2": "Đầy đủ", "C3": "Đầy đủ"}))))).entries
    assert st["SV001"]["status"] == "needs_review" and "truncated" in st["SV001"]["flags"]


def test_cli_scan(assignment: AgentConfig, tmp_path: Path, capsys):
    from agent.__main__ import main

    assert main(["scan", "-c", str(tmp_path / "agent.yaml")]) == 0
    out = capsys.readouterr().out
    assert "SV001" in out and "no_valid_files" in out and "[skip] khong-dung-ten: bad_name" in out


def test_example_config_and_rubric_are_valid():
    from app.grading.rubric_validator import validate_rubric
    from app.schemas.rubric import Rubric

    ex = Path(__file__).resolve().parents[1] / "agent" / "examples"
    cfg = load_config(ex / "agent.yaml")
    rubric = Rubric.model_validate_json((ex / "rubric_approved.json").read_text(encoding="utf-8"))
    assert cfg.assessment_context == "homework" and rubric.status == "approved"
    assert validate_rubric(rubric, "homework") == []
