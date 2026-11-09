"""Write results for the lecturer: results.csv (Excel-friendly), results.json, feedback/<student_ref>.md."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from agent.config import AgentConfig
from agent.scanner import ScanResult
from agent.selector import select_files

RESULT_LABEL = {"PASS": "Đạt", "FAIL": "Không đạt"}
STATUS_LABEL = {"final": "Đã chốt", "needs_review": "Chờ GV xem", "pending_review": "Chờ GV duyệt"}


def export(cfg: AgentConfig, state: dict[str, dict[str, Any]]) -> list[Path]:
    out = cfg.output_dir
    (out / "feedback").mkdir(parents=True, exist_ok=True)
    rows = [state[k] for k in sorted(state)]
    criteria_ids: list[str] = []
    for r in rows:
        for c in r.get("criteria", []):
            if c["criterion_id"] not in criteria_ids:
                criteria_ids.append(c["criterion_id"])

    csv_path = out / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        if cfg.assessment_context == "homework":
            w.writerow(["student_ref", "ket_qua", "trang_thai", "tieu_chi_bat_buoc_chua_dat", "flags", "warnings"])
            for r in rows:
                w.writerow([r["student_ref"], RESULT_LABEL.get(r.get("result") or "", ""),
                            STATUS_LABEL.get(r["status"], r["status"]), ",".join(r.get("unmet_required", [])),
                            ",".join(r.get("flags", [])), ",".join(r.get("warnings", []))])
        else:
            w.writerow(["student_ref", "diem_de_xuat", "diem_toi_da", *criteria_ids, "trang_thai", "flags", "warnings"])
            for r in rows:
                pts = {c["criterion_id"]: c.get("points") for c in r.get("criteria", [])}
                w.writerow([r["student_ref"], r.get("suggested_total"), r.get("max_score"),
                            *(pts.get(cid, "") for cid in criteria_ids), STATUS_LABEL.get(r["status"], r["status"]),
                            ",".join(r.get("flags", [])), ",".join(r.get("warnings", []))])

    json_path = out / "results.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    for r in rows:
        (out / "feedback" / f"{r['student_ref']}.md").write_text(_feedback_md(cfg, r), encoding="utf-8")
    return [csv_path, json_path, out / "feedback"]


def _feedback_md(cfg: AgentConfig, r: dict[str, Any]) -> str:
    lines = [f"# Kết quả chấm: {r['student_ref']}", ""]
    if cfg.assessment_context == "homework":
        lines.append(f"**Kết quả:** {RESULT_LABEL.get(r.get('result') or '', 'Chưa có')}  ")
    else:
        lines.append(f"**Điểm đề xuất:** {r.get('suggested_total')} / {r.get('max_score')} (chờ giảng viên duyệt)  ")
    lines += [f"**Trạng thái:** {STATUS_LABEL.get(r['status'], r['status'])}", ""]
    if r.get("flags"):
        lines += [f"> Cần giảng viên xem: {', '.join(r['flags'])}", ""]
    lines += ["## Nhận xét", "", r.get("feedback") or "", ""]
    if r.get("criteria"):
        lines += ["## Theo tiêu chí", ""]
        for c in r["criteria"]:
            mark = ("✔" if c.get("met") else "✘") if cfg.assessment_context == "homework" \
                else f"{c.get('points')}/{c.get('max_points')}"
            req = " (bắt buộc)" if c.get("required") else ""
            lines.append(f"- **{c['criterion_id']} {c['name']}{req}**: {c['level']} {mark}. {c.get('reason', '')}")
        lines.append("")
    files = r.get("files") or {}
    if files.get("included"):
        lines += ["## File đã chấm", "", *(f"- {p}" for p in files["included"]), ""]
    return "\n".join(lines)


def write_scan_report(cfg: AgentConfig, scanned: ScanResult) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.output_dir / "scan_report.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["entry", "student_ref", "accepted_files", "rejected_files", "problems"])
        for name, reason in scanned.ignored:
            w.writerow([name, "", 0, 0, reason])
        for s in scanned.submissions:
            if s.root is None:
                w.writerow([s.source.name, s.student_ref, 0, 0, "; ".join(s.errors)])
                continue
            sel = select_files(s.root, cfg)
            problems = "no_valid_files" if not sel.accepted else ""
            w.writerow([s.source.name, s.student_ref, len(sel.accepted), len(sel.rejected), problems])
    return path
