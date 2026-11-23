"""H12 AI Agent Milestone 6 tests: 6-criteria PI 4.1 evaluator, deployment config & thesis artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import evaluate_h12

ROOT = Path(__file__).resolve().parents[1]


def test_h12_dataset_validation_and_malformed_rejection(tmp_path):
    dataset = evaluate_h12.load_h12_dataset()
    assert dataset["metadata"]["dataset_version"] == "1.0.0"
    for criterion in evaluate_h12.REQUIRED_CRITERIA:
        assert len(dataset[criterion]) >= 4

    # Ensure >= 4 approved lesson versions are represented (3 built-in seeds + >= 1 extra_lesson).
    approved_extra = [
        item["lesson"]["lesson_id"]
        for item in dataset.get("extra_lessons", [])
        if item.get("lesson", {}).get("approved") is True
    ]
    assert len(approved_extra) >= 1

    # CLI --validate-only succeeds.
    assert evaluate_h12.main(["--validate-only"]) == 0

    # Malformed dataset fails with ValueError.
    bad_file = tmp_path / "bad_eval.json"
    bad_file.write_text('{"metadata": {}}', encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_h12.load_h12_dataset(bad_file)


@pytest.mark.asyncio
async def test_h12_six_criteria_offline_evaluation_passes_all_thresholds():
    result = await evaluate_h12.run_h12_evaluation(mode="mock")
    assert result["overall_passed"] is True
    assert result["adapter_mode"] == "mock"
    assert result["is_mock"] is True
    assert result["zero_silent_grade_publication"] is True
    assert result["zero_answer_key_leakage"] is True
    assert result["lesson_count"] >= 4

    criteria = result["criteria"]
    for key in evaluate_h12.REQUIRED_CRITERIA:
        assert criteria[key]["met"] is True, f"Criterion {key} did not meet threshold: {criteria[key]}"

    assert criteria["understanding_followup_quality"]["security_pass_rate"] == 1.0
    assert criteria["failure_handling"]["pass_rate"] == 1.0

    md = evaluate_h12.render_markdown_report(result)
    for key in evaluate_h12.REQUIRED_CRITERIA:
        assert key in md
    assert "adapter_mode=mock" in md
    assert "Limitations" in md


def test_deployment_hardening_and_thesis_documentation_present():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    security_doc = (ROOT / "docs" / "SECURITY.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    thesis_doc = (ROOT / "docs" / "THESIS_H12_GUIDE.md").read_text(encoding="utf-8")
    eval_md = ROOT / "reports" / "h12_evaluation_offline.md"

    assert "AGENT_SESSION_DB_PATH=/app/data/sessions/h12_sessions.db" in dockerfile
    assert "/app/data/sessions" in dockerfile
    assert "session_data:/app/data/sessions" in compose
    assert "H12 AI Agent Security & Boundaries" in security_doc
    assert "H12 AI Agent Module (Blended Learning)" in readme
    for pi_idx in (
        "PI 1.1",
        "PI 1.2",
        "PI 1.3",
        "PI 2.1",
        "PI 2.2",
        "PI 2.3",
        "PI 3.1",
        "PI 3.2",
        "PI 3.3",
        "PI 3.4",
        "PI 4.1",
        "PI 4.2",
    ):
        assert pi_idx in thesis_doc
    assert eval_md.is_file()
