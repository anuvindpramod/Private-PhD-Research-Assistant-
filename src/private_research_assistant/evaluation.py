from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evidence import validate_evidence_refs
from .exceptions import UserFacingError
from .indexing import load_chunks
from .models import Opportunity
from .dates import is_active_deadline
from datetime import date
from .indexing import load_index_manifest
from .paths import latest_output_path


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@dataclass
class EvalCaseResult:
    case_id: str
    question: str
    passing: bool
    detail: str


@dataclass
class EvalRunResult:
    fixture_results: list[EvalCaseResult]
    live_passing: bool | None
    live_detail: str | None

    @property
    def passing(self) -> bool:
        fixture_ok = all(result.passing for result in self.fixture_results)
        live_ok = True if self.live_passing is None else self.live_passing
        return fixture_ok and live_ok


def run_evaluation(*, include_live: bool = False, base_dir: str | Path | None = None) -> EvalRunResult:
    chunks = _load_fixture_chunks()
    opportunities = [Opportunity.from_mapping(item) for item in _load_json(FIXTURE_DIR / "fixture_opportunities.json")]
    questions = _load_json(FIXTURE_DIR / "eval_questions.json")
    results = [_evaluate_case(question, opportunities, chunks) for question in questions]

    live_passing: bool | None = None
    live_detail: str | None = None
    if include_live:
        live_passing, live_detail = _run_live_evaluation(base_dir)

    return EvalRunResult(fixture_results=results, live_passing=live_passing, live_detail=live_detail)


def _evaluate_case(
    case: dict[str, Any],
    opportunities: list[Opportunity],
    chunks: dict[str, dict[str, Any]],
) -> EvalCaseResult:
    opportunity = next((item for item in opportunities if item.title == case["opportunity_title"]), None)
    if opportunity is None:
        return EvalCaseResult(case["id"], case["question"], False, "Opportunity not found in fixture table.")

    field = case["field"]
    actual_value = getattr(opportunity, field)
    expected_value = case["expected_value"]
    expected_ref = case["expected_ref"]
    refs = opportunity.evidence_refs.get(field, [])
    evidence_text = " ".join(chunks[ref]["text"] for ref in refs if ref in chunks)

    checks = [
        (actual_value == expected_value, f"expected {expected_value!r}, got {actual_value!r}"),
        (expected_ref in refs, f"expected ref {expected_ref!r}, got {refs!r}"),
        (case["evidence_must_contain"] in evidence_text, "expected evidence text fragment missing"),
    ]
    failing = [detail for passed, detail in checks if not passed]
    if failing:
        return EvalCaseResult(case["id"], case["question"], False, "; ".join(failing))
    return EvalCaseResult(case["id"], case["question"], True, "passed")


def _run_live_evaluation(base_dir: str | Path | None = None) -> tuple[bool, str]:
    path = latest_output_path(base_dir)
    if not path.exists():
        return False, f"No live extraction output found at {path}. Run `pra extract` first."
    payload = json.loads(path.read_text(encoding="utf-8"))
    opportunities = [Opportunity.from_mapping(item) for item in payload.get("opportunities", [])]
    if not opportunities:
        return False, "Live output has no opportunities."
    try:
        chunks = load_chunks(payload.get("run_id"), base_dir)
    except UserFacingError as exc:
        return False, str(exc)
    errors: list[str] = []
    manifest = load_index_manifest(payload.get("run_id"), base_dir)
    crawl_date = date.fromisoformat(manifest["crawl_date"])
    for opportunity in opportunities:
        validation = validate_evidence_refs(opportunity, chunks)
        if not validation.passing:
            errors.extend(f"{opportunity.title}: {error}" for error in validation.errors)
        if not is_active_deadline(opportunity.deadline, crawl_date):
            errors.append(f"{opportunity.title}: inactive or unparseable deadline")
    if errors:
        return False, "; ".join(errors[:10])
    return True, f"Textual support checks passed for {len(opportunities)} opportunities as of {crawl_date}; not a factual correctness guarantee."

def _load_fixture_chunks() -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    with (FIXTURE_DIR / "fixture_chunks.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                chunks[item["evidence_id"]] = item
    return chunks


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
