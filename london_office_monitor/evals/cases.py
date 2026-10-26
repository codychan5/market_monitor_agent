"""Evaluation cases: a prompt, the role whose model is under test, and code-graded checks (evals/cases.yaml)."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..vocab import Metric, Submarket


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FactSelector(StrictModel):
    """The latest value from each broker that reports this metric for this submarket."""
    metric: Metric
    submarket: Submarket


class ChatChecks(StrictModel):
    expect_latest: list[FactSelector] = []
    # Routing: the playbook a turn loads is its intent label.
    skills_loaded: list[str] = []
    skills_not_loaded: list[str] = []
    no_skills: bool = False
    tools_called: list[str] = []
    allowed_tools: list[str] | None = None
    cites_source_kinds: list[Literal["news", "gazette", "internal"]] = []
    must_contain: list[str] = []
    must_not_average: list[FactSelector] = []
    answer_language: Literal["en", "zh"] | None = None
    required_headings: list[str] = []


class ScanChecks(StrictModel):
    must_surface: list[str] = []
    must_not_surface: list[str] = []
    canary: str | None = None
    allowed_tools: list[str] | None = None


class EvalCase(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9_]{1,40}$")
    kind: Literal["chat", "scan"]
    role: Literal["orchestrator", "investigator"]
    prompt: str = Field(min_length=1, max_length=500)
    checks: ChatChecks | ScanChecks

    @model_validator(mode="before")
    @classmethod
    def _checks_for_kind(cls, data):
        """Parse checks by the case's kind, so an unknown check name is an error rather than a silent default."""
        if isinstance(data, dict) and isinstance(data.get("checks"), dict):
            checks_type = ScanChecks if data.get("kind") == "scan" else ChatChecks
            data = {**data, "checks": checks_type(**data["checks"])}
        return data


def load_cases(path: Path) -> list[EvalCase]:
    """The cases in a YAML list, with every problem reported together; ids must be unique."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a list of cases")
    cases, problems = [], []
    for index, item in enumerate(raw):
        try:
            cases.append(EvalCase(**item) if isinstance(item, dict) else EvalCase.model_validate(item))
        except (ValidationError, TypeError) as e:
            problems.append(f"case {index + 1}: {e}")
    ids = [c.id for c in cases]
    problems += [f"duplicate case id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return cases
