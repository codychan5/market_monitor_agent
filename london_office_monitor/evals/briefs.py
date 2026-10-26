"""Measures of the weekly briefs the model comparisons produced (case c6), computed in code from each brief's text:
how many broker figures it cites for what changed, how many separate stories for the news, whether it covers Bank
Rate and the gilt, and how many portfolio assets it discusses. The eval's own checks only pass or fail a brief on its
headings and the tools called; these show what each model's briefs contain. They are measures, not grades: a longer
brief is not a better one."""
import json
import re
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..agent.grounding import UNVERIFIED_MARKER
from ..citations import CITATION_PATTERN
from ..config import MODEL_PREFIX

BRIEF_CASE = "c6_weekly_brief"
# The sections the brief playbook requires, in order (evals/cases.yaml, required_headings of c6).
BRIEF_SECTIONS = ("## What changed", "## News themes", "## Rates and economy", "## What it means for our portfolio",
                  "## Coming up")
CHANGES, NEWS, RATES, PORTFOLIO = BRIEF_SECTIONS[0], BRIEF_SECTIONS[1], BRIEF_SECTIONS[2], BRIEF_SECTIONS[3]
# The two series the brief's rates section is meant to cover, as the brief names them.
RATE_TERMS = (re.compile(r"\bBank Rate\b", re.IGNORECASE), re.compile(r"\bgilt", re.IGNORECASE))
HEADING = re.compile(r"^## .+$", re.MULTILINE)
# A model comparison report's name is its UTC start time; the judge, embedder and brief reports carry a prefix.
COMPARISON_REPORT = re.compile(r"^\d{8}T\d{6}Z\.json$")


@dataclass(frozen=True)
class BriefMeasures:
    sections: int        # required sections present
    changes_cited: int   # distinct broker figures cited under "What changed"
    stories_cited: int   # distinct documents cited under "News themes"
    rates_covered: int   # of Bank Rate and the 10-year gilt, how many "Rates and economy" mentions
    assets_named: int    # portfolio assets named under "What it means for our portfolio"
    words: int
    unverified: bool     # the grounding check left a visible warning


def _sections(text: str) -> dict[str, str]:
    """Each "## " heading's text, up to the next heading."""
    starts = [(m.group(0).strip(), m.start(), m.end()) for m in HEADING.finditer(text)]
    return {heading: text[end:(starts[i + 1][1] if i + 1 < len(starts) else len(text))]
            for i, (heading, _, end) in enumerate(starts)}


def _cited(text: str, kind: str) -> set[str]:
    """Distinct figures ("F") or distinct documents ("D", any chunk) cited in the text."""
    return {c.split("#")[0] for c in CITATION_PATTERN.findall(text) if c.startswith(kind)}


def brief_measures(text: str, asset_names: list[str]) -> BriefMeasures:
    sections = _sections(text)
    portfolio = sections.get(PORTFOLIO, "").lower()
    return BriefMeasures(
        sections=sum(s in sections for s in BRIEF_SECTIONS),
        changes_cited=len(_cited(sections.get(CHANGES, ""), "F")),
        stories_cited=len(_cited(sections.get(NEWS, ""), "D")),
        rates_covered=sum(bool(term.search(sections.get(RATES, ""))) for term in RATE_TERMS),
        assets_named=sum(name.lower() in portfolio for name in asset_names),
        words=len(text.split()), unverified=UNVERIFIED_MARKER in text)


@dataclass(frozen=True)
class BriefRecord:
    model: str
    started_at: str
    report: str
    passed: bool
    text: str


def collect_briefs(results_dir: Path) -> list[BriefRecord]:
    """Every weekly brief with an answer in the model comparison reports, oldest report first."""
    briefs = []
    for path in sorted(p for p in results_dir.glob("*.json") if COMPARISON_REPORT.match(p.name)):
        report = json.loads(path.read_text(encoding="utf-8"))
        for run in report.get("runs", []):
            if run.get("case") == BRIEF_CASE and run.get("answer"):
                briefs.append(BriefRecord(model=run["model"].removeprefix(MODEL_PREFIX),
                                          started_at=report["started_at"], report=path.name,
                                          passed=bool(run.get("passed")), text=run["answer"]))
    return briefs


def _mean(measures: list[BriefMeasures], field: str) -> float:
    return statistics.mean(getattr(m, field) for m in measures)


def write_brief_report(briefs: list[BriefRecord], asset_names: list[str], out_dir: Path) -> Path:
    """evals/results/briefs-<UTC timestamp>.md: each model's mean measures, then every brief's own."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"briefs-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.md"
    measured = [(b, brief_measures(b.text, asset_names)) for b in briefs]
    summary = []
    for model in dict.fromkeys(b.model for b in briefs):
        records = [b for b, _ in measured if b.model == model]
        measures = [m for b, m in measured if b.model == model]
        summary.append(f"| {model} | {len(records)} | {sum(b.passed for b in records)}/{len(records)} | "
                       f"{_mean(measures, 'changes_cited'):.1f} | {_mean(measures, 'stories_cited'):.1f} | "
                       f"{_mean(measures, 'rates_covered'):.1f} | {_mean(measures, 'assets_named'):.1f} | "
                       f"{_mean(measures, 'words'):.0f} | {sum(m.unverified for m in measures)} |")
    rows = [f"| {b.report.removesuffix('.json')} | {b.model} | {'✓' if b.passed else '✗'} | {m.sections} | "
            f"{m.changes_cited} | {m.stories_cited} | {m.rates_covered} | {m.assets_named} | {m.words} |"
            for b, m in measured]
    md = ["# Weekly brief comparison", "",
          (f"Every weekly brief (case {BRIEF_CASE}) in the model comparison reports, measured in code. Figures cited "
           "counts distinct broker figures under What changed; stories cited counts distinct documents under News "
           f"themes; rates covered counts Bank Rate and the gilt (of 2); assets named counts the {len(asset_names)} "
           "portfolio assets under What it means for our portfolio. These are measures, not grades."), "",
          ("| Model | Briefs | Passed | Figures cited | Stories cited | Rates covered | Assets named | Words | "
           "Unverified |"), "|---|---|---|---|---|---|---|---|---|", *summary, "", "## Every brief", "",
          "| Report | Model | Passed | Sections | Figures | Stories | Rates | Assets | Words |",
          "|---|---|---|---|---|---|---|---|---|", *rows, ""]
    md_path.write_text("\n".join(md), encoding="utf-8")
    return md_path
