"""The `lom` command line. Every command prints a one-line cause and fix on failure, never a traceback."""
import asyncio
import logging
import uuid
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import NoReturn

import typer
from pydantic import ValidationError

from .config import MAX_PORT, Settings, describe_validation_error
from .db.connect import DatabaseUnavailable, DbRole, connect
from .db.setup import apply_roles, apply_schema, create_database, setup_checkpointer
from .embed import EmbeddingError, get_embedder
from .ingest.http import HttpClient
from .ingest.run import ingest as run_ingest
from .paths import FIXTURES_DIR, INTERNAL_DIR, SEED_DIR
from .redact import redact
from .seed.load import SeedError, load_fixtures, load_internal, load_seed, read_seed
from .vocab import SourceKind

app = typer.Typer(no_args_is_help=True, add_completion=False, help="London Office Monitor: an AI agent for a "
                  "buy-side analyst covering the London office market.")


class TerminalLogFormatter(logging.Formatter):
    """A log record as one terminal line: an exception reads as its type and message, never a traceback (the
    command has already printed a plain cause), and every secret from the settings is redacted."""

    def __init__(self, settings: Settings | None) -> None:
        super().__init__()
        self.settings = settings

    def format(self, record: logging.LogRecord) -> str:
        # Built from the record itself: another handler may already have cached the full traceback in exc_text.
        line = f"{record.levelname} {record.name}: {record.getMessage()}"
        if record.exc_info and record.exc_info[0] is not None:
            line += f" ({record.exc_info[0].__name__}: {record.exc_info[1]})"
        line = " ".join(line.splitlines())
        return redact(line, self.settings) if self.settings is not None else line


def log_to_terminal() -> None:
    """Send warnings and errors to stderr as single redacted lines. Without a handler Python printed whole
    tracebacks, for example when a chat turn failed."""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h.formatter, TerminalLogFormatter)]:
        root.removeHandler(handler)
    try:
        settings = Settings()
    except ValidationError:
        settings = None  # the command reports invalid settings itself
    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    handler.setFormatter(TerminalLogFormatter(settings))
    root.addHandler(handler)


@app.callback()
def main() -> None:
    """London Office Monitor: an AI agent for a buy-side analyst covering the London office market."""
    log_to_terminal()


def fail(message: str) -> NoReturn:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def load_settings() -> Settings:
    """Settings from the environment and .env; a problem names the field, never the value it was given."""
    try:
        return Settings()
    except ValidationError as e:
        fail(f"Settings are invalid ({describe_validation_error(e)}). Copy .env.example to .env and fill in "
             f"OPENROUTER_API_KEY.")


def prepare_database(settings: Settings, *, reset: bool, demo_items: bool = True) -> None:
    """Create settings.db_name with its schema, roles and checkpoint tables, then load the seed and the snapshot.
    The seed is validated first, so a bad seed never leaves a half-reset database. Without demo_items, the made-up
    news items and mock internal documents stay out, for a database of real data only."""
    bundle = read_seed(SEED_DIR) if SEED_DIR.is_dir() else None
    create_database(settings, settings.db_name, reset=reset)
    with connect(settings, DbRole.OWNER, autocommit=True) as conn:
        apply_schema(conn)
        apply_roles(conn, settings, settings.db_name)
    setup_checkpointer(settings, settings.db_name)
    typer.echo(f"Database {settings.db_name} ready (schema, roles, checkpoints).")
    typer.echo("Loading the embedding model (the first run downloads about 1.1 GB)...")
    embedder = get_embedder(settings)
    with connect(settings, DbRole.OWNER, autocommit=True) as conn:
        if bundle is None:
            typer.echo(f"No seed directory at {SEED_DIR}; skipped loading broker figures.")
        else:
            counts = load_seed(conn, bundle, embedder)
            typer.echo(f"Seed loaded: {counts.metrics} broker figures from {counts.sources} sources, "
                       f"{counts.definitions} definitions, {counts.area_maps} area labels, {counts.assets} "
                       f"portfolio assets, {counts.mandate_statements} mandate statements.")
        if FIXTURES_DIR.is_dir():
            stored = load_fixtures(conn, FIXTURES_DIR, embedder, settings, demo_items=demo_items)
            typer.echo(f"Snapshot loaded: {stored.new} documents ({stored.duplicates} already stored), "
                       f"{stored.joined_story} merged into existing stories.")
        if INTERNAL_DIR.is_dir():
            internal = load_internal(conn, INTERNAL_DIR, embedder, settings, demo_items=demo_items)
            typer.echo(f"Internal documents loaded: {internal.new} ({internal.duplicates} already stored); "
                       f"the bundled ones are mock examples.")
    if not demo_items:
        typer.echo("Demo items left out: this database holds real data only.")


@app.command()
def setup(reset: bool = typer.Option(False, "--reset", help="Drop and recreate the database first."),
          demo_items: bool = typer.Option(True, "--demo-items/--no-demo-items",
                                          help="Load the made-up demo items (six news items and the mock internal "
                                               "documents); leave them out for a database of real data only.")) -> None:
    """Create the database, schema, roles and checkpoint tables."""
    settings = load_settings()
    try:
        prepare_database(settings, reset=reset, demo_items=demo_items)
    except (DatabaseUnavailable, SeedError, EmbeddingError) as e:
        fail(str(e))


@app.command()
def ingest(kinds: str = typer.Option("news,gazette,boe", help="Comma-separated: news, gazette, boe."),
           capture: Path | None = typer.Option(None, help="Also write what was fetched to snapshot files here.")) -> None:
    """Fetch the live sources into the store (live mode only; demo mode uses the stored snapshot)."""
    settings = load_settings()
    try:
        chosen = {SourceKind(k.strip()) for k in kinds.split(",") if k.strip()}
    except ValueError:
        fail(f"Unknown source kind in {kinds!r}; use news, gazette and/or boe.")
    if not chosen:
        fail("Choose at least one source kind: news, gazette and/or boe.")
    try:
        with connect(settings, DbRole.OWNER, autocommit=True) as conn:
            report = run_ingest(settings, chosen, conn=conn, embedder=get_embedder(settings),
                                http=HttpClient(settings), capture_dir=capture)
    except (DatabaseUnavailable, EmbeddingError) as e:
        fail(str(e))
    for name, outcome in report.per_source.items():
        status = outcome.skipped or outcome.error or f"{outcome.fetched} fetched, {outcome.new} new"
        if outcome.note:
            status += f" ({outcome.note})"
        typer.echo(f"{name:24} {status}")
    if not settings.live:
        typer.echo("Set LOM_LIVE=true to fetch the live feeds.")


@app.command()
def mcp(port: int | None = typer.Option(None, min=1, max=MAX_PORT, help="Port to serve on (default LOM_MCP_PORT)."),
        db: str | None = typer.Option(None, help="Database to serve (default LOM_DB_NAME).")) -> None:
    """Run the office-data MCP server on localhost."""
    import uvicorn

    from .mcp_server.server import build_app

    settings = load_settings()
    updates = {key: value for key, value in (("mcp_port", port), ("db_name", db)) if value}
    settings = settings.model_copy(update=updates) if updates else settings
    try:
        application = build_app(settings)
    except EmbeddingError as e:
        fail(str(e))
    typer.echo(f"office-data MCP server on {settings.mcp_url} (database {settings.db_name}, "
               f"{'live' if settings.live else 'demo'} mode). Ctrl+C to stop.")
    uvicorn.run(application, host=settings.mcp_host, port=settings.mcp_port, log_level="warning")


def turn_footer(settings: Settings, turn_id: str) -> str:
    """One line on how the answer was produced: skills, tool calls and model cost."""
    from .agent.telemetry import turn_summary

    try:
        with connect(settings, DbRole.APP) as conn:
            summary = turn_summary(conn, turn_id)
    except DatabaseUnavailable:
        return ""
    return (f"skills: {', '.join(summary.skills_loaded) or 'none'} · tools: {len(summary.tool_calls)} · "
            f"cost: ${summary.cost_usd:.4f}")


@app.command()
def chat(thread: str | None = typer.Option(None, help="Continue an earlier conversation by its thread id.")) -> None:
    """Ask the agent questions in the terminal. An empty line or Ctrl+D ends the session."""
    from .agent.graph import resume_turn, run_turn
    from .mcp_client import McpServerError, ensure_mcp_server

    settings = load_settings()
    try:
        server = ensure_mcp_server(settings)
    except McpServerError as e:
        fail(str(e))
    thread_id = thread or f"cli-{uuid.uuid4().hex[:12]}"
    typer.echo(f"London Office Monitor · conversation {thread_id} · {'live' if settings.live else 'demo'} mode. "
               f"Ask a question; an empty line ends.")
    try:
        while True:
            try:
                question = input("\nyou> ").strip()
            except EOFError:
                break
            if not question:
                break
            result = asyncio.run(run_turn(settings, thread_id, question))
            while result.approval is not None and result.error is None:
                print_cards(result.approval.cards)
                result = asyncio.run(resume_turn(settings, thread_id, ask_decisions(result.approval.cards),
                                                 mode="chat"))
            typer.echo(f"\n{result.error or result.answer or '(no answer)'}")
            footer = turn_footer(settings, result.turn_id)
            if footer:
                typer.secho(footer, dim=True)
    finally:
        server.close()


def print_cards(cards) -> None:
    from .ui.view import score_breakdown

    for card in cards:
        headline, parts = score_breakdown(card)
        typer.echo(f"\n[{card.score if card.score is not None else '-'}] {card.title}  "
                   f"({card.signal_type.value}, {card.submarket.value}; confidence {card.confidence})")
        typer.echo(f"    {card.summary}")
        typer.echo(f"    Why it matters: {card.why_it_matters}")
        typer.echo(f"    Next step: {card.suggested_next_step}")
        typer.echo(f"    {headline}")
        for part in parts:
            typer.echo(f"      {part}")


def ask_decisions(cards):
    from .agent.approval import CardDecision

    decisions = []
    for card in cards:
        answer = typer.prompt(f"Approve card {card.card_id} '{card.title[:60]}'? [y/n]", default="n").strip().lower()
        reason = typer.prompt("Reason (optional)", default="", show_default=False)
        decisions.append(CardDecision(card_id=card.card_id, decision="approve" if answer.startswith("y") else "reject",
                                      reason=reason))
    return decisions


UI_APP = Path(__file__).parent / "ui" / "app.py"


@app.command()
def ui(port: int = typer.Option(8501, min=1, max=MAX_PORT, help="Port for the web app on localhost.")) -> None:
    """Start the data server and the web app on localhost."""
    import subprocess
    import sys

    from .mcp_client import McpServerError, ensure_mcp_server

    settings = load_settings()
    try:
        server = ensure_mcp_server(settings)
    except McpServerError as e:
        fail(str(e))
    typer.echo(f"London Office Monitor: open http://127.0.0.1:{port} (Ctrl+C to stop).")
    try:
        # Headless: no browser launch and no first-run email prompt; usage statistics off; no Deploy menu.
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(UI_APP), "--server.address", "127.0.0.1",
                        "--server.port", str(port), "--server.headless", "true",
                        "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal"], check=False)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


@app.command()
def scan() -> None:
    """Run the opportunity scan once; what it finds waits for review in the web app (Opportunities › To review)."""
    from .agent.graph import SCAN_PROMPT, run_turn
    from .mcp_client import McpServerError, ensure_mcp_server

    settings = load_settings()
    try:
        server = ensure_mcp_server(settings)
    except McpServerError as e:
        fail(str(e))
    try:
        thread_id = f"scan-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        result = asyncio.run(run_turn(settings, thread_id, SCAN_PROMPT, mode="scan"))
    finally:
        server.close()
    if result.error:
        fail(result.error)
    if result.approval is None:
        typer.echo(result.answer or "The scan ended without proposing cards.")
    else:
        print_cards(result.approval.cards)
        typer.echo(f"\n{len(result.approval.cards)} opportunities wait for review in the web app "
                   "(Opportunities › To review).")
    footer = turn_footer(settings, result.turn_id)
    if footer:
        typer.secho(footer, dim=True)


def _choose(raw: str | None, allowed: list[str], what: str) -> list[str]:
    """The comma-separated choices, each checked against what exists; none given means all."""
    chosen = [x.strip() for x in raw.split(",") if x.strip()] if raw else list(allowed)
    unknown = [x for x in chosen if x not in allowed]
    if unknown or not chosen:
        fail(f"Unknown {what}: {', '.join(unknown) or '(none given)'}. Choose from: {', '.join(allowed)}.")
    return chosen


def _eval_models(models: str) -> list[str]:
    """The comma-separated models to compare, each named openrouter:<id>."""
    from .config import MODEL_PREFIX

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    bad = [m for m in model_list if not m.startswith(MODEL_PREFIX) or len(m) == len(MODEL_PREFIX)]
    if bad or not model_list:
        fail(f"Models must be named {MODEL_PREFIX}<id>: {', '.join(bad) or '(none given)'}.")
    return model_list


def _eval_budget(budget_usd: str | None, settings: Settings) -> Decimal:
    """The spending cap: a finite amount above 0 and at most LOM_EVAL_BUDGET_USD, which is also the default."""
    cap_limit = Decimal(str(settings.eval_budget_usd))
    try:
        cap = Decimal(budget_usd) if budget_usd else cap_limit
    except InvalidOperation:
        fail(f"--budget-usd must be a number, got {budget_usd!r}.")
    if not cap.is_finite() or not Decimal(0) < cap <= cap_limit:  # NaN parses, then fails every comparison
        fail(f"--budget-usd must be above 0 and at most {cap_limit} (LOM_EVAL_BUDGET_USD).")
    return cap


@app.command("eval")
def eval_command(
        models: str = typer.Option(..., help="Comma-separated models to compare, each openrouter:<id>."),
        roles: str = typer.Option("orchestrator,investigator", help="Roles whose cases run."),
        cases: str | None = typer.Option(None, help="Comma-separated case ids (default: all)."),
        repeats: int = typer.Option(1, min=1, max=3, help="Runs of each case per model."),
        budget_usd: str | None = typer.Option(None, help="Spending cap in USD; at most LOM_EVAL_BUDGET_USD."),
        concurrency: int = typer.Option(3, min=1, max=5, help="Cases run at the same time.")) -> None:
    """Compare models per role on the evaluation cases under a hard spending cap, in a fresh lom_eval database."""
    from .evals.cases import load_cases
    from .evals.runner import (CASES_FILE, EVAL_DB, EVAL_MCP_PORT, RESULTS_DIR, Budget, run_matrix,
                               write_report)
    from .mcp_client import McpServerError, ensure_mcp_server

    settings = load_settings()
    if settings.db_name == EVAL_DB:
        fail(f"LOM_DB_NAME is {EVAL_DB}, the database lom eval drops and recreates; give the app's database another "
             f"name.")
    model_list = _eval_models(models)
    cap = _eval_budget(budget_usd, settings)
    all_cases = load_cases(CASES_FILE)
    role_list = _choose(roles, ["orchestrator", "investigator"], "role")
    ids = _choose(cases, [c.id for c in all_cases], "case")
    chosen = [c for c in all_cases if c.id in ids and c.role in role_list]
    if not chosen:
        fail("No cases match those roles and ids.")
    eval_settings = settings.model_copy(update={"db_name": EVAL_DB, "mcp_port": EVAL_MCP_PORT, "live": False})
    try:
        prepare_database(eval_settings, reset=True)
        server = ensure_mcp_server(eval_settings, port=EVAL_MCP_PORT, db_name=EVAL_DB)
    except (DatabaseUnavailable, SeedError, EmbeddingError, McpServerError) as e:
        fail(str(e))
    typer.echo(f"Running {len(chosen)} cases x {len(model_list)} models x {repeats} repeats, cap ${cap}...")
    try:
        result = asyncio.run(run_matrix(eval_settings, model_list, chosen, repeats, Budget(cap),
                                        concurrency=concurrency))
    finally:
        server.close()
    json_path, md_path = write_report(result, RESULTS_DIR)
    typer.echo(md_path.read_text(encoding="utf-8"))
    typer.echo(f"Report: {md_path} (every run: {json_path})")


@app.command("eval-judge")
def eval_judge_command(
        models: str = typer.Option(..., help="Comma-separated judge models to compare, each openrouter:<id>."),
        repeats: int = typer.Option(1, min=1, max=3, help="Reviews of each card per model."),
        budget_usd: str | None = typer.Option(None, help="Spending cap in USD; at most LOM_EVAL_BUDGET_USD."),
        concurrency: int = typer.Option(3, min=1, max=5, help="Reviews run at the same time.")) -> None:
    """Compare judge models on hand-labelled cards (evals/judge_cases.yaml) under a hard spending cap. No database
    is needed: each card carries the evidence and context the judge sees in a scan."""
    from .evals import judge, runner

    settings = load_settings()
    model_list = _eval_models(models)
    cap = _eval_budget(budget_usd, settings)
    try:
        cases = judge.load_judge_cases(judge.JUDGE_CASES_FILE)
    except ValueError as e:
        fail(str(e))
    typer.echo(f"Reviewing {len(cases.cases)} cards x {len(model_list)} judges x {repeats} repeats, cap ${cap}...")
    result = asyncio.run(judge.run_judge_matrix(settings, model_list, cases, repeats, runner.Budget(cap),
                                                concurrency=concurrency))
    json_path, md_path = judge.write_judge_report(result, runner.RESULTS_DIR)
    typer.echo(md_path.read_text(encoding="utf-8"))
    typer.echo(f"Report: {md_path} (every review: {json_path})")


@app.command("eval-embedders")
def eval_embedders_command(
        models: str | None = typer.Option(None, help="Comma-separated embedding models (default: every candidate).")
) -> None:
    """Compare local embedding models on the stored corpus with questions in English and Chinese
    (evals/retrieval_cases.yaml). It costs nothing: the models run locally, and each downloads on first use."""
    from .evals import embedders, runner
    from .seed.load import read_fixture_documents, read_internal

    settings = load_settings()
    names = _choose(models, list(embedders.EMBEDDER_SPECS), "embedding model")
    try:
        cases = embedders.load_retrieval_cases(embedders.RETRIEVAL_CASES_FILE)
        docs = read_fixture_documents(FIXTURES_DIR) + read_internal(INTERNAL_DIR)
    except (ValueError, SeedError) as e:
        fail(str(e))
    corpus = embedders.build_corpus(docs, settings)
    typer.echo(f"Embedding {len(corpus.urls)} documents ({len(corpus.chunk_texts)} chunks) and "
               f"{len(cases.topics) * len(embedders.LANGUAGES)} questions with {len(names)} models...")
    results = embedders.compare_embedders([embedders.EMBEDDER_SPECS[name] for name in names], cases, corpus)
    json_path, md_path = embedders.write_embedder_report(results, runner.RESULTS_DIR)
    typer.echo(md_path.read_text(encoding="utf-8"))
    typer.echo(f"Report: {md_path} (every question: {json_path})")


@app.command("eval-briefs")
def eval_briefs_command() -> None:
    """Measure every weekly brief the model comparisons wrote (case c6): figures, stories, rates and portfolio assets
    covered, per model. It reads the saved reports and calls no model."""
    from .evals import briefs, runner

    try:
        asset_names = [asset.name for asset in read_seed(SEED_DIR).assets]
    except SeedError as e:
        fail(str(e))
    found = briefs.collect_briefs(runner.RESULTS_DIR)
    if not found:
        fail(f"No weekly briefs in {runner.RESULTS_DIR}: run lom eval with the {briefs.BRIEF_CASE} case first.")
    md_path = briefs.write_brief_report(found, asset_names, runner.RESULTS_DIR)
    typer.echo(md_path.read_text(encoding="utf-8"))
    typer.echo(f"Report: {md_path}")
