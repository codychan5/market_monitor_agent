"""The analyst's web app: Chat (cited answers, the weekly brief) and Opportunities (running the scan, deciding what
it found, and the approved list the team sees). Start it with `lom ui`, which starts the data server first;
Streamlit runs this file as a script on every interaction."""
import asyncio
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from london_office_monitor import clock
from london_office_monitor.agent.approval import (ApprovalRequest, CardDecision, approval_status, load_cards,
                                                 pending_approvals, team_list)
from london_office_monitor.agent.cards import StoredCard
from london_office_monitor.agent.charts import Chart
from london_office_monitor.agent.graph import (BRIEF_PROMPT, SCAN_PROMPT, TurnResult, decide_card, finish_approval,
                                              run_turn)
from london_office_monitor.agent.telemetry import turn_summary
from london_office_monitor.citations import get_citation
from london_office_monitor.config import MODEL_PREFIX, Settings, describe_validation_error
from london_office_monitor.db.connect import DatabaseUnavailable, DbRole, connect
from london_office_monitor.facts import NotInData
from london_office_monitor.mcp_client import McpServerError, ensure_mcp_server
from london_office_monitor.ui.view import (DEMO_HELP, card_details_markdown, card_markdown, chart_spec, cited,
                                           has_demo_label, how_i_got_this, linkify_citations, screen_markdown,
                                           shown_window, sources_block)

PAGES = ("Chat", "Opportunities")
THREAD_PARAM = "thread"
# Thread ids arrive in the URL, so anything else is replaced with a new conversation.
THREAD_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_REASON_CHARS = 500
# Shown as bubbles in an empty conversation: questions this data answers well (checked live, 24–25 Oct 2026; the
# last one reads the mock internal documents).
EXAMPLE_QUESTIONS = ("What's City prime rent now, and do the brokers agree?",
                     "Why do BNP Paribas and Savills report different City prime rents?",
                     "How has Central London take-up changed since last quarter?",
                     "How has City prime rent moved?",
                     "What does the supply pipeline mean for 99 City Road?",
                     "What are Bank Rate and the 10-year gilt yield doing?",
                     "西區的空置率是多少？",
                     "What did we decide on Example House before, and why?")
# How often the live view of a running request refreshes, and how many of its latest steps it shows.
POLL_SECONDS = 0.5
MAX_SHOWN_STEPS = 40
ASSETS = Path(__file__).parent / "assets"
# The product mark is the browser-tab icon, the sidebar logo and the assistant's avatar; the analyst's avatar is a
# plain figure. Both are tiles with their own colours, so they read on the light and the dark theme alike.
MARK = str(ASSETS / "mark.svg")
AVATARS = {"user": str(ASSETS / "user.svg"), "assistant": MARK}
PAGE_ICONS = {"Chat": ":material/forum:", "Opportunities": ":material/travel_explore:"}
SCAN_NOTE = "Reads the week's news and insolvency notices against the portfolio and mandate · about 4 minutes"


@dataclass
class Job:
    """A request running in a background thread, so it carries on when the analyst moves to another page; its
    steps arrive on a queue and the page that started it shows them while it runs. The chat and the opportunity
    scan each have their own, so one can run while the other is in use."""
    question: str
    mode: str
    events: queue.Queue
    worker: threading.Thread
    started: float
    result: list
    steps: list[str] = field(default_factory=list)


def load_settings() -> Settings | None:
    try:
        return Settings()
    except ValidationError as e:
        st.error(f"Settings are invalid ({describe_validation_error(e)}). Copy .env.example to .env and fill in "
                 f"OPENROUTER_API_KEY, then restart lom ui.")
        return None


@st.cache_resource
def data_server(_settings: Settings) -> str:
    """Start the office-data server once per app process (lom ui has usually started it already)."""
    return ensure_mcp_server(_settings).url


def data_server_error(settings: Settings) -> str | None:
    """Why the data server cannot be reached, or None once it is up. Call it in the script thread, not a worker:
    it uses Streamlit's resource cache."""
    try:
        data_server(settings)
    except McpServerError as e:
        return str(e)
    return None


def failed_turn(error: str) -> TurnResult:
    return TurnResult(thread_id="", turn_id="", answer=None, error=error)


def run_agent(settings: Settings, call: Callable[[], Coroutine[Any, Any, TurnResult]]) -> TurnResult:
    """Run one agent call to completion; a data server that cannot start becomes the turn's error."""
    error = data_server_error(settings)
    return failed_turn(error) if error else asyncio.run(call())


def new_thread() -> str:
    thread = f"ui-{uuid.uuid4().hex[:12]}"
    st.query_params[THREAD_PARAM] = thread
    return thread


def init_state() -> None:
    if "thread" not in st.session_state:
        from_url = st.query_params.get(THREAD_PARAM, "")
        st.session_state.thread = from_url if THREAD_ID.match(from_url) else new_thread()
    st.session_state.setdefault("exchanges", [])
    st.session_state.setdefault("queued", None)
    st.session_state.setdefault("flash", None)
    st.session_state.setdefault("chat_job", None)
    st.session_state.setdefault("scan_job", None)
    st.session_state.setdefault("scan_result", None)
    if st.session_state.get("page") not in PAGES:  # a session from before the pages were renamed
        st.session_state.page = PAGES[0]


def resolver(conn):
    """get_citation with a per-page cache; a marker that does not resolve is None."""
    cache = {}

    def resolve(citation: str):
        if citation not in cache:
            try:
                cache[citation] = get_citation(conn, citation)
            except NotInData:
                cache[citation] = None
        return cache[citation]

    return resolve


def data_as_of(settings: Settings, conn) -> str:
    if not settings.live:
        return f"Demo mode · data as of {clock.now(settings).astimezone(clock.LONDON):%d %b %Y}"
    newest = conn.execute("SELECT max(fetched_at) FROM documents").fetchone()[0]
    return f"Live mode · news fetched {newest.astimezone(clock.LONDON):%d %b %Y %H:%M}" if newest else \
        "Live mode · nothing fetched yet"


def go_to(page: str) -> None:
    """Section-button callback: show this page."""
    st.session_state.page = page


def queue_prompt(prompt: str, mode: str) -> None:
    """Button callback: run this prompt in the chat."""
    st.session_state.queued = (prompt, mode)


def ask_example() -> None:
    """Example-bubble callback: ask the chosen question, then clear the choice."""
    question, st.session_state.example = st.session_state.example, None
    if question:
        st.session_state.queued = (question, "chat")


def show_examples() -> None:
    """An empty conversation starts with example questions that answer well; choosing one asks it."""
    st.markdown("#### What would you like to know?")
    st.caption("Every figure and passage comes with its source. Try one of these, or ask your own below.")
    st.pills("Example questions", EXAMPLE_QUESTIONS, key="example", on_change=ask_example,
             label_visibility="collapsed")


def start_new_conversation() -> None:
    st.session_state.update(thread=new_thread(), exchanges=[], page="Chat")


def show_markdown(text: str) -> None:
    """Markdown as the app shows it; when it carries a demo badge, a tooltip says what the badges mean."""
    st.markdown(screen_markdown(text), help=DEMO_HELP if has_demo_label(text) else None)


def show_card(card: StoredCard, resolve) -> None:
    """An opportunity card: the plain-words part, then its sources and score folded away for anyone checking."""
    show_markdown(card_markdown(card))
    with st.expander(f"Sources ({len(card.evidence)}) and how the score was worked out",
                     icon=":material/fact_check:"):
        show_markdown(card_details_markdown(card, resolve))


def chat_message(role: str):
    """A chat bubble for "user" or "assistant", with the app's own avatar instead of Streamlit's face and robot."""
    return st.chat_message(role, avatar=AVATARS[role])


def sidebar(settings: Settings, conn, to_review: int) -> None:
    st.logo(MARK, size="large")
    chat_busy = st.session_state.chat_job is not None
    with st.sidebar:
        st.subheader("London Office Monitor", anchor=False)
        st.caption(f":material/database: {data_as_of(settings, conn)}")
        # The pages are buttons, the one on screen filled in. A button keeps no value between runs, so the count in
        # the Opportunities label can change freely. The radio before them sent back its choice as its label text: a
        # decision changed the count, the browser's label no longer matched, and the next click landed on Chat and
        # dropped that click's decision.
        for page in PAGES:
            badge = f" :orange-badge[{to_review}]" if page == "Opportunities" and to_review else ""
            st.button(page + badge, key=f"section-{page}", icon=PAGE_ICONS[page], width="stretch",
                      type="primary" if page == st.session_state.page else "tertiary", on_click=go_to, args=(page,))
        # What comes and goes sits in slots that every run fills or clears: a page keeps its run going while it
        # follows a request (see chat_page), and anything a run leaves out stays on screen until the run ends.
        running, chat_actions = st.empty(), st.empty()
        working = [f"Working on: {st.session_state.chat_job.question[:60]}"] if chat_busy else []
        if st.session_state.scan_job is not None:
            working.append("Scanning for opportunities")
        if working:
            running.caption(screen_markdown("  \n".join(f":material/hourglass_top: {line}" for line in working)))
        if st.session_state.page == "Chat":
            with chat_actions.container():
                st.divider()
                st.button("Write this week's brief", icon=":material/article:", width="stretch",
                          on_click=queue_prompt, args=(BRIEF_PROMPT, "chat"), disabled=chat_busy)
                st.button("New conversation", icon=":material/add_comment:", width="stretch",
                          on_click=start_new_conversation, disabled=chat_busy)
        st.divider()
        models = [f"{role}: {model.removeprefix(MODEL_PREFIX)}"
                  for role, model in (("Orchestrator", settings.model_orchestrator),
                                      ("Investigators", settings.model_investigator), ("Judge", settings.model_judge))]
        st.caption("  \n".join([":material/memory: Models (OpenRouter)", *models]))


def approval_panel(settings: Settings, conn, request: ApprovalRequest, resolve, key: str) -> TurnResult | None:
    """A scan's proposed cards (opportunities, on screen), decided one at a time: each decision is saved at once (an
    approved one reaches the Approved tab), the others stay open, and the scan writes its closing report after the
    last."""
    current = load_cards(conn, [c.card_id for c in request.cards])
    cards = [current[c.card_id][0] for c in request.cards if c.card_id in current]
    open_cards = [c for c in cards if c.status == "pending"]
    st.markdown(f"**{len(cards) - len(open_cards)} of {len(cards)} reviewed.** Approved opportunities move to the "
                f"Approved tab straight away; the scan writes its summary after your last decision.")
    for card in cards:
        if card.status in ("approved", "rejected"):
            mark = (":green[:material/check_circle:] Approved" if card.status == "approved"
                    else ":red[:material/cancel:] Rejected")
            reason = f" ({card.decision_reason})" if card.decision_reason else ""
            show_markdown(f"{mark}: {card.title}{reason}")
    for card in open_cards:
        with st.container(border=True):
            show_card(card, resolve)
            with st.form(f"decide-{key}-{card.card_id}", border=False):
                reason = st.text_input("Reason (optional)", max_chars=MAX_REASON_CHARS,
                                       key=f"reason-{key}-{card.card_id}")
                # Side by side at the width their labels need: in fixed-ratio columns a narrow window cut them to
                # "A…" and "R…".
                with st.container(horizontal=True):
                    approve = st.form_submit_button("Approve", type="primary", icon=":material/check:")
                    reject = st.form_submit_button("Reject", icon=":material/close:")
        if approve or reject:
            choice = CardDecision(card_id=card.card_id, decision="approve" if approve else "reject",
                                  reason=reason.strip())
            with st.spinner("Saving your decision..." if len(open_cards) > 1
                            else "Saving your decision; the scan is writing its summary..."):
                return run_agent(settings, lambda chosen=choice: decide_card(settings, request.thread_id,
                                                                             request.approval_id, chosen))
    if not open_cards and approval_status(conn, request.approval_id) == "pending":
        # Every decision is saved but the scan did not take them (its resume failed): offer the resume again.
        st.warning("Every opportunity is decided, but the scan has not picked up the decisions yet.")
        if st.button("Finish the scan", key=f"finish-{key}-{request.approval_id}"):
            with st.spinner("The scan is writing its summary..."):
                return run_agent(settings, lambda: finish_approval(settings, request.thread_id, request.approval_id))
    return None


def after_decision(outcome: TurnResult | None, *, in_chat: bool) -> None:
    """What the page does after a card decision: show a failure where it happened, refresh the cards while some
    remain open, and show the scan's closing report after the last one."""
    if outcome is None:
        return
    if outcome.error:
        st.error(outcome.error)
        return
    if outcome.answer:
        if in_chat:
            st.session_state.exchanges.append({"question": None, "result": outcome})
        else:
            st.session_state.flash = outcome.answer
    st.rerun()


def steps_markdown(steps: list[str]) -> str:
    shown = steps[-MAX_SHOWN_STEPS:]
    earlier = f"- … {len(steps) - len(shown)} earlier steps\n" if len(steps) > len(shown) else ""
    return earlier + "\n".join(f"- {step}" for step in shown)


def start_job(settings: Settings, text: str, mode: str, thread_id: str) -> Job:
    """Run the request in a background thread; its steps go on a queue for its page to show."""
    events: queue.Queue = queue.Queue()
    result: list = []

    def work() -> None:
        try:
            result.append(asyncio.run(run_turn(settings, thread_id, text, mode=mode, progress=events.put)))
        except Exception as e:  # noqa: BLE001  the page must always get a result to show, never a silent stop
            result.append(TurnResult(thread_id=thread_id, turn_id="", answer=None,
                                     error=f"The request failed: {type(e).__name__}"))

    worker = threading.Thread(target=work, name=f"turn-{thread_id}", daemon=True)
    job = Job(question=text, mode=mode, events=events, worker=worker, started=time.monotonic(), result=result)
    worker.start()
    return job


def start_chat_job(settings: Settings, text: str, mode: str) -> None:
    error = data_server_error(settings)
    if error:
        st.session_state.exchanges.append({"question": text, "result": failed_turn(error)})
        return
    st.session_state.chat_job = start_job(settings, text, mode, st.session_state.thread)


def watch_job(job: Job, label: str) -> TurnResult:
    """Show a running request's steps as they arrive, under its label and the time taken; when it ends, return its
    result. If the analyst moves to another page meanwhile, the request carries on and is picked up on return."""
    status = st.status(label, expanded=True)
    shown = status.empty()
    while True:
        while not job.events.empty():
            job.steps.append(job.events.get_nowait())
        elapsed = int(time.monotonic() - job.started)
        # expanded=True on every refresh: Streamlit's update() collapses the box when it is left out.
        status.update(label=f"{label} · {elapsed // 60}:{elapsed % 60:02d}", expanded=True)
        shown.markdown(screen_markdown(steps_markdown(job.steps) or "- Starting"))
        if not job.worker.is_alive():
            break
        time.sleep(POLL_SECONDS)
    while not job.events.empty():
        job.steps.append(job.events.get_nowait())
    return job.result[0] if job.result else failed_turn("The request stopped without a result.")


def follow_chat_job(job: Job) -> None:
    """The running chat request as an exchange in progress; when it ends, its result and steps join the
    conversation."""
    with chat_message("user"):
        st.markdown(screen_markdown(job.question))
    with chat_message("assistant"):
        result = watch_job(job, "Scanning" if job.mode == "scan" else "Working on it")
    st.session_state.exchanges.append({"question": job.question, "result": result, "steps": list(job.steps)})
    st.session_state.chat_job = None
    st.rerun()


def show_chart(chart: Chart, resolve) -> None:
    """One chart under an answer: brokers (or rates) as lines, every point from a tool result, with its data as a
    table for anyone who cannot read the colours."""
    st.markdown(screen_markdown(f"**{chart.title}** · {chart.unit}"))
    st.vega_lite_chart(chart_spec(chart, dark=st.context.theme.type == "dark"), width="stretch")
    if chart.citations:
        figures = linkify_citations("".join(f"[{c}]" for c in chart.citations), resolve)
        st.caption(screen_markdown(f"Figures: {figures}"))
    else:
        st.caption(f"Source: Bank of England Database · shown: {shown_window(chart)}")
    with st.expander("Chart data", icon=":material/table_chart:"):
        frame = pd.DataFrame({name: dict(points) for name, points in chart.series.items()}).reindex(chart.x)
        frame.index.name = "Date" if chart.x_kind == "date" else "Period"
        st.dataframe(frame)


def show_result(settings: Settings, conn, result: TurnResult, resolve, index: int,
                steps: list[str] | None = None) -> None:
    if result.error:
        (st.warning if result.approval is not None else st.error)(result.error)
    if result.answer:
        show_markdown(linkify_citations(result.answer, resolve))
        for chart in result.charts:
            show_chart(chart, resolve)
        views = [v for v in (resolve(c) for c in cited(result.answer)) if v is not None]
        if views:
            with st.expander(f"Sources ({len(views)})", icon=":material/menu_book:"):
                show_markdown(sources_block(views))
    if steps:
        with st.expander(f"Steps ({len(steps)})", icon=":material/checklist:"):
            st.markdown(screen_markdown(steps_markdown(steps)))
    if result.turn_id:
        with st.expander("How I got this", icon=":material/account_tree:"):
            lines = how_i_got_this(turn_summary(conn, result.turn_id), grounding_rewrites=result.grounding_rewrites,
                                   answer=result.answer or "")
            st.markdown(screen_markdown("\n".join(f"- {line}" for line in lines)))
    if result.approval is not None:
        after_decision(approval_panel(settings, conn, result.approval, resolve, key=f"chat{index}"), in_chat=True)


def chat_page(settings: Settings, conn, resolve) -> None:
    # The example questions come first, in a slot every run fills or clears. Streamlit keeps a run's elements on
    # screen until the next run draws over them or ends, and the run after a question lasts as long as the request
    # (minutes for a scan): without the slot, the bubbles stayed under the progress box until the answer came.
    examples = st.empty()
    shown = set()
    for index, exchange in enumerate(st.session_state.exchanges):
        if exchange["question"]:
            with chat_message("user"):
                st.markdown(screen_markdown(exchange["question"]))
        with chat_message("assistant"):
            show_result(settings, conn, exchange["result"], resolve, index, steps=exchange.get("steps"))
        if exchange["result"].approval is not None:
            shown.add(exchange["result"].approval.approval_id)
    if st.session_state.chat_job is not None:
        st.chat_input("Working on your request...", disabled=True)
        follow_chat_job(st.session_state.chat_job)
        return
    waiting = pending_approvals(conn, st.session_state.thread)
    for request in waiting:
        # A click elsewhere while a scan runs can drop its result from the page; the cards still wait in the
        # database, so show them here rather than a locked chat box with nothing above it.
        if request.approval_id not in shown:
            with chat_message("assistant"):
                st.markdown("This conversation's scan found these opportunities.")
                outcome = approval_panel(settings, conn, request, resolve, key="waiting")
            after_decision(outcome, in_chat=True)
    if not st.session_state.exchanges and not waiting:
        with examples.container():
            show_examples()
    typed = st.chat_input("Ask about the London office market", disabled=bool(waiting))
    if waiting:
        st.caption("Approve or reject the opportunities above to continue this conversation.")
    queued, st.session_state.queued = st.session_state.queued, None
    text, mode = queued if queued else (typed, "chat")
    if not text:
        return
    start_chat_job(settings, text, mode)
    st.rerun()


def open_count(requests: list[ApprovalRequest]) -> int:
    """How many proposed opportunities still wait for a decision."""
    return sum(card.status == "pending" for request in requests for card in request.cards)


def scan_label(request: ApprovalRequest) -> str:
    """A scan named by when it ran (London time), not by its conversation id."""
    if request.created_at is None:
        return "Scan waiting for review"
    return f"Scan on {request.created_at.astimezone(clock.LONDON):%d %b %Y, %H:%M}"


def start_scan(settings: Settings) -> None:
    """Run the opportunity scan in a conversation of its own, so the chat stays free while it works."""
    error = data_server_error(settings)
    if error:
        st.session_state.scan_result = failed_turn(error)
        return
    st.session_state.scan_job = start_job(settings, SCAN_PROMPT, "scan", f"scan-ui-{uuid.uuid4().hex[:12]}")


def show_scan_result(result: TurnResult, resolve) -> None:
    """What the last scan run from this page came back with: a failure, what it found, or its report of finding
    nothing worth a look."""
    if result.error:
        (st.warning if result.approval is not None else st.error)(result.error)
    elif result.approval is not None:
        found = len(result.approval.cards)
        st.success(f"The scan found {found} {'opportunity' if found == 1 else 'opportunities'}: review them below.")
    elif result.answer:
        st.info(screen_markdown(linkify_citations(result.answer, resolve)))


def review_tab(settings: Settings, conn, resolve, requests: list[ApprovalRequest]) -> None:
    """Scans' finds waiting for a decision, the button that runs a new scan, and the live progress of one running.
    Messages and progress sit in slots, as on the chat page, because this tab keeps its run going while a scan
    works."""
    report, outcome = st.empty(), st.empty()
    if st.session_state.flash:
        report.success(screen_markdown(st.session_state.flash))
        st.session_state.flash = None
    reported = st.session_state.scan_result is not None
    if reported:
        with outcome.container():
            show_scan_result(st.session_state.scan_result, resolve)
        st.session_state.scan_result = None
    job = st.session_state.scan_job
    with st.container(horizontal=True, vertical_alignment="center"):  # the button at its label's width, as above
        clicked = st.button("Run a new scan", type="primary", icon=":material/radar:", disabled=job is not None)
        st.caption(SCAN_NOTE)
    if clicked:
        start_scan(settings)
        st.rerun()
    progress = st.empty()  # a running scan's steps, or the note that nothing waits (which the steps replace)
    if not requests and job is None and not reported:
        progress.info("Nothing to review. Run a new scan to look for opportunities.", icon=":material/inbox:")
    for request in requests:
        st.subheader(scan_label(request), anchor=False)
        after_decision(approval_panel(settings, conn, request, resolve, key="review"), in_chat=False)
    if job is not None:
        with progress.container():
            result = watch_job(job, "Scanning")
        st.session_state.update(scan_job=None, scan_result=result)
        st.rerun()


def approved_tab(cards: list[StoredCard], resolve) -> None:
    st.caption("What the team sees: every opportunity the analyst approved, newest first. Rejected ones never "
               "appear here.")
    if not cards:
        st.info("No approved opportunities yet.", icon=":material/inbox:")
    for card in cards:
        with st.container(border=True):
            show_card(card, resolve)


def opportunities_page(settings: Settings, conn, resolve, requests: list[ApprovalRequest]) -> None:
    st.header("Opportunities")
    st.caption("What the scan found in the week's news and insolvency notices, and what has been approved for the "
               "team.")
    approved = team_list(conn)
    to_review, approved_view = st.tabs([f"To review ({open_count(requests)})", f"Approved ({len(approved)})"])
    with approved_view:  # drawn first: the review tab keeps its run going while it follows a scan
        approved_tab(approved, resolve)
    with to_review:
        review_tab(settings, conn, resolve, requests)


def main() -> None:
    st.set_page_config(page_title="London Office Monitor", page_icon=MARK, layout="wide")
    settings = load_settings()
    if settings is None:
        return
    init_state()
    try:
        with connect(settings, DbRole.APP) as conn:
            resolve = resolver(conn)
            requests = pending_approvals(conn)
            sidebar(settings, conn, to_review=open_count(requests))
            if st.session_state.page == "Chat":
                chat_page(settings, conn, resolve)
            else:
                opportunities_page(settings, conn, resolve, requests)
    except DatabaseUnavailable as e:
        st.error(str(e))


main()
