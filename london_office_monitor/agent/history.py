"""What the model is sent of earlier turns. Every model call resends the whole conversation, so the raw results of
read-only data tools from before the user's last message are replaced with a short note. The grounding check accepts
figures only from this turn's results, so those older results can no longer back an answer; the questions, answers
and tool calls stay, with the citations needed to fetch the data again. Only the request is edited: the
conversation's checkpoint keeps every message."""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, ToolMessage

from .grounding import turn_start

# The office-data tools that only read, so calling one again fetches its data afresh. Every other result stays in
# view: skill playbooks are instructions, and refresh_sources and the agent's own tools write, so calling one again
# would repeat that work.
REFETCHABLE_TOOLS = frozenset({
    "get_coverage", "get_metric", "compare_brokers", "metric_change", "metric_trend", "get_macro", "search_documents",
    "get_portfolio", "get_mandate", "match_to_mandate", "similar_past_decisions", "get_citation",
})
CLEARED_RESULT = ("[Cleared: a result from an earlier turn. If this turn needs its data, call the tool again, or "
                  "get_citation on a citation you used.]")


def without_earlier_results(messages: list[BaseMessage]) -> list[BaseMessage]:
    """The messages with each read-only tool result from before the current turn replaced by CLEARED_RESULT. The
    replacements are copies, so the given messages are left as they are."""
    start = turn_start(messages)
    return [m.model_copy(update={"content": CLEARED_RESULT})
            if i < start and isinstance(m, ToolMessage) and m.name in REFETCHABLE_TOOLS else m
            for i, m in enumerate(messages)]


class EarlierResultsMiddleware(AgentMiddleware):
    """Sends each model call the conversation without earlier turns' read-only tool results."""

    async def awrap_model_call(self, request, handler):
        return await handler(request.override(messages=without_earlier_results(request.messages)))
