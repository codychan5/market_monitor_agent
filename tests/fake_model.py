"""A scripted chat model for driving the real agent graph in tests (adapted from the nasdaq agent's)."""
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class ScriptedChatModel(GenericFakeChatModel):
    """Plays the scripted AIMessages in order. Records the tool names bound before each call and the messages each
    call received, so tests can assert which tools the skills middleware offered and what the model was sent."""
    bound_tool_names: list[list[str]] = []
    received: list[list[BaseMessage]] = []

    def bind_tools(self, tools, **kwargs):
        object.__setattr__(self, "bound_tool_names", self.bound_tool_names + [[t.name for t in tools]])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        object.__setattr__(self, "received", self.received + [list(messages)])
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(messages: list[AIMessage]) -> ScriptedChatModel:
    return ScriptedChatModel(messages=iter(messages))
