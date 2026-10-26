"""Skills: SKILL.md playbooks loaded on demand (progressive disclosure). The system prompt lists only each skill's
name and description; load_skill pulls in the body, and only then are that skill's tools offered to the model."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NotRequired

import yaml
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from ..vocab import MCP_TOOL_NAMES

SKILL_FILE = "SKILL.md"
AGENT_TOOL_NAMES = frozenset({"investigate_signals", "propose_opportunities"})
ALWAYS_ON_TOOLS = ("load_skill", "read_skill_file", "get_coverage")
KNOWN_TOOLS = MCP_TOOL_NAMES | AGENT_TOOL_NAMES | {"read_skill_file"}
REQUIRED_KEYS = frozenset({"name", "description", "allowed-tools"})
OPTIONAL_KEYS = frozenset({"subagent-only"})


class SkillError(Exception):
    """A SKILL.md file is malformed; the message names the file and the problem."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    allowed_tools: tuple[str, ...]
    body: str
    folder: Path
    subagent_only: bool


def _split_frontmatter(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise SkillError(f"{path}: must start with a --- frontmatter block")
    end = text.find("\n---", 4)
    if end == -1:
        raise SkillError(f"{path}: frontmatter is not closed with ---")
    try:
        meta = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError as e:
        raise SkillError(f"{path}: frontmatter is not valid YAML ({e})") from e
    if not isinstance(meta, dict):
        raise SkillError(f"{path}: frontmatter must be a mapping")
    return meta, text[end + 4:].lstrip("\n")


def _tools(value, path: Path) -> tuple[str, ...]:
    tools = [t.strip() for t in value.split(",")] if isinstance(value, str) else value
    if not isinstance(tools, list) or not all(isinstance(t, str) and t for t in tools):
        raise SkillError(f"{path}: allowed-tools must be a list of tool names")
    unknown = sorted(set(tools) - KNOWN_TOOLS)
    if unknown:
        raise SkillError(f"{path}: unknown tool(s) in allowed-tools: {', '.join(unknown)}")
    return tuple(tools)


def load_skills(skills_dir: Path) -> dict[str, Skill]:
    """Every skills_dir/<name>/SKILL.md, validated. Raises SkillError on the first malformed skill."""
    skills = {}
    for path in sorted(skills_dir.glob(f"*/{SKILL_FILE}")):
        meta, body = _split_frontmatter(path)
        keys = set(meta)
        if not REQUIRED_KEYS <= keys or keys - REQUIRED_KEYS - OPTIONAL_KEYS:
            raise SkillError(f"{path}: frontmatter keys must be {sorted(REQUIRED_KEYS)} plus optional "
                             f"{sorted(OPTIONAL_KEYS)}; got {sorted(keys)}")
        if meta["name"] != path.parent.name:
            raise SkillError(f"{path}: name {meta['name']!r} must equal its folder name {path.parent.name!r}")
        skills[meta["name"]] = Skill(name=meta["name"], description=str(meta["description"]).strip(),
                                     allowed_tools=_tools(meta["allowed-tools"], path), body=body,
                                     folder=path.parent, subagent_only=bool(meta.get("subagent-only", False)))
    return skills


def skill_index(skills: dict[str, Skill]) -> str:
    return "\n".join(f"- {s.name}: {s.description}" for s in sorted(skills.values(), key=lambda s: s.name)
                     if not s.subagent_only)


def allowed_tool_names(skills: dict[str, Skill], loaded: list[str]) -> set[str]:
    allowed = set(ALWAYS_ON_TOOLS)
    for name in loaded:
        if name in skills:
            allowed.update(skills[name].allowed_tools)
    return allowed


def make_skill_tools(skills: dict[str, Skill]) -> list[BaseTool]:
    loadable = sorted(s.name for s in skills.values() if not s.subagent_only)

    @tool("load_skill")
    def load_skill(name: str, runtime: ToolRuntime[Any, Any]) -> Command | str:
        """Load a skill's playbook by name (see the skill list in your instructions). Its tools become available
        after loading. Load the skill before doing that kind of task."""
        skill = skills.get(name)
        if skill is None or skill.subagent_only:
            return f"ERROR: no skill named {name!r}. Skills: {', '.join(loadable)}."
        current = runtime.state.get("loaded_skills") or []
        return Command(update={"loaded_skills": sorted(set(current) | {name}),
                               "messages": [ToolMessage(skill.body, tool_call_id=runtime.tool_call_id)]})

    @tool("read_skill_file")
    def read_skill_file(skill: str, path: str) -> str:
        """Read a reference file (.md) that a skill's playbook points to, e.g. read_skill_file("reconcile-brokers",
        "definitions.md")."""
        found = skills.get(skill)
        if found is None:
            return f"ERROR: no skill named {skill!r}."
        folder = found.folder.resolve()
        target = (folder / path).resolve()
        if not target.is_relative_to(folder) or target.suffix != ".md" or not target.is_file():
            return f"ERROR: {path!r} is not a .md file inside the {skill} skill."
        return target.read_text(encoding="utf-8")

    return [load_skill, read_skill_file]


class SkillsState(AgentState):
    loaded_skills: NotRequired[list[str]]


def _tool_name(t) -> str | None:
    return getattr(t, "name", None) or (t.get("name") if isinstance(t, dict) else None)


class SkillsMiddleware(AgentMiddleware):
    """Offers the model only the tools of the skills it has loaded, and refuses any other tool call. Loaded skills
    are agent state, so they persist for the conversation's thread."""
    state_schema = SkillsState

    def __init__(self, skills: dict[str, Skill]):
        super().__init__()
        self.skills = skills

    def _allowed(self, state) -> set[str]:
        return allowed_tool_names(self.skills, (state or {}).get("loaded_skills") or [])

    def _owner(self, tool_name: str) -> str | None:
        return next((s.name for s in sorted(self.skills.values(), key=lambda s: s.name)
                     if tool_name in s.allowed_tools and not s.subagent_only), None)

    async def awrap_model_call(self, request, handler):
        allowed = self._allowed(request.state)
        return await handler(request.override(tools=[t for t in request.tools if _tool_name(t) in allowed]))

    async def awrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        if name in self._allowed(request.state):
            return await handler(request)
        owner = self._owner(name)
        hint = f" until its skill is loaded: load_skill('{owner}')" if owner else ""
        return ToolMessage(f"ERROR: {name} is not available{hint}", tool_call_id=request.tool_call["id"],
                           status="error")
