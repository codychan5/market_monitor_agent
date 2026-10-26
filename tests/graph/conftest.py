import pytest
from langgraph.checkpoint.memory import InMemorySaver

from london_office_monitor.agent.graph import AgentDeps
from london_office_monitor.agent.skills import load_skills
from london_office_monitor.agent.telemetry import MemorySink
from london_office_monitor.paths import SKILLS_DIR
from tests.fake_model import scripted


@pytest.fixture
def agent_deps(mcp_tools, fake_embedder) -> AgentDeps:
    """Real MCP tools over the seeded test database; scripted models; in-memory telemetry and checkpoints."""
    return AgentDeps(skills=load_skills(SKILLS_DIR), mcp_tools=mcp_tools,
                     models={"orchestrator": scripted([]), "investigator": scripted([]), "judge": scripted([])},
                     embedder=fake_embedder, sink=MemorySink(), checkpointer=InMemorySaver(), db_name="lom_test")
