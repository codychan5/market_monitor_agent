# Spike findings (Task 1, 2026-10-24)

The throwaway scripts lived in `spike/` and were deleted after this run. Every answer below was observed on this machine, using the owner's OpenRouter key.

## Changes needed

None to the design. The answers below adjust some implementation details; each is recorded as a ruling in the plan ledger.

## Resolved versions (`uv sync --all-extras`, Python 3.12.12)

| Package | Version |
|---|---|
| langchain | 1.4.7 |
| langchain-core | 1.7.0 |
| langchain-openai | 1.7.2 |
| langgraph | 1.2.17 |
| langgraph-checkpoint | 4.3.0 |
| langgraph-checkpoint-postgres | 3.2.0 |
| langgraph-prebuilt | 1.1.1 |
| langchain-mcp-adapters | 0.3.2 |
| mcp | 1.30.0 |
| fastembed | 0.9.0 |
| onnxruntime | 1.31.0 |
| streamlit | 1.66.1 |
| psycopg | 3.3.6 |
| psycopg-pool | 3.3.3 |
| pgvector | 0.5.1 |
| pydantic | 2.14.1 |
| pydantic-settings | 2.16.1 |
| openai | 3.37.0 |
| feedparser | 6.0.14 |
| httpx | 0.28.1 |
| starlette | 1.9.0 |
| uvicorn | 0.54.0 |
| typer | 0.27.3 |
| pytest | 9.1.1 |
| pytest-asyncio | 1.4.0 |
| respx | 0.23.1 |

## Q1: OpenRouter cost and data policy

```python
ChatOpenAI(model="z-ai/glm-5.3", base_url="https://openrouter.ai/api/v1", api_key=key, temperature=0,
           extra_body={"usage": {"include": True}, "provider": {"data_collection": "deny"}})
msg.response_metadata["token_usage"]["cost"]          # float USD, e.g. 0.0002026
msg.response_metadata["token_usage"]["cost_details"]  # upstream prompt/completion split
msg.usage_metadata                                    # input/output tokens; output_token_details.reasoning
```

- All four models answered under `data_collection: deny`: GLM-5.3, DeepSeek V4.1 Flash, Kimi K3 and Qwen3.7 Plus.
- `response_metadata["model_provider"]` is LangChain's provider (`openai`), not the upstream host. The upstream host isn't exposed, so telemetry records `provider = None`.
- **Reasoning tokens count against `max_tokens`:**
  - "Reply ok" used 37–151 reasoning tokens.
  - Keep `llm_max_output_tokens` generous (8192).

## Q2: structured output

`create_agent(model, tools=[], response_format=ToolStrategy(Model))` returned `structured_response` for GLM-5.3, DeepSeek V4.1 Flash and Kimi K3.

`ProviderStrategy` failed for DeepSeek and Kimi: their reasoning used up the 400-token cap (`LengthFinishReasonError`). **Use `ToolStrategy`.**

## Q3: FastMCP over streamable HTTP with a shared secret

```python
mcp = FastMCP("office-data", host="127.0.0.1", port=8765, stateless_http=True, json_response=True)
@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request): return JSONResponse({"ok": True})
app = mcp.streamable_http_app()          # Starlette app; keeps the session-manager lifespan
app.add_middleware(BearerSecret)         # pure ASGI middleware: 401 unless Authorization: Bearer <secret>
uvicorn.run(app, host="127.0.0.1", port=8765)
```

- Without the secret, `POST /mcp` returned 401. `/healthz` stays open: it reveals nothing and is bound to localhost.
- **Unknown arguments are silently ignored.** `call_tool("typed", {"metric": "prime_rent", "unknown_field": 1})` returned `isError=False`. Task 8 therefore adds its own check that rejects unknown keys in `tools/call`.

## Q4: MCP tools in the agent

```python
client = MultiServerMCPClient({"office-data": {"transport": "streamable_http", "url": url,
                                               "headers": {"Authorization": f"Bearer {secret}"}}})
tools = await client.get_tools()
```

- The tool schema carries the enum (`{"enum": ["prime_rent", "vacancy_rate"]}`).
- **`await tool.ainvoke(args)` returns content blocks, not a string**: `[{"type": "text", "text": "<JSON>", "id": ...}]`. Tests parse `json.loads(result[0]["text"])`.
- **Errors come back as text; nothing is raised.**
  - A server exception gives `"Error executing tool boom: deliberate failure"`.
  - A bad enum gives `"Error executing tool typed: 1 validation error ..."`.
  - Under `create_agent`, both become a `ToolMessage` with `status="error"`.

## Q5: middleware hooks under `ainvoke`

A middleware with only `wrap_model_call` raised `NotImplementedError: Asynchronous implementation of awrap_model_call is not available` under `ainvoke`. **Every middleware we write must implement the async hooks**: `abefore_model`, `aafter_model`, `awrap_model_call` and `awrap_tool_call`. All of them run as expected.

## Q6: skill state updated from a tool

```python
class SkillsState(AgentState): loaded_skills: NotRequired[list[str]]
class SkillsMw(AgentMiddleware): state_schema = SkillsState
@tool
def load_skill(name: str, runtime: ToolRuntime) -> Command:
    current = runtime.state.get("loaded_skills") or []
    return Command(update={"loaded_skills": sorted(set(current) | {name}),
                           "messages": [ToolMessage(body, tool_call_id=runtime.tool_call_id)]})
```

The final state showed `loaded_skills == ["market-questions"]`.

## Q7: checkpoint tables

`AsyncPostgresSaver.setup()` creates `checkpoint_blobs`, `checkpoint_migrations`, `checkpoint_writes` and `checkpoints`, all in `public`.

## Q8: interrupt and resume across processes

- In process 1, `ainvoke` returned `{"messages": ..., "__interrupt__": [Interrupt(value={...}, id=...)]}`.
- `(await agent.aget_state(cfg)).interrupts` held the same `Interrupt`, and `next == ("tools",)`.
- Process 2 built a new saver and a new agent, then called `ainvoke(Command(resume={...}), cfg)`. The tool received the value, and the model continued.
- **The tool body re-ran from the start on resume** ("tool body started" printed in both processes). So anything before `interrupt()` must be idempotent.

## Q9: `runtime.context` in middleware

With `create_agent(..., context_schema=Ctx)` and `ainvoke(..., context=Ctx(turn_id="u1"))`, `request.runtime.context` held `Ctx(turn_id='u1')` in both `awrap_model_call` and `awrap_tool_call`.

## Q10: `Qwen/Qwen3-Embedding-0.6B-Q` through fastembed

| Measure | Result |
|---|---|
| First load, including the 1.1 GB download into `.cache/fastembed` | 48.6 s |
| Embedding 500 short texts | 8.2 s, about 60 per second |
| Output | 1024 dimensions, L2-normalized |

- **`query_embed` adds no instruction.** It equals `passage_embed` for the same text, so the query prefix is added by hand:
  `PREFIX = "Instruct: Given a question about the London office market, retrieve relevant news passages\nQuery:"`
- **Chinese query against English passages:** 「國王十字有沒有人工智能公司租寫字樓？」

  | Query form | AI-letting passage | Retail-park passage |
  |---|---|---|
  | Without the prefix | 0.659 | 0.351 |
  | With the prefix | 0.669 | 0.321 |

  The prefix separates the two slightly better, so queries use it.
