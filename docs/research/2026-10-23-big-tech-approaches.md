# How big tech and financial institutions build research and monitoring agents

Research notes gathered on 2026-10-23 by a research sub-agent using web search; figures carry their source and date inline, and items marked unverified were not confirmed against a primary source.

## 1. Big tech research and monitoring agents

The vendors converge on one design. A planner sends parallel search and read sub-agents off to run in the background, a synthesis pass adds citations, data comes in through MCP connectors, and a human approves either the plan or the output.

- **OpenAI.** Deep research now runs on GPT-5.2, can use "any MCP server or app", and can be restricted to trusted sites ([10 Feb 2026](https://openai.com/index/introducing-deep-research/)). Pulse was retired and folded into rebuilt Scheduled Tasks ([17 Jun 2026](https://help.openai.com/en/articles/6825453-chatgpt-release-notes)). Workspace agents add schedules and memory ([22 Apr 2026](https://9to5mac.com/2026/04/22/openai-updates-chatgpt-with-codex-powered-workspace-agents-for-teams/)). ChatGPT agent was reportedly withdrawn in Aug 2026 (unverified). Agent Builder and Evals shut on 30 Nov 2026, and OpenAI points users to the code-first Agents SDK ([AgentKit page, updated 3 Jun 2026](https://openai.com/index/introducing-agentkit/)).
- **Google.** Deep Research and Deep Research Max combine web search, remote MCP and file search; web access can be switched off, and the user approves the plan before it runs ([21 Apr 2026](https://blog.google/innovation-and-ai/models-and-research/gemini-models/next-generation-gemini-deep-research/)). Gemini Enterprise for Financial Services (preview) has a Financial Research agent that gives confidence scores, its methodology and "data snapshots" for audit. It ships with 50+ skills and 13 data connectors ([25 Aug 2026](https://www.googlecloudpresscorner.com/2026-08-25-Google-Cloud-Launches-Gemini-Enterprise-for-Financial-Services)).
- **Microsoft.** Researcher runs deep research over the firm's Microsoft 365 data; Analyst reasons and runs Python. In "Critique" mode GPT drafts and Claude reviews; in "Council" mode both draft and a judge model compares them. Microsoft's own DRACO tests claim about 13.8% over the next-best system ([30 Mar 2026](https://www.computerworld.com/article/4152323/microsoft-adds-multi-model-ai-to-copilot-researcher-raising-accuracy-stakes.html)).
- **Anthropic.** A skill is a folder with a SKILL.md file. It loads in three steps (progressive disclosure) and can bundle scripts for deterministic work ([16 Oct 2025](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)). Ten finance agent templates each bundle skills, connectors (FactSet, CapIQ, MSCI, PitchBook, LSEG) and subagents. Managed Agents add an audit log, and humans approve before anything is acted on ([5 May 2026](https://www.anthropic.com/news/finance-agents)).
- **Perplexity.** Labs runs multi-tool jobs of 10+ minutes. Computer adds scheduled tasks and 400+ connectors ([help centre](https://www.perplexity.ai/help-center/en/articles/13901210-computer-for-enterprise)).
- **Amazon.** AgentCore is generally available, with runtime, memory, an MCP gateway, identity and observability ([13 Oct 2025](https://aws.amazon.com/about-aws/whats-new/2025/10/amazon-bedrock-agentcore-available)). Quick Research mixes company data, the web and S&P/FactSet/IDC data, with citations ([26 Nov 2025](https://aws.amazon.com/about-aws/whats-new/2025/11/amazon-quick-research-third-party-industry-intelligence)).

## 2. Financial institutions

- **Bloomberg ASKB** went generally available on [15 Oct 2026](https://www.prnewswire.com/news-releases/the-next-evolution-of-askb-your-insights-bloombergs-intelligence-one-conversation-302909392.html). Clients' own chats, worksheets and files become permissioned sources, cited to the passage. Bloomberg's CTO calls evals "make-or-break": domain experts write the benchmarks, humans grade the hard cases, and each task goes to the cheapest model that is good enough. Always-on monitoring is next ([Fortune, 28 Apr 2026](https://fortune.com/2026/04/28/bloomberg-askb-ai-agents-lessons-from-bloomberg-cto-shawn-edwards-eye-on-ai/)).
- **Morgan Stanley** ran evals (translation, summarisation, comparison with experts) before rollout, and 98% of advisor teams now use the assistant ([OpenAI case study](https://openai.com/index/morgan-stanley/)). AskResearchGPT searches about 70k reports a year and drafts emails ([Oct 2024](https://www.morganstanley.com/press-releases/morgan-stanley-research-announces-askresearchgpt)).
- **BlackRock Aladdin Copilot** lets domain teams register their own tools in a shared plugin registry. A LangGraph supervisor narrows thousands of tools to 20–30, because sending more than 40–50 hurts quality. Inputs and outputs pass guardrails, and LLM-judged evals run on every pull request ([2025 talk summary](https://www.zenml.io/llmops-database/agentic-ai-architecture-for-investment-management-platform)).
- **Goldman Sachs and JPMorgan** run multi-model internal platforms. Goldman cut its Hong Kong bankers' access to Claude, and Anthropic said Claude was "never officially supported" in HK ([Reuters, 28 Apr 2026](https://money.usnews.com/investing/news/articles/2026-04-28/goldman-sachs-bars-hong-kong-bankers-from-anthropic-ai-use-ft-reports)). JPMorgan reportedly did the same ([Semafor, 18 Jun 2026](https://www.semafor.com/article/06/18/2026/jpmorgan-restricts-anthropic-in-hong-kong)).
- **Real estate investors:**
  - **Blackstone's "AI CIO"**, trained on 14 years of investment-committee memos, writes a second memo on every deal but has no vote. This comes from a Bloomberg story I could only see second-hand, so it is unverified ([report](https://finance.yahoo.com/technology/ai/articles/blackstone-jon-gray-plots-once-114532301.html)).
  - **Blackstone, Brookfield and GIC** are building in-house, backing AI deployment joint ventures with Anthropic and OpenAI ([Commercial Observer, 3 Aug 2026](https://commercialobserver.com/2026/08/ai-real-estate-tech-brookfield-blackstone-proptech/)).
  - **CBRE** runs an Ellis AI research agent.
  - **Green Street's MCP server** feeds its commercial real estate data into Claude, ChatGPT and Gemini, checking each request against the client's subscription ([GA 11 Aug 2026](https://www.greenstreet.com/mcp-server-commercial-real-estate/)).
  - I found nothing published by Hines, Prologis or Hong Kong developers on research agents.

## 3. Published lessons

- **Start simple.** Use the simplest thing that works, and prefer fixed workflows to agents; agents give up speed and cost for performance ([Anthropic, 19 Dec 2024](https://www.anthropic.com/engineering/building-effective-agents)). OpenAI says to use an agent only for nuanced judgment, brittle rules or unstructured data, "otherwise, a deterministic solution may suffice". It also recommends one agent first, a baseline on the best model before downsizing, and handing to a human after repeated failures or before high-risk actions ([OpenAI, Apr 2025](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf)).
- **Multi-agent research** ([Anthropic, 13 Jun 2025](https://www.anthropic.com/engineering/multi-agent-research-system)):
  - It beat a single agent by 90.2%, but uses about 15× the tokens of a chat; token use explained 80% of the variance in results.
  - Early versions preferred SEO content farms until source-quality rules were added.
  - A separate CitationAgent places the citations.
  - Evals started with about 20 queries, scored by an LLM judge and checked by human testers.
  - Failed runs resume from checkpoints.
  - It is a poor fit when the sub-tasks depend tightly on each other.
- **Evals.** Start with 20–50 tasks drawn from real failures and use code-based graders where possible. For research agents, check that claims are supported by the sources, that key facts are covered, and that sources are authoritative. Measure consistency by requiring all repeated runs to pass ([Anthropic, 9 Jan 2026](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).
- **Google's whitepapers** say to grade the agent's sequence of steps, not just the final answer, and to use evaluator agents to check grounding. I only found second-hand summaries of these ([May 2025](https://www.marktechpost.com/2025/05/06/google-releases-76-page-whitepaper-on-ai-agents-a-deep-technical-dive-into-agentic-rag-evaluation-frameworks-and-real-world-architectures/)).
- **Citations are still weak.** In an EBU/BBC study, 45% of assistants' news answers had a significant issue and 31% had sourcing problems ([22 Oct 2025](https://www.infodocket.com/2025/10/22/bbc-largest-study-of-its-kind-shows-ai-assistants-misrepresent-news-content-45-of-the-time-regardless-of-language-or-territory/)). AI search tools gave wrong citations in over 60% of the Tow Center's tests ([Mar 2025](https://www.niemanlab.org/2025/03/ai-search-engines-fail-to-produce-accurate-citations-in-over-60-of-tests-according-to-new-tow-center-study/)).

## 4. General assistants out of the box (late 2026)

**What an analyst can already do:**
- Upload broker PDFs into projects or notebooks.
- Run deep research with MCP connectors or a restricted list of sites.
- Schedule briefs: ChatGPT Scheduled Tasks, Gemini scheduled actions, [Claude Cowork (Feb 2026)](https://x.com/claudeai/status/2026720870631354429) and Perplexity Computer.
- Use SKILL.md skills in Claude, and in ChatGPT too according to [OpenAI Academy](https://openai.com/academy/skills/) (I only saw this via search).
- Connect data sources such as Green Street.

**Limitations:**
- **No persistent, queryable data.** Memory and projects hold files and preferences, not a table of rents, vacancy and take-up with each figure's source and definition (my inference).
- **Numbers drift.** Results change between runs and brokers define terms differently, with nothing reconciling them. The best score on a finance-agent benchmark is about 64% ([Anthropic, May 2026](https://www.anthropic.com/news/finance-agents)).
- **No firm context** unless the analyst supplies it again each time.
- **Licensing.** CoStar's terms reportedly forbid exposing its content to open AI tools ([terms](https://www.costar.com/about/termsofuse); I could not check the exact wording). Paywalled trade press is out of reach.
- **Hong Kong availability.** This could block the obvious tooling, so legal should review it:
  - **Anthropic:** Hong Kong is unsupported ([list](https://www.anthropic.com/supported-countries)). Any entity more than 50% owned by a company headquartered in an unsupported region is barred "regardless of where they operate" ([4 Sep 2025](https://www.anthropic.com/news/updating-restrictions-of-sales-to-unsupported-regions)). That likely covers Nan Fung's London entities, but legal must confirm. Using Claude through AWS Bedrock is reportedly blocked by billing address (unverified).
  - **OpenAI:** its API has been blocked in Hong Kong since 9 Jul 2024. Microsoft serves eligible Hong Kong customers from Azure regions outside HK ([SCMP, 2024](https://www.scmp.com/tech/big-tech/article/3268233/microsoft-maintains-ai-services-hong-kong-openai-curbs-api-access-china)).
  - **Gemini:** the web app opened to Hong Kong users ([16 Mar 2026](https://www.caixinglobal.com/2026-03-17/google-to-open-gemini-to-all-users-in-hong-kong-102423987.html)); developer API access is unclear.
  - **Microsoft 365 Copilot:** generally available in Hong Kong since Dec 2023. Whether its Claude-backed features (such as Critique) work for Hong Kong tenants is not documented.
  - **US export risk:** in Jun 2026 the US briefly barred non-US nationals from Anthropic's most advanced models ([secondary source](https://en.wikipedia.org/wiki/Claude_Mythos)).
  - **Privacy:** the Hong Kong privacy regulator's (PCPD) checklist on staff use of generative AI applies ([31 Mar 2025](https://www.pcpd.org.hk/english/news_events/media_statements/press_20250331.html)).

## Patterns worth borrowing for a 2-day PoC
1. A fixed pipeline (ingest, extract, store, compare against last period, draft), with the agent used only for triage and synthesis.
2. One SKILL.md per routine task (broker-report extraction, market note, opportunity screen), with Python scripts doing all the arithmetic.
3. A fact store where every figure keeps its submarket, period, broker definition, document and page.
4. Parallel reader sub-agents with token caps, followed by a separate citation and critic pass.
5. A scheduled brief triggered by changes in the store against a watchlist, rather than a generic news feed.
6. Opportunity memos as a "second opinion" with no vote, and human approval before anything is shared.
7. 20–50 eval questions taken from last quarter's reports: exact match on numbers, every claim traced to a source, and all repeated runs passing.
8. A model-agnostic setup, since SKILL.md and MCP are portable. Candidate providers are Azure OpenAI and Copilot Studio; check each one's Hong Kong eligibility.

## Differences from using a general assistant directly
- A versioned, queryable London dataset with sources, instead of chat memory.
- The same question gets the same number, with brokers' definitions reconciled.
- Nan Fung's mandate, return hurdles and existing holdings are applied automatically.
- Opportunity triggers are auditable and link to their evidence.
- Evals, an audit log and approval gates.
- Control over which licensed sources are used, and a model provider Nan Fung can actually use from Hong Kong.

I wrote no files. The fetch tool did save a cached copy of the OpenAI guide PDF in its own results folder.
