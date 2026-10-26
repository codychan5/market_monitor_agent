# Model choices

Every job in the proof of concept that uses a model was given to three candidate models on the same cases. This page shows the evidence for each job and what it supports. Runs were made from 24 to 26 Oct 2026, and the reports are in `evals/results/`. Total evaluation spend across every run is $4.00, under the $5 cap.

The language models run through OpenRouter, because Anthropic and the OpenAI API do not serve Hong Kong. The three candidates come from three labs: GLM-5.3 from Zhipu, DeepSeek V4.1 Flash from DeepSeek, and Kimi K3 from Moonshot. The judge's candidates exclude the labs whose models write the cards. The embedding models run locally.

## Summary

| Job | Models compared | Evidence | Default today | What the evidence supports |
|---|---|---|---|---|
| Answering questions | GLM-5.3, DeepSeek V4.1 Flash, Kimi K3 | 8 chat cases × 2 runs | GLM-5.3 | **DeepSeek**: 16/16, at a quarter of GLM's cost |
| Writing the weekly brief | the same three | 3 briefs each: code measures and a blind review | GLM-5.3 | **Kimi** wrote the best briefs; GLM ranked last in 2 of 3 sets |
| Running the scan (choosing and ranking leads) | the same three | 2 scans each | GLM-5.3 | A tie: all 2/2. DeepSeek is cheapest |
| Investigating leads | the same three | 16 scans in all, plus 6 in which DeepSeek investigated for each orchestrator | DeepSeek | **DeepSeek**: as reliable as GLM at half the cost. Kimi is weakest |
| Checking cards (judge) | Qwen3.7 Plus, Kimi K3, MiniMax M3 | 15 labelled cards × 2 | Qwen3.7 Plus | **Kimi**: let no wrong card through and rejected no sound one, but returned no verdict on 2 of 30 reviews |
| Search embeddings | Qwen3-Embedding-0.6B, EmbeddingGemma-300m, MiniLM | 18 questions, each asked in English and Chinese | Qwen3-Embedding-0.6B | **Qwen3-Embedding-0.6B**: best, and no migration needed |

**The defaults have not been changed.** The evidence points away from two of them: GLM-5.3 as orchestrator and Qwen3.7 Plus as judge. All live testing of the app ran on the current defaults, so switching is a decision for the owner. See [What to change](#what-to-change).

## 1. Answering questions

The orchestrator model answers chat questions. The 8 chat cases cover:

- a single figure
- brokers disagreeing
- "average them"
- a figure that is not in the data
- the same question in Chinese
- a past decision
- a trend
- an off-topic request

Each case was run on 25 and 26 Oct and graded by code.

| Model | Passed | Mean cost per question | Typical time |
|---|---|---|---|
| GLM-5.3 (default) | 14/16 | $0.0071 | 10 s |
| **DeepSeek V4.1 Flash** | **16/16** | **$0.0017** | **4 s** |
| Kimi K3 | 13/16 | $0.0236 | 11 s |

- **GLM failed one case on each day.** On 25 Oct it left a broker out of "average them". On 26 Oct it answered "do the brokers agree" without the broker comparison tool.
- **Kimi failed three times.** It answered "do the brokers agree" without the comparison tool on both days, and left a broker out of "average them" once.
- **Every model loaded the right playbook on every case that checks routing.**
- **Why DeepSeek, not GLM:** it passed every case on both days, and it is four times cheaper and twice as fast. GLM was chosen before any evaluation ran, on the strength of an earlier project; this evidence does not support keeping it.

## 2. Writing the weekly brief

The orchestrator also writes the weekly brief. Every brief from all three models passed the evaluation's checks:

- every figure verified and every citation resolving
- the five required headings present
- the right playbook and tools used

None of those checks looks at what the brief says. Two further comparisons did:

**In code** (`lom eval-briefs`), across all 9 briefs:

| Model | Figures cited | Stories cited | Rates covered (of 2) | Portfolio assets named (of 5) | Words | Cost per brief |
|---|---|---|---|---|---|---|
| GLM-5.3 | 11.3 | 6.3 | 2.0 | 5.0 | 408 | $0.030–0.047 |
| DeepSeek V4.1 Flash | 10.0 | 10.7 | 2.0 | 5.0 | 445 | $0.005–0.012 |
| Kimi K3 | 10.0 | 7.7 | 2.0 | 4.7 | 487 | $0.052–0.083 |

**A blind review.** The three briefs from each run were shuffled and labelled A, B and C. A reviewer then scored them on five criteria: insight, portfolio relevance, story selection, clarity and care. The reviewer was a separate Claude agent from a lab outside the candidates, and it did not know which model wrote which brief.

| Run | 1st | 2nd | 3rd |
|---|---|---|---|
| 24 Oct | Kimi | GLM | DeepSeek |
| 25 Oct | DeepSeek | Kimi | GLM |
| 26 Oct | Kimi | DeepSeek | GLM |
| **Mean rank (score of 25)** | **Kimi 1.3 (16.7)** | **DeepSeek 2.0 (15.7)** | **GLM 2.7 (14.3)** |

- **Kimi writes the best briefs.** It picks the stories that move City values and links them to named assets.
- **GLM wrote the weakest brief of the nine.** In the 26 Oct run it presented the synthetic distress items as live deals to screen.
- **Why not Kimi for everything:** the brief is one job of the orchestrator, and Kimi is the weakest at answering questions. Using a different model just for the brief would need a per-playbook model setting, which is not built. On briefs, DeepSeek ranks second, ahead of GLM.
- **A product fix the review found:** several briefs used synthetic items as if they were real. The `market-brief` playbook should require any synthetic item it cites to be labelled.

## 3. Running the scan

The orchestrator chooses up to five leads, has them investigated, ranks the cards and writes the report. Until 26 Oct only GLM had done this. Each scan case now has a twin that tests the orchestrator (`s1_scan_orchestrator`, `s2_injection_orchestrator`), with the investigators held at DeepSeek.

| Model | Passed (planted leads; prompt injection) | Cost per scan | Typical time |
|---|---|---|---|
| GLM-5.3 (default) | 2/2 | $0.051 | 88 s |
| DeepSeek V4.1 Flash | 2/2 | $0.046 | 105 s |
| Kimi K3 | 2/2 | $0.080 | 83 s |

- **All three chose both planted leads and ignored the injected instruction.** This is one run per case, so it shows that any of the three can run a scan, not which one is best.
- **This removes the remaining reason to keep GLM as orchestrator.** The worry was that DeepSeek had never run a scan.

## 4. Investigating leads

Investigators are sub-agents that check one lead each, in parallel. The orchestrator was held at GLM, and the judge at Qwen.

- **The planted-leads scan (`s1_scan`)** must surface an insolvency and an AI letting and discard two off-mandate decoys.
- **The injection scan (`s2_injection`)** must ignore an instruction planted in a news item.

| Investigator | Injection scan | Insolvency lead found | AI letting found | Cost per scan |
|---|---|---|---|---|
| GLM-5.3 | 2/2 | 2/4 | 3/3 | $0.112–0.133 |
| **DeepSeek V4.1 Flash** (default) | 2/2, and 3/3 in the twins | 5/7 | 5/5 since the judge fix | **$0.055–0.086** |
| Kimi K3 | 2/2 | 1/2 | 1/2 | $0.154–0.220 |

The counts include every valid run on 24 and 26 Oct.

- **DeepSeek's counts include 3 runs from the orchestrator twins**, in which DeepSeek investigated for each of the three orchestrators.
- **Two AI-letting results from 26 Oct are excluded.** A change made by another session that night skips any lead already carded in the same database. The eval runs all its scans in one database, so later scans skipped the letting (more in [Limits](#limits-of-the-evidence)).

Why each model missed a planted lead differs, and it matters more than the counts:

- **GLM**: twice its investigation of the insolvency lead "stopped without a result". Both runs contain an investigator reply that stopped at exactly the 8,192-token output cap (`LOM_LLM_MAX_OUTPUT_TOKENS`). This is the failure the owner's earlier project traced to GLM-5.3 spending its whole output budget on reasoning; it raised its cap to 16,384.
- **DeepSeek**: twice the judge rightly rejected its insolvency card.
  - Once it listed Example House, which Nan Fung does not own, as an affected asset.
  - Once it wrote "declined at £68m" while citing the section of the investment-committee note that does not contain that figure.
- **Kimi**:
  - Once it kept sending made-up argument names to the search tool (`source_types_array`, `source_types[]`). The data server refused each one, until the investigation ran out of time.
  - It also missed the insolvency lead on 24 Oct.

**Why DeepSeek stays:** it finds the planted leads at least as often as GLM, at half the cost. Its misses were caught by the judge rather than lost silently. GLM's misses are likely to shrink if the output cap is raised, which also matters because GLM is DeepSeek's fallback.

## 5. Checking cards (the judge)

A second model checks every card against its evidence before the card is stored. `lom eval` cannot test this, because a judge's mistakes in a scan show only through which cards survive. So `lom eval-judge` sends 15 hand-labelled cards through the production review call:

- **7 sound cards.**
- **8 cards, each with one planted error:** a wrong figure, the wrong company, a wrong fact about Nan Fung's own asset, an invented price, the wrong broker credited, an overstated event, a misquoted broker figure, and a misread internal decision.

The cards were written by hand, so no candidate reviews its own lab's writing. Their evidence and context are frozen exactly as production shows them to the judge. The candidates come from labs that do not write cards in production: the investigator is DeepSeek and its fallback is GLM.

| Judge | Correct | Wrong cards let through | Sound cards rejected | No verdict returned | Same verdict both times | Cost per card |
|---|---|---|---|---|---|---|
| Qwen3.7 Plus (default) | 28/30 | 2/16 | 0/14 | 0 | 15/15 | $0.0007 |
| **Kimi K3** | **28/30** | **0/16** | **0/14** | 2 | 13/15 | $0.0112 |
| MiniMax M3 | 25/30 | 2/16 | 3/14 | 0 | 14/15 | $0.0004 |

- **Qwen** let the forecast credited to the wrong broker (CBRE instead of Savills) through both times.
- **MiniMax** let an invented £40m asking price through both times, and rejected three sound cards.
- **Kimi** caught every planted error and rejected no sound card. Twice it returned no verdict. In the product, a review without a verdict counts as a failed lead, so the lead is lost.
- **Why Kimi, not Qwen:** a wrong figure or source reaching the team list is the costliest mistake the judge can make. Kimi made none, for about $0.05 a scan.
  - Its missing verdicts need a fix first: retry once, or fall back to Qwen, a lab that also writes no cards.
  - Kimi should also leave the investigator comparison, so that the judge never reviews its own lab's cards. It came last there anyway.

## 6. Search embeddings

The embedding model turns news, notices and internal documents into vectors for search, for de-duplicating stories and for matching leads to the mandate. `lom eval-embedders` embeds the corpus exactly as `lom setup` stores it: 197 documents and 223 passages. It then asks 18 questions, each once in English and once in Traditional Chinese, and checks where each question's answer ranks. Each model gets the prefixes its authors specify.

| Model | Answer ranked 1st (English / Chinese) | In the top 5 (English / Chinese) | MRR | Time | Dimensions | Licence |
|---|---|---|---|---|---|---|
| **Qwen3-Embedding-0.6B** (default) | 100% / 89% | 100% / 94% | 0.96 | 37 s | 1024 | Apache 2.0 |
| EmbeddingGemma-300m | 89% / 89% | 100% / 94% | 0.91 | 11 s | 768 | Gemma terms of use |
| MiniLM (multilingual) | 78% / 50% | 94% / 83% | 0.76 | 7 s | 384 | Apache 2.0 |

MRR is the mean reciprocal rank of the first right answer.

- **Why Qwen:** it ranks the right answer first more often in English, and ties Gemma in Chinese.
- **Gemma is the fallback option** if speed matters: it embeds three times faster. Switching would need a schema migration from 1024 to 768 dimensions and a legal check of Gemma's terms.
- **MiniLM is rejected.** It misses half the Chinese questions.
- **Switching embedders is not just a setting.** The production embedder adds Qwen's query instruction to every question whatever model is configured, and the schema fixes vectors at 1024 dimensions.

## 7. Fallbacks

Each role's fallback is the other evaluated model from a different lab, on other hosts:

- **The orchestrator's fallback** is DeepSeek. It would become GLM if DeepSeek became the orchestrator.
- **The investigator's fallback** is GLM.
- **Kimi is no one's fallback** because of its cost and its tool-argument quirk.

A fallback takes over only on timeouts, connection errors, server errors and rate limits. It does not take over on a bad key or exhausted credit, because it would fail too. The evaluation runs with fallbacks off, so every result above measures one model. The handover itself has never happened in a live run: the main database's telemetry shows no fallback has ever answered a call.

## What to change

These are the owner's decisions; none has been applied.

| Change | Why | Cost | Risk |
|---|---|---|---|
| Orchestrator: GLM-5.3 → DeepSeek V4.1 Flash, with GLM as its fallback | Best at answering questions (16/16), as good at running scans, second-best briefs, a quarter of the cost | One setting | The app's live testing ran on GLM; worth one more pass of the demo questions |
| Judge: Qwen3.7 Plus → Kimi K3, with a retry or a Qwen fallback when no verdict comes back | Let no wrong card through, against Qwen's 2/16 | About $0.05 more a scan, and a small code change for the retry | Without the retry, about 1 lead in 15 is lost |
| Raise the output cap from 8,192 to 16,384 tokens, and handle a reply cut off at the cap | GLM investigators hit the cap and lost the insolvency lead | One setting; larger credit reservations per request | None known |
| Show the judge each passage's section heading and date | A condition such as "Revisit if: the owner is forced to sell" reaches the judge as a fact | Small code change in `_evidence_texts` | Judge prompts get slightly longer |
| Keep DeepSeek as investigator and Qwen3-Embedding-0.6B as embedder | Best on the evidence | None | None |

## Limits of the evidence

- **Small numbers.** Most cases ran once or twice, so a difference of one run is noise. These results show which models are good enough for which job; they are not a benchmark.
- **Synthetic scan items.** The planted leads are labelled synthetic items, because the real snapshot had no London office insolvency or AI letting that week.
- **Hand-written test sets.** The judge cards and the retrieval questions were written by Claude, the assistant that built this. The blind review of briefs was also done by a Claude agent. None of these comes from a candidate lab, but they are one writer's view.
- **Code grading has limits.** The chat and scan cases are graded by code. A playbook change can move results; one did on 24 Oct, when a fix to `market-questions` changed the "average them" results.
- **One run was disturbed by another change.** A change made on 26 Oct at 02:05 Hong Kong time skips any lead already carded in the same database. The 02:08 repeat run was affected, and its skipped leads are excluded above. The skip has since been turned off in evaluation runs, so scan cases no longer depend on the order they run in.
- **The investigator playbook changed after these runs.** The `investigate-signal` playbook (plain-English cards, a card `kind`, numbered next steps) changed on 26 Oct after the last scan run, so the investigator results above predate it.

## Re-run it

```
uv run lom eval --models openrouter:z-ai/glm-5.3,openrouter:deepseek/deepseek-v4.1-flash,openrouter:moonshotai/kimi-k3
uv run lom eval-judge --models openrouter:qwen/qwen3.7-plus,openrouter:moonshotai/kimi-k3,openrouter:minimax/minimax-m3 --repeats 2
uv run lom eval-embedders
uv run lom eval-briefs
```

- **`lom eval` and `lom eval-judge`** stop before a spending cap (`--budget-usd`, at most `LOM_EVAL_BUDGET_USD`).
- **`lom eval-embedders`** runs locally and costs nothing. Each model downloads on first use (about 1.5 GB for the two alternatives).
- **`lom eval-briefs`** reads the saved reports and calls no model.

| Report | Files in `evals/results/` |
|---|---|
| Chat runs | `20261025T143050Z`, `20261025T143155Z`, `20261025T174227Z` |
| Scan runs | `20261024T114708Z`, `20261025T180232Z`, `20261025T181545Z` |
| Judge | `judge-20261025T180620Z` (the earlier `judge-20261025T175004Z` used three cards whose labels were later corrected) |
| Embeddings | `embedders-20261025T175718Z` |
| Briefs | `briefs-20261025T175948Z` |
