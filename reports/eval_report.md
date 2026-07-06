# Task 1 — Evaluation: Multimodal vs. Text-only RAG

- **Embedder:** `openai`  |  **VLM (captioner):** `ollama`
- **Queries:** 18 labelled (11 flagged image-answer)
- **Relevance:** judged at document level via `relevant_doc_ids`.

> With `mock` backends the numbers are illustrative of pipeline behaviour; run with an OpenAI key + Ollama for graded semantic results. The shape — baseline blind to image-only docs — holds either way.

## Overall (all queries)
| Metric   | Multimodal | Text-only |
|----------|-----------|-----------|
| R@1      |  100.0% |   50.0% |
| R@3      |  100.0% |   50.0% |
| R@5      |  100.0% |   50.0% |
| MRR      |    1.00 |    0.50 |
| nDCG@5   |    1.00 |    0.50 |
| lat p50  |    335ms |    360ms |
| lat p95  |    938ms |    543ms |

## Visual-answer subset (answer lives in a chart/diagram/image)
_11 queries — the subset that should favour multimodal._

| Metric   | Multimodal | Text-only |
|----------|-----------|-----------|
| R@1      |  100.0% |   18.2% |
| R@3      |  100.0% |   18.2% |
| R@5      |  100.0% |   18.2% |
| MRR      |    1.00 |    0.18 |
| nDCG@5   |    1.00 |    0.18 |
| lat p50  |    356ms |    356ms |
| lat p95  |    938ms |    543ms |

## Per-query results

| Query | image? | MM top-3 | Baseline top-3 | MM R@1 | BL R@1 |
|-------|:------:|----------|----------------|:------:|:------:|
| What is the company password policy? | — | security_policy, remote_work_policy, onboarding_guide | security_policy, remote_work_policy, onboarding_guide | 1 | 1 |
| How do I get set up as a new employee  | — | onboarding_guide, remote_work_policy, security_policy | onboarding_guide, remote_work_policy, security_policy | 1 | 1 |
| How many days per week can I work remo | — | remote_work_policy, security_policy, onboarding_guide | remote_work_policy, security_policy, onboarding_guide | 1 | 1 |
| Show me the quarterly revenue report. | — | revenue_report, expense_breakdown, product_roadmap | revenue_report, expense_breakdown, headcount_trend | 1 | 1 |
| Which quarter had the highest revenue? | ✅ | revenue_report, product_roadmap, expense_breakdown | revenue_report, expense_breakdown, headcount_trend | 1 | 1 |
| What is the largest departmental expen | ✅ | expense_breakdown, revenue_report, salary_table | expense_breakdown, revenue_report, arxiv_gpt3 | 1 | 1 |
| Which component connects directly to t | ✅ | network_architecture, error_screenshot, wiki_internet_map | arxiv_vit, arxiv_resnet, arxiv_attention | 1 | 0 |
| Show the employee salary / compensatio | ✅ | salary_table, expense_breakdown, org_chart | expense_breakdown, headcount_trend, revenue_report | 1 | 0 |
| Who reports to the CTO in the org char | ✅ | org_chart, expense_breakdown, security_policy | revenue_report, headcount_trend, onboarding_guide | 1 | 0 |
| Which server is overloaded with high C | ✅ | server_dashboard, network_architecture, error_screenshot | arxiv_resnet, arxiv_vit, arxiv_word2vec | 1 | 0 |
| What database error caused the outage? | ✅ | error_screenshot, server_dashboard, network_architecture | security_policy, arxiv_resnet, arxiv_word2vec | 1 | 0 |
| How has headcount grown over the last  | — | headcount_trend, revenue_report, org_chart | headcount_trend, revenue_report, remote_work_policy | 1 | 1 |
| Find the paper that introduced the Tra | — | arxiv_attention, arxiv_vit | arxiv_attention, arxiv_vit, arxiv_bert | 1 | 1 |
| Which paper is about deep residual lea | — | arxiv_resnet, arxiv_vit | arxiv_resnet, arxiv_vit | 1 | 1 |
| Show me the visualization of internet  | ✅ | wiki_internet_map, server_dashboard, network_architecture | arxiv_resnet, arxiv_attention, arxiv_vit | 1 | 0 |
| Which chart shows transistor counts do | ✅ | wiki_moores_law, revenue_report, headcount_trend | headcount_trend, arxiv_attention, arxiv_resnet | 1 | 0 |
| Find the labelled diagram of DNA struc | ✅ | wiki_dna, arxiv_attention, arxiv_gpt3 | arxiv_gpt3, arxiv_word2vec, arxiv_attention | 1 | 0 |
| Which document shows historical sunspo | ✅ | wiki_sunspots, remote_work_policy, wiki_moores_law | arxiv_gpt3, arxiv_vit, arxiv_attention | 1 | 0 |

## Discussion — where multimodal helped (and where it didn't)

- On the **visual-answer subset**, multimodal Recall@1 is **100%** vs the baseline's **18%** (**+82 pt**). The baseline is *structurally blind* to standalone image documents — no extractable text — so it cannot retrieve them at any K.
- On **pure-text queries** the pipelines **tie**: the visual signal adds nothing and multimodal pays extra ingest latency (VLM captioning) for no gain.
- On **mixed docs** the baseline can still find the *document* via prose keywords even when it cannot answer the visual question — so retrieval metrics *understate* multimodal's value for answering.

### Failure modes (observed during development)

- **Context-window overflow on high-res images** (observed): a large uploaded screenshot exceeded Ollama's 4096-token context (its vision tokens alone were ~4.1k) → caption failed. Fix shipped: `num_ctx=8192` + downscale images to ≤1024px before captioning; the app now surfaces a clear error instead of silently searching junk.
- **Embedding confusability** (observed): 'which server is overloaded' matched an error-screenshot caption (also about servers) above the dashboard. The optional Qwen-VL rerank fixes it (#3 → #1) at ~15 s/image.
- **Caption hallucination**: the VLM may assert values not in the image, poisoning the index. Mitigation: low temperature + a verify/rerank pass.
- **No OCR in the baseline** (deliberate, to isolate the multimodal signal); in production OCR + captions are complementary.
- **Latency**: live VLM captioning dominates *ingest* time; query latency is embedding-only.

_Regenerate: `python evaluate.py`._