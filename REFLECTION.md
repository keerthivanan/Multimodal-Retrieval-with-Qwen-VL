# Task 3 — Written Reflection

*AI/ML Engineer Technical Assessment · Multimodal RAG (Task 1) + MCP Agent (Task 2)*

---

### 1. If you had one more week, what is the single highest-impact thing you would add or change across both tasks, and why?

**A shared, automated evaluation + observability layer that turns "it works on my examples" into a measured, regression-tested quality bar.**

- **Task 1:** grow the labelled query set from 18 to a few hundred queries (stratified by *visual-only* vs *text* answers) and add a **hybrid retriever** — page-image embeddings (CLIP/SigLIP) *alongside* the current Qwen-VL captions — then A/B the two strategies on that set. Today the pipeline is caption-only, which is strong for text-described visuals but weak for pure *visual* similarity ("find docs like this screenshot" when the match is stylistic, not textual). Measuring the hybrid against caption-only is the only honest way to decide whether the extra complexity is worth it.
- **Task 2:** add a labelled set of natural-language → expected-result pairs and score the agent on **answer accuracy** and **error-recovery success rate**, not just "the demo ran."

**Why this over anything flashier:** both tasks currently *demonstrate* correctness on a handful of hand-picked cases. The highest-leverage improvement is making quality **measurable and continuously tested**, so every future change is validated by data instead of vibes — which is exactly the "evaluation, not a demo" bar the assessment sets. Everything else (better UI, more models, faster inference) is easy to justify *once you can measure it*.

---

### 2. Name one thing in your submission you are NOT happy with, and what the correct production approach would be.

**Task 1: the multimodal index depends on VLM captions with no verification step, and it is rebuilt in memory at startup.**

- **The problem:** Qwen-VL captions are how images become searchable, but a *hallucinated* caption (e.g. asserting a chart value that isn't there) silently poisons the index — and the retriever can't tell a good caption from a confabulated one. I also rebuild the whole vector index in memory on each start, and cap PDF ingestion at N pages to bound captioning cost, so very deep figures aren't indexed.
- **The correct production approach:**
  1. **Caption verification / grounding** — cross-check each caption against OCR text and/or a second cheaper pass, and confidence-gate low-agreement captions (flag or drop them) instead of trusting every caption blindly.
  2. **Persistent, incremental vector store** — pgvector or a managed vector DB, so embeddings are computed once, versioned, and updated per-document rather than recomputed on every boot.
  3. **Async, budgeted full-document ingestion** — caption *all* pages via a queue with a cost budget and page-importance heuristics (figure/table detection), removing the hard page cap.
  4. **Human-in-the-loop feedback** — log click-throughs to fine-tune ranking and catch systematic caption errors.

Being explicit about this is the point: I made a defensible *prototype* trade-off (speed and cost over completeness), but I would not ship it as-is.

---

### 3. For Task 1, when is multimodal retrieval worth the extra cost/latency over text-only RAG, and when would you advise a team to skip it?

**It is a function of one number: the fraction of answer-bearing content that lives in non-extractable visual form.**

**Worth it when** a meaningful share of answers live *only* in pixels:
- charts/graphs where the value has no text label, diagrams/flowcharts, **scanned** documents and tables, screenshots, dashboards, slide decks, engineering drawings, medical/insurance imagery.
- Domains: financial & research reports, ops/observability, manufacturing, healthcare, any scan-heavy back office.
- My measured result quantifies the payoff: on the visual-answer subset, multimodal scored **Recall@1 ≈ 100%** vs the text-only baseline's **≈ 18%** — because the baseline is *structurally blind* to image-only documents. When that gap exists, the VLM cost is easily justified.

**Skip it (advise text-only) when** the corpus is predominantly **born-digital text** — wikis, tickets, code, emails, contracts and PDFs with selectable text. There, text-only RAG *matches* multimodal at a fraction of the cost and latency (VLM captioning is seconds/page plus GPU/API spend, versus milliseconds for text embedding). Also skip when latency SLAs are tight and content is text-heavy, or when **cheap OCR** already recovers the little visual text that exists.

**The rule I would give a team:** measure the fraction of real queries whose answer is visual-only *before* building. If it's low, multimodal is paying a large, recurring tax for a rare benefit — start text-only (optionally + OCR), and add multimodal *selectively* only for the document types that need it.
