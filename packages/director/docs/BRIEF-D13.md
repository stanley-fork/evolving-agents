# D13: retrieval without the 12B choosing, and the 12B as a filter over a short list

**Written 2026-10-17, before anything runs.** D8–D12 found where the 12B helps and where it does not:

- It is precise as a strict filter over a short list (D8: ~0.9).
- It is weak at choosing where to go in the memory. The walk reached 0.90 of the grounding statements on one bank and
  0.47 on another, while the best trajectories covered 1.00.
- It is weak at turning rules into a plan (D11 0.68 and D12 0.35–0.39, against Sonnet's 0.87–0.92).

The user's decision on 2026-10-17 was to follow the recommendation. Retrieval moves to code (lexical + local
embeddings), the 12B only filters a short list, and the plan is written in Claude's lane. D13 measures the first two
parts. It needs no judge, so it costs US$0.

## Arms (retrieval over D9's memory, 286 statements; query = the case's situation)

| arm | what it returns |
|---|---|
| **BM25** | D8's BM25 over When + Do + Because, top k |
| **EMB** | `embeddinggemma` (local Ollama) cosine between the situation (`task: search result \| query: …`) and each statement (`title: none \| text: When … Do …`), top k |
| **HYB** | reciprocal rank fusion of BM25 and EMB (constant 60), top k |
| **HYB→F** | HYB's top 40, filtered by the 12B with D8's inclusion question in batches of 10 |

Fixed before running: k = 20, the size of Sonnet's citation set in D11 and D12 (~19); the RRF constant 60; batches of
10.

**References already on disk:**
- the S10 walk's statements (D11, D12);
- Sonnet's citations (T).

## Banks and metric

- **Banks:** D11 (n01–n12) and D12 (m01–m12), with their `grounds`.
- **Metric:** grounds recall, meaning the share of a case's grounding statements among those returned, averaged over
  cases.
- **Also reported:**
  - list size;
  - for HYB→F, the share of HYB@40's grounds that the filter keeps, and how much of the list it removes.

## Verdicts (fixed here)

| question | verdict | condition |
|---|---|---|
| retrieval | **READY** | HYB@20 grounds recall ≥ 0.85 on **both** banks |
| | **PARTIAL** | ≥ 0.70 on both |
| | **NOT READY** | otherwise |
| the filter | **USEFUL** | HYB→F keeps ≥ 0.90 of HYB@40's grounds **and** removes ≥ 30 % of the list |
| | **NOT USEFUL** | otherwise |

**Caveats declared now:**
- Both banks have been read for diagnosis (D11, D12), though no retrieval setting here was tuned on them.
- Their situations were written from the statements, which favours lexical retrieval.

**Next, only if retrieval is READY or PARTIAL:** D14 measures the plan. Sonnet gets the situation plus the retrieved
statements and no tools. Its plans are compared with T's (Sonnet walking with tools) on score and cost. It is briefed
separately.

## Outcome (2026-10-17) — retrieval **PARTIAL**, filter **NOT USEFUL**

US$0, 0 JSON failures. Grounds recall, by bank:

| | BM25@20 | EMB@20 | **HYB@20** | HYB@40 | HYB→F (12B filter on HYB@40) |
|---|---|---|---|---|---|
| D11 bank | 0.742 | 0.612 | **0.771** | 0.808 | 0.689 (9.0 statements) |
| D12 bank | 0.688 | 0.569 | **0.736** | 0.799 | 0.694 (8.7 statements) |
| for reference: S10 walk | 0.90 | | | | |
| for reference: S10 walk | 0.47 (D12 bank) | | | | |
| for reference: Sonnet's citations | 0.95 / 0.93 | | | | |

- **Retrieval is PARTIAL:** 0.77 and 0.74, against the ≥ 0.85 that READY needs.
- **Retrieval is stable across banks.** The fusion varies by 0.03 between banks; the 12B's walk varied by 0.43.
- **Fusion beats each source.** Embeddings alone are the weakest source.
- **The filter is NOT USEFUL.** It removes 78 % of the list but keeps only 0.85 of HYB@40's grounds (the floor is
  0.90).
- **Next:** D14 runs with **HYB@20**, as its brief fixed for a filter that is not useful.
