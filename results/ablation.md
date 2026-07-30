# Retrieval ablation

52 in-scope gold questions, retrieval only (no generation). Each row is the full pipeline with one stage switched off, so the difference between rows is that stage's contribution.


## Stage contributions

| Configuration | exact hit@3 | exact hit@5 | document hit@3 | MRR@5 |
|---|---|---|---|---|
| dense only (baseline) | 0.788 | 0.827 | 0.904 | 0.667 |
| dense + authority prior | 0.846 | 0.865 | 0.962 | 0.739 |
| dense + authority + BM25 (C1) | 0.750 | 0.865 | 0.942 | 0.719 |
| dense + authority + cross-encoder (C2) | 0.923 | 0.942 | 1.000 | 0.864 |
| all stages (C1 + C2) | 0.923 | 0.942 | 1.000 | 0.864 |

**exact hit@k** requires the precise (document, slide/page) pair named in the gold set. **document hit@3** requires only the right document, and is the fairer measure where the corpus answers a question correctly in more than one place. **MRR@5** is the mean reciprocal rank of the expected passage within the top 5, so it rewards moving the right passage up even when it was already retrieved.

