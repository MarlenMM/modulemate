# NFR1 — interactive latency

10 in-scope questions spread across intents, each separated by a 12s pause to approximate a student reading the previous answer before asking the next. Model `llama-3.3-70b-versatile`.

| Statistic | Seconds |
|---|---|
| Median | 3.12 |
| Mean | 3.08 |
| Min | 2.33 |
| Max | 3.74 |

Per-question: 2.33, 2.63, 3.02, 2.48, 3.58, 3.23, 3.59, 3.42, 3.74, 2.74

Compare with the burst figure in `eval_results.md`, measured while the harness issues 49 questions plus 38 judge calls with no pause. The difference between the two is provider throttling under sustained load, not model or pipeline speed — risk R3 in the register.
