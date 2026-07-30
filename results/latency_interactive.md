# NFR1 — interactive latency

10 in-scope questions spread across intents, each separated by a 12s pause to approximate a student reading the previous answer before asking the next. Model `llama-3.1-8b-instant`.

| Statistic | Seconds |
|---|---|
| Median | 1.18 |
| Mean | 1.17 |
| Min | 0.68 |
| Max | 1.87 |

Per-question: 1.45, 0.74, 0.68, 1.03, 1.31, 1.10, 1.87, 1.27, 0.79, 1.43

Compare with the burst figure in `eval_results.md`, measured while the harness issues 49 questions plus 38 judge calls with no pause. The difference between the two is provider throttling under sustained load, not model or pipeline speed — risk R3 in the register.
