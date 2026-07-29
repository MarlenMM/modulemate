# NFR1 — interactive latency

10 in-scope questions spread across intents, each separated by a 12s pause to approximate a student reading the previous answer before asking the next. Model `llama-3.1-8b-instant`.

| Statistic | Seconds |
|---|---|
| Median | 0.63 |
| Mean | 0.64 |
| Min | 0.58 |
| Max | 0.72 |

Per-question: 0.62, 0.58, 0.64, 0.72, 0.63, 0.63, 0.68, 0.66, 0.65, 0.63

Compare with the burst figure in `eval_results.md`, measured while the harness issues 49 questions plus 38 judge calls with no pause. The difference between the two is provider throttling under sustained load, not model or pipeline speed — risk R3 in the register.
