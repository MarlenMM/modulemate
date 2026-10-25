# ModuleMate 2.2 validation

Validation was performed on 16 October 2026 on an Apple M4 Pro with 24 GiB memory.
The product is an English-language chatbot grounded in the user's supplied text,
across subjects. It does not supply an outside factual answer when material is
missing. Text explicitly supplied in a request can also be rewritten or organized.

## Automated and integration checks

The offline run passed **174 tests**, with five real-model tests intentionally
skipped. Those **five real-model integration tests** were run separately against
the prepared local encoder, reranker and extractive reader. Lint, formatting,
legacy lexical evaluation and the installed-environment dependency audit were
also checked. These checks are engineering evidence, not semantic accuracy.

The offline suite covers fresh and legacy workspace migration, document/course
boundaries, source deletion, citation/evidence binding, bounded numeric tools,
conversation actions, lesson continuation, output recovery and Streamlit widget
journeys. Model tests are separate so CI does not silently download weights.

The release check scans tracked files for runtime databases, private corpus paths,
private environment files and obvious keys. It does not erase or certify Git
history. No coursework or private chat output is included in the evaluation.

A real PDF was checked on a temporary copy of the local workspace. The assistant
produced a partial summary with cited pages and correctly reported the missing
seminar location. It did not strictly obey the requested five-bullet format. The
original workspace's content digest remained unchanged. This was a document-task
smoke check, not a statistics curriculum benchmark; its private output is not published.

That check exposed an actual card-copying failure. Card generation now selects
source-span IDs, so the app supplies the original answer text and removes duplicate
backs instead of asking the model to reproduce a quotation. Real local generation
then produced cards with source-matching backs. Invalid span IDs are rejected.
The final isolated Streamlit AppTest opened a fresh workspace without login,
loaded the original Learning Lab example, selected the prepared 9B model, and
completed a two-sentence comparison with two source passages. The same check
produced two distinct cards with exact source backs after deduplication. This
tested application code/widget state; it did not inspect the blocked live browser.

## Local conversation observations

The default model is `mlx-community/Qwen3.5-9B-4bit`, revision
`8b2b98c00a6b4d291155e4890773ca8f769aee53`. An optional 4B model was explored earlier;
its smaller footprint is not a quality-equivalence claim. Models run locally through
MLX LM with thinking disabled and seeded sampling. No external GPU or training on
student material was used.

The primary development fixture is `evaluation/general_documents.json`: nine
original fictional documents across seven areas. The runner stores real answers,
evidence, all model calls, latency, model identity, and source/data fingerprints.
We inspect prose as well as word/phrase constraints. A constraint pass is not an
intelligence score or proof that every implication is supported.

The first full English run passed 22/32 automated constraint checks. Manual reading
identified unnecessary archived comparisons, over-refusal, false-premise errors,
rejected missing-information citations and excessive rewrite commentary. One
constraint failure was a false positive: “cannot be determined if it will rain”
contained a forbidden substring despite correctly refusing to invent a forecast.
This run informed the subsequent corrections; it is not a blinded benchmark.

The next full run satisfied 28/32 constraints, and a later full run satisfied
30/32. These are successive development runs, not independent accuracy estimates.
In the 30/32 run, the weather failure was the same substring false positive; the
age-rule response conservatively separated registration from attendance and had a
different response type from the fixture's expectation. Manual reading still found
unsupported expansions that those checks accepted.

After the final presentation cleanup, a seven-turn focused rerun satisfied 6/7
constraints and used the same code fingerprint as the retained configuration. It
confirmed ordered comparison follow-ups, a false-premise correction, source-footer
removal and the removal of a dangling reasoning tag. Its remaining phrase failure
required the word “plant” while the answer used “seedling”; the answer also had a
more substantive attribution problem described below.

A separate four-case, writer-only trial of 512-token capped thinking took about
28–33 seconds per draft and still mishandled a false premise. A five-turn broader
critic trial approved unsupported details. Neither experiment was adopted. These
small exploratory trials are not controlled estimates of a model's general ability.

The subsequent eight-turn probe on new fictional facts satisfied 7/8 constraints.
Manual reading found the current quantity, explicit permission, optional item,
reservation condition, missing funder, author and polite rewrite consistent with
their provided text. The weight question correctly disclosed absent measurements
and described the assembly directions, but used a conservative response type.
This small probe was written after the general-set failures, so it is not a blinded
or representative accuracy estimate.

[Recorded questions, answers, actual source chunks and timings](../evaluation/observations/v2.2-conversations.json)
retain all three runs, including failures and source/data fingerprints. The full
32-turn run predates the presentation cleanup. The seven-turn recovery and eight-turn
probe match QA checkpoint commit `f300674`; the later card-only correction was
verified separately. Medians were 17.29, 16.88 and 9.61 seconds respectively, on this
machine and these small inputs; they are not a general latency guarantee.

## Boundaries and remaining limits

- Citation binding proves that a quoted span occurs in a selected source. It does
  not prove that the answer correctly applies that span or answers the question.
- A model can still over-refuse, misread a condition or attach related evidence.
  Targeted model review is an additional fallible check, not a correctness oracle.
- Actual reading caught an example attributing a mandatory move to a “sunny spot”
  to a handout that only described light and stored food reserves. A bridge answer
  also inferred a traffic category not explicitly named in its source. These
  semantic expansions survived phrase checks and the broader review experiment.
  The retained system still has this limitation; source validity is not entailment.
- Long libraries are budgeted; headings and scheduled lesson targets make partial
  coverage explicit. Image-only pages, plots and diagrams still need OCR/text.
- Bounded arithmetic uses explicit input data. It does not solve arbitrary maths,
  execute uploaded code, or verify every numeric claim in an explanation.
- Legacy intent-classifier and retrieval/excerpt benchmarks remain reproducible,
  but their old scores do not measure the generative conversation system.
- Direct Streamlit AppTest exercises application code and widget state. Live
  browser inspection was blocked by the automatic browser security policy in this
  run, so there is no new visual-browser verification claim.

See the [cycle log](INTELLIGENCE_REBUILD.md) for decisions informed by actual
failures, including removal of the earlier statistics-specific lesson experiment
and translation stage after the user's scope corrections.
